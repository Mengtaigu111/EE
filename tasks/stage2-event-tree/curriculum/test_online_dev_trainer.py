"""Real CPU LoRA training exercises online feedback at optimizer boundaries."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import (GPT2Config, GPT2LMHeadModel, TrainerCallback,
                          TrainingArguments, default_data_collator, set_seed)
from sorted_first_sampler import IndexedDataset, ObservedCollator
from online_dev_trainer import OnlineDevTrainer, validate_online_checkpoint


class EvaluationCalls(TrainerCallback):
    def __init__(self):
        self.steps = []

    def on_evaluate(self, args, state, control, **kwargs):
        self.steps.append(state.global_step)


def make_trainer(path, scores, calls, *, steps=9):
    set_seed(42)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_positions=8, n_embd=8,
        n_layer=1, n_head=1, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2,
        lora_alpha=4, target_modules=['c_attn']))
    rows = [{'input_ids': [1, 2+i%10, 3, 4], 'labels': [-100, 2+i%10, 3, 4]}
            for i in range(20)]
    args = TrainingArguments(output_dir=str(path/'checkpoints'), use_cpu=True,
        max_steps=steps, per_device_train_batch_size=1, gradient_accumulation_steps=2,
        learning_rate=.001, eval_strategy='steps', eval_steps=1,
        save_strategy='steps', save_steps=1, logging_steps=1, report_to='none',
        disable_tqdm=True, remove_unused_columns=False, label_names=['labels'])
    def evaluator(resident, step):
        assert resident is model
        calls.append((step, bool(resident.training)))
        return {f'{metric}_{part}': scores[step]
                for metric in ('TI', 'TC', 'AI', 'AC', 'EC_I', 'EC_C')
                for part in ('p', 'r', 'f1')}
    callback = EvaluationCalls()
    trainer = OnlineDevTrainer(model=model, args=args,
        train_dataset=IndexedDataset(rows, [str(i) for i in range(len(rows))]),
        eval_dataset=rows, data_collator=ObservedCollator(default_data_collator),
        observation_path=path/'observed.jsonl', dev_evaluator=evaluator,
        feedback_path=path/'feedback.jsonl', warmup_steps=1, callbacks=[callback])
    return trainer, callback


class OnlineTrainerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_bad_three_reduces_and_bad_six_stops_during_real_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            scores = {i: .5 for i in range(1, 10)}
            calls = []
            trainer, callback = make_trainer(path, scores, calls)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 7)
            self.assertEqual(callback.steps, list(range(1, 8)))
            self.assertEqual([s for s, _ in calls], callback.steps)
            records = [json.loads(line) for line in (path/'feedback.jsonl').read_text().splitlines()]
            self.assertTrue(records[3]['reduce_lr'])
            self.assertFalse(records[2]['reduce_lr'])
            self.assertAlmostEqual(records[3]['LR_after'][0], .0005)
            self.assertTrue(records[6]['stop'])
            self.assertAlmostEqual(records[6]['LR_after'][0], .0005)
            self.assertEqual(trainer.state.best_metric, .5)
            self.assertTrue(trainer.state.best_model_checkpoint.endswith('checkpoint-7'))
            state = validate_online_checkpoint(path/'checkpoints/checkpoint-7', 7, 9)
            self.assertEqual(state['global_step'], 7)
            observed = [json.loads(line) for line in (path/'observed.jsonl').read_text().splitlines()]
            self.assertEqual([r['index'] for r in observed], list(range(14)))

    def test_raw_best_including_tie_and_checkpoint_feedback_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            scores = {1: .5, 2: .5004, 3: .5004, 4: .49, 5: .48, 6: .47}
            full, callback = make_trainer(path/'full', scores, [], steps=6)
            full.train()
            self.assertTrue(full.state.best_model_checkpoint.endswith('checkpoint-3'))
            cp = path/'full/checkpoints/checkpoint-4'
            saved = json.loads((cp/'feedback_state.json').read_text())
            resumed, resumed_cb = make_trainer(path/'resumed', scores, [], steps=6)
            resumed.train(resume_from_checkpoint=str(cp))
            self.assertEqual(resumed_cb.steps, [5, 6])
            self.assertEqual(full.feedback_controller.state_dict(), resumed.feedback_controller.state_dict())
            self.assertEqual(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            self.assertTrue(resumed.resume_evidence['feedback_exact'])
            for name, parameter in full.model.named_parameters():
                self.assertTrue(torch.equal(parameter, dict(resumed.model.named_parameters())[name]), name)
            tail = [json.loads(line) for line in (path/'resumed/observed.jsonl').read_text().splitlines()]
            self.assertEqual([r['index'] for r in tail], [8, 9, 10, 11])
            self.assertEqual([r['microbatch'] for r in tail], [8, 9, 10, 11])
            self.assertEqual(saved['controller']['bad_count'], 3)
            continuity = json.loads((cp/'continuity.json').read_text())
            self.assertIn('feedback_state.json', continuity['sha256'])
            (cp/'feedback_state.json').write_text('{}')
            with self.assertRaises(ValueError):
                validate_online_checkpoint(cp)

    def test_incomplete_or_nonfinite_official_metrics_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual, _ = make_trainer(Path(tmp), {1: .5}, [], steps=1)
            actual.dev_evaluator = lambda model, step: {'AC_f1': .5}
            with self.assertRaises(ValueError):
                actual.train()
        with tempfile.TemporaryDirectory() as tmp:
            actual, _ = make_trainer(Path(tmp), {1: float('nan')}, [], steps=1)
            with self.assertRaises(ValueError):
                actual.train()


if __name__ == '__main__':
    unittest.main()
