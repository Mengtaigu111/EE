"""Real CPU PEFT/Trainer tests for curriculum checkpoint continuity."""
import importlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, default_data_collator, set_seed

try:
    curriculum = importlib.import_module('curriculum_trainer')
except ModuleNotFoundError:
    curriculum = None


def model():
    set_seed(42)
    base = GPT2LMHeadModel(GPT2Config(vocab_size=24, n_positions=16, n_embd=16,
                                    n_layer=1, n_head=2, resid_pdrop=0, embd_pdrop=0,
                                    attn_pdrop=0))
    return get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=2,
                                         lora_alpha=4, lora_dropout=.1,
                                         target_modules=['c_attn']))


def dataset(size=12):
    return [{'input_ids': torch.tensor([1, 2 + i % 12, 3, 4]),
             'labels': torch.tensor([1, 2 + i % 12, 3, 4])} for i in range(size)]


def trainer(path, stop=None, resume_new_data=False, size=12, observed=None):
    args = TrainingArguments(output_dir=str(path), use_cpu=True, max_steps=6,
                             per_device_train_batch_size=1, gradient_accumulation_steps=2,
                             learning_rate=.001, lr_scheduler_type='cosine', warmup_ratio=.03,
                             seed=42, data_seed=42, report_to='none', disable_tqdm=True,
                             save_strategy='steps', save_steps=2, logging_steps=1,
                             ignore_data_skip=resume_new_data, label_names=['labels'])
    def collate(rows):
        if observed is not None:
            observed.extend(int(row['input_ids'][1]) for row in rows)
        return default_data_collator(rows)
    return curriculum.CurriculumTrainer(model=model(), args=args,
        train_dataset=dataset(size), data_collator=collate,
        callbacks=[curriculum.PhaseStopCallback(stop)] if stop else [])


def assert_nested(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left.cpu(), right.cpu()))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_nested(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(type(left), type(right))
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_nested(test, a, b)
    else:
        test.assertEqual(left, right)


class CurriculumTrainerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(curriculum, 'safe full-state curriculum trainer is missing')
        torch.set_num_threads(1)

    def test_phase_stop_preserves_global_horizon_and_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = trainer(Path(tmp), stop=3)
            first.train()
            checkpoint = Path(tmp) / 'checkpoint-3'
            state = curriculum.validate_resume_checkpoint(checkpoint, expected_step=3,
                                                          expected_horizon=6)
            self.assertEqual(state['global_step'], 3)
            self.assertEqual(first.lr_scheduler.state_dict()['last_epoch'], 3)
            self.assertGreater(first.optimizer.param_groups[0]['lr'], 0)
            self.assertAlmostEqual(first.optimizer.param_groups[0]['lr'], .0006545084971874737)
            self.assertFalse((checkpoint / 'optimizer.pt').exists())

    def test_resume_reproduces_uninterrupted_weights_optimizer_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = trainer(Path(tmp) / 'full')
            full.train()
            full_weights = {k: v.detach().clone() for k, v in full.model.named_parameters()}
            full_rng = (random.random(), np.random.rand(), torch.rand(3))
            interrupted = trainer(Path(tmp) / 'split', stop=3)
            interrupted.train()
            resumed = trainer(Path(tmp) / 'resumed')
            resumed.train(resume_from_checkpoint=str(Path(tmp) / 'split/checkpoint-3'))
            assert_nested(self, full_weights, dict(resumed.model.named_parameters()))
            assert_nested(self, full.optimizer.state_dict(), resumed.optimizer.state_dict())
            assert_nested(self, full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())
            assert_nested(self, full_rng, (random.random(), np.random.rand(), torch.rand(3)))
            self.assertEqual(resumed.state.global_step, 6)
            self.assertTrue(resumed.resume_evidence['optimizer_exact'])
            self.assertTrue(resumed.resume_evidence['scheduler_exact'])
            self.assertTrue(resumed.resume_evidence['rng_exact'])
            self.assertEqual(resumed.resume_evidence['global_step'], 3)

    def test_changed_dataset_resumes_without_skipping_phase2_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_samples = []
            reference = trainer(Path(tmp) / 'reference', stop=3, observed=reference_samples)
            reference.train()
            first = trainer(Path(tmp) / 'first', stop=3, size=6)
            first.train()
            phase2_samples = []
            second = trainer(Path(tmp) / 'second', resume_new_data=True, size=12,
                             observed=phase2_samples)
            second.train(resume_from_checkpoint=str(Path(tmp) / 'first/checkpoint-3'))
            self.assertEqual(phase2_samples[:6], reference_samples[:6])
            self.assertEqual(second.state.global_step, 6)
            updates = [v['step'] for v in second.state.log_history
                       if 'loss' in v and v['step'] > 3]
            self.assertEqual(updates, [4, 5, 6])
            self.assertEqual(second.resume_evidence['global_step'], 3)
            self.assertTrue(second.resume_evidence['rng_exact'])
            self.assertEqual(second.resume_evidence['data_skip'], False)

    def test_corrupt_or_incomplete_checkpoint_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = trainer(Path(tmp), stop=3)
            first.train()
            checkpoint = Path(tmp) / 'checkpoint-3'
            with self.assertRaises(ValueError):
                curriculum.validate_resume_checkpoint(checkpoint, expected_step=4)
            with self.assertRaises(ValueError):
                curriculum.validate_resume_checkpoint(checkpoint, expected_horizon=7)
            metadata = checkpoint / 'optimizer.json'
            original = metadata.read_text()
            metadata.write_text(original + '\n')
            with self.assertRaises(ValueError):
                curriculum.validate_resume_checkpoint(checkpoint)
            metadata.write_text(original)
            (checkpoint / 'optimizer.safetensors').unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                curriculum.validate_resume_checkpoint(checkpoint)

    def test_model_and_trainer_state_are_covered_by_integrity_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = trainer(Path(tmp), stop=3)
            first.train()
            checkpoint = Path(tmp) / 'checkpoint-3'
            for filename in ('adapter_model.safetensors', 'adapter_config.json',
                             'trainer_state.json'):
                file = checkpoint / filename
                original = file.read_bytes()
                file.write_bytes(original + b'\n')
                with self.assertRaises(ValueError, msg=filename):
                    curriculum.validate_resume_checkpoint(checkpoint)
                file.write_bytes(original)

    def test_nonfinite_raw_loss_stops_before_logging_can_hide_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            actual = trainer(Path(tmp))
            parameter = next(p for p in actual.model.parameters() if p.requires_grad)
            with torch.no_grad():
                parameter.fill_(float('nan'))
            with self.assertRaises(FloatingPointError):
                actual.train()


if __name__ == '__main__':
    unittest.main()
