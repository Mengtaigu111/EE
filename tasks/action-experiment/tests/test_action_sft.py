"""Action SFT selection, source integrity and real CPU training checks."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / 'tasks/action-experiment/train_sft.py'
sys.path.insert(0, str(MODULE.parent))


def load_module():
    spec = importlib.util.spec_from_file_location('action_sft', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FeedbackTests(unittest.TestCase):
    def test_tc_then_ti_then_earlier(self):
        controller = load_module().TCFeedbackController()
        self.assertTrue(controller.update(1, {'TC_f1': .5, 'TI_f1': .8})['new_best'])
        self.assertFalse(controller.update(2, {'TC_f1': .49, 'TI_f1': 1.})['new_best'])
        self.assertTrue(controller.update(3, {'TC_f1': .5, 'TI_f1': .81})['new_best'])
        self.assertFalse(controller.update(4, {'TC_f1': .5, 'TI_f1': .81})['new_best'])
        self.assertEqual(controller.best_step, 3)

    def test_patience_halves_once_then_stops(self):
        controller = load_module().TCFeedbackController()
        controller.update(1, {'TC_f1': .5, 'TI_f1': .8})
        decisions = [controller.update(step, {'TC_f1': .5, 'TI_f1': .8}) for step in range(2, 8)]
        self.assertEqual([d['reduce_lr'] for d in decisions], [False, False, True, False, False, False])
        self.assertTrue(decisions[-1]['stop'])

    def test_tiny_improvement_selects_best_without_resetting_plateau(self):
        controller = load_module().TCFeedbackController()
        controller.update(1, {'TC_f1': .5, 'TI_f1': .8})
        controller.update(2, {'TC_f1': .5, 'TI_f1': .8})
        decision = controller.update(3, {'TC_f1': .50000001, 'TI_f1': .8})
        self.assertTrue(decision['new_best'])
        self.assertEqual(decision['bad_count'], 2)
        decision = controller.update(4, {'TC_f1': .5001, 'TI_f1': .8})
        self.assertTrue(decision['reduce_lr'])
        decision = controller.update(5, {'TC_f1': .501, 'TI_f1': .8})
        self.assertEqual(decision['bad_count'], 0)

    def test_invalid_scores_and_steps(self):
        for value in (float('nan'), -1, 2):
            with self.assertRaises(ValueError):
                load_module().TCFeedbackController().update(1, {'TC_f1': value, 'TI_f1': .8})
        controller = load_module().TCFeedbackController()
        controller.update(1, {'TC_f1': .5, 'TI_f1': .8})
        with self.assertRaises(ValueError):
            controller.update(1, {'TC_f1': .5, 'TI_f1': .8})

    def test_source_completion_and_hash(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp:
            cot = Path(temp)
            adapter = cot / 'final-adapter'
            adapter.mkdir()
            (adapter / 'adapter_model.safetensors').write_bytes(b'adapter')
            (adapter / 'adapter_config.json').write_text('{}')
            (cot / 'config.json').write_text(json.dumps({'initialization': 'fresh base + new LoRA'}))
            (cot / 'train_metrics.json').write_text(json.dumps({'status': 'completed', 'final_adapter_sha256': module.sha(adapter / 'adapter_model.safetensors')}))
            self.assertEqual(module.inspect_cot_source(adapter)['status'], 'completed')
            with self.assertRaises(ValueError):
                module.inspect_cot_source(adapter, model_path='/requested/base')
            (adapter / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': '/requested/base'}))
            self.assertEqual(module.inspect_cot_source(adapter, '/requested/base')['status'], 'completed')
            (adapter / 'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                module.inspect_cot_source(adapter)


class TinyTrainerTests(unittest.TestCase):
    def test_accumulation_tail_uses_actual_assistant_token_count(self):
        import torch
        module = load_module()
        logits = torch.tensor([[[1., 0.], [0., 1.], [1., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, 0, 1]])
        full = module.assistant_token_loss({'logits': logits}, labels)
        piece = module.assistant_token_loss({'logits': logits}, labels, num_items_in_batch=4)
        tail = module.assistant_token_loss({'logits': logits}, labels, num_items_in_batch=2)
        self.assertTrue(torch.allclose(full, 2 * piece))
        self.assertTrue(torch.allclose(full, tail))
        tail.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_masked_prompt_logits_do_not_affect_loss(self):
        import torch
        module = load_module()
        logits = torch.randn(1, 10, 5)
        labels = torch.tensor([[-100] * 8 + [2, 3]])
        expected = module.assistant_token_loss({'logits': logits}, labels)
        modified = logits.clone()
        modified[:, :7] = float('nan')
        actual = module.assistant_token_loss({'logits': modified}, labels)
        self.assertTrue(torch.allclose(expected, actual))

    def test_trainer_tail_receives_actual_group_count(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
        module = load_module()
        trainer_cls = module.make_trainer_class()
        from sorted_first_sampler import IndexedDataset, ObservedCollator
        sys.path.insert(0, str(ROOT / 'tasks/stage2-event-tree/scripts'))
        from model_utils import AssistantCollator
        counts = []
        def loss_function(outputs, labels, num_items_in_batch=None):
            counts.append(int(num_items_in_batch))
            return module.assistant_token_loss(outputs, labels, num_items_in_batch)
        rows = [{'input_ids': [1, 2, 3, 4], 'attention_mask': [1] * 4,
                 'labels': [-100, -100, 3, 4]} for _ in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trainer = trainer_cls(model=GPT2LMHeadModel(GPT2Config(vocab_size=16,
                n_embd=8, n_layer=1, n_head=1, n_positions=8)),
                args=TrainingArguments(output_dir=str(directory / 'checkpoints'),
                    use_cpu=True, per_device_train_batch_size=1,
                    gradient_accumulation_steps=3, max_steps=2, report_to='none',
                    save_strategy='no', remove_unused_columns=False, disable_tqdm=True),
                train_dataset=IndexedDataset(rows, ['a', 'b', 'c', 'd']),
                data_collator=ObservedCollator(AssistantCollator(0)),
                observation_path=directory / 'observed-order.jsonl',
                warmup_updates=1, compute_loss_func=loss_function)
            trainer.train()
        self.assertEqual(counts, [6, 6, 6, 2])

    def test_real_peft_update_selection_and_reload(self):
        import torch
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel
        from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments
        module = load_module()
        trainer_cls = module.make_trainer_class()
        from sorted_first_sampler import IndexedDataset, ObservedCollator
        sys.path.insert(0, str(ROOT / 'tasks/stage2-event-tree/scripts'))
        from model_utils import AssistantCollator
        class Tokenizer:
            def save_pretrained(self, directory):
                (Path(directory) / 'test-tokenizer.json').write_text('{}')
        torch.manual_seed(42)
        configuration = GPT2Config(vocab_size=32, n_positions=32, n_embd=16,
                                  n_layer=1, n_head=2, pad_token_id=0)
        model = get_peft_model(GPT2LMHeadModel(configuration),
            LoraConfig(r=2, lora_alpha=4, lora_dropout=0., target_modules=['c_attn'],
                       task_type='CAUSAL_LM'))
        original = {key: tensor.clone() for key, tensor in get_peft_model_state_dict(model).items()}
        rows = [{'input_ids': [1, 2, 3, 4], 'attention_mask': [1] * 4,
                 'labels': [-100, -100, 3, 4]} for _ in range(4)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model.save_pretrained(directory / 'cot-adapter')
            args = TrainingArguments(output_dir=str(directory / 'checkpoints'),
                per_device_train_batch_size=1, gradient_accumulation_steps=1,
                max_steps=3, learning_rate=.01, save_strategy='no',
                eval_strategy='no', report_to='none', use_cpu=True,
                remove_unused_columns=False, dataloader_num_workers=0,
                disable_tqdm=True, seed=42)
            trainer = trainer_cls(model=model, args=args,
                train_dataset=IndexedDataset(rows, ['a', 'b', 'c', 'd']),
                data_collator=ObservedCollator(AssistantCollator(0)),
                observation_path=directory / 'observed-order.jsonl', warmup_updates=1)
            scores = {1: (.7, .8), 2: (.6, .9), 3: (.7, .8)}
            def evaluator(model, step):
                tc, ti = scores[step]
                return {'metrics': {'TC_f1': tc, 'TI_f1': ti}}
            callback = module.make_dev_callback(trainer, evaluator, directory,
                                                Tokenizer(), [1, 2, 3],
                                                source={'source_adapter': str(directory / 'cot-adapter')})
            trainer.add_callback(callback)
            result = trainer.train()
            self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
            self.assertEqual(callback.feedback.best_step, 1)
            evidence = json.loads((directory / 'handoff_evidence.json').read_text())
            self.assertTrue(evidence['adapter_exact'])
            self.assertTrue(evidence['optimizer_state_empty'])
            changed = get_peft_model_state_dict(model)
            self.assertTrue(any(not torch.equal(original[key], changed[key]) for key in original))
            reloaded = PeftModel.from_pretrained(GPT2LMHeadModel(configuration),
                                                 directory / 'best-adapter')
            from safetensors.torch import load_file
            saved = load_file(str(directory / 'best-adapter/adapter_model.safetensors'))
            actual = get_peft_model_state_dict(reloaded)
            self.assertTrue(all(torch.equal(saved[key], actual[key]) for key in saved))
            self.assertEqual(json.loads((directory / 'best.json').read_text())['step'], 1)
            observed = [json.loads(line) for line in (directory / 'observed-order.jsonl').read_text().splitlines()]
            self.assertEqual([row['index'] for row in observed], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
