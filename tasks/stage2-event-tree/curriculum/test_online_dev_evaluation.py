"""CPU tests for resident-model evaluation; fake generation is never a result."""
import json
from pathlib import Path
import random
import tempfile
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import online_dev_evaluation as evaluation

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    padding_side = 'right'
    truncation_side = 'right'
    pad_token_id = 0
    eos_token_id = 2
    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False, truncation=False)
        return [3, 4]
    def pad(self, encoded, **kwargs):
        assert self.padding_side == 'left'
        return Inputs(input_ids=torch.tensor([item['input_ids'] for item in encoded]))
    def batch_decode(self, ids, **kwargs):
        return ['(ROOT)' for _ in ids]


class Model:
    training = True
    device = torch.device('cpu')
    config = SimpleNamespace(use_cache=False, max_position_embeddings=1000)
    def train(self, mode=True):
        self.training = mode
        return self
    def eval(self):
        return self.train(False)
    def generate(self, input_ids, **kwargs):
        assert not self.training and self.config.use_cache
        assert torch.is_inference_mode_enabled()
        assert kwargs == dict(do_sample=False, max_new_tokens=319, pad_token_id=0, eos_token_id=2,
                              temperature=None, top_p=None, top_k=None)
        random.random(); np.random.random(); torch.rand(1)
        if getattr(self, 'fail', False):
            raise RuntimeError('generation failed')
        return torch.cat([input_ids, torch.tensor([[9, 2]] * len(input_ids))], dim=1)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_rows(STAGE / 'data/sentence-c-appendix-no-copy-20261002/dev.jsonl')
        document = self.rows[0]['document_id']
        self.rows = [row for row in self.rows if row['document_id'] == document]
        self.gold = read_rows(STAGE / 'data/none/dev.jsonl')
        self.gold = [row for row in self.gold if row['id'] == document]
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.prefix = Path(self.temp.name) / 'dev-step-401'
        self.gold_path = Path(self.temp.name) / 'gold.jsonl'
        self.gold_path.write_text(''.join(json.dumps(row) + '\n' for row in self.gold))

    def test_happy_restore_and_complete(self):
        model, tokenizer = Model(), Tokenizer()
        before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        metrics = evaluation.evaluate_current_model(model, tokenizer, self.rows, self.gold_path, self.prefix)
        self.assertEqual(len(metrics['official']), 18)
        self.assertEqual(metrics['scope'], 'full')
        predictions = read_rows(metrics['paths']['predictions'])
        self.assertEqual([p['id'] for p in predictions], [r['id'] for r in self.rows])
        self.assertTrue(Path(metrics['paths']['complete']).exists())
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(tokenizer.padding_side, 'right')
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        with self.assertRaises(FileExistsError):
            evaluation.evaluate_current_model(model, tokenizer, self.rows, self.gold_path, self.prefix)

    def test_exception_restore(self):
        model, tokenizer = Model(), Tokenizer()
        model.fail = True
        before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            evaluation.evaluate_current_model(model, tokenizer, self.rows, self.gold_path, self.prefix)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(tokenizer.padding_side, 'right')
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))
        self.assertFalse(Path(str(self.prefix) + '.complete.json').exists())

    def test_cuda_rng_restore_without_gpu(self):
        model, tokenizer = Model(), Tokenizer()
        model.fail = True
        fake_state = [torch.tensor([42], dtype=torch.uint8)]
        with patch.object(torch.cuda, 'is_initialized', return_value=True), \
             patch.object(torch.cuda, 'get_rng_state_all', return_value=fake_state), \
             patch.object(torch.cuda, 'set_rng_state_all') as restore:
            with self.assertRaises(RuntimeError):
                evaluation.evaluate_current_model(model, tokenizer, self.rows, self.gold_path, self.prefix)
            restore.assert_called_once_with(fake_state)

    def test_scoring_failure_restores_eval_state(self):
        model, tokenizer = Model().eval(), Tokenizer()
        model.config = SimpleNamespace(use_cache=True, max_position_embeddings=1000)
        before = torch.get_rng_state().clone()
        with patch.object(evaluation, '_score', side_effect=ValueError('score failed')):
            with self.assertRaisesRegex(ValueError, 'score failed'):
                evaluation.evaluate_current_model(model, tokenizer, self.rows, self.gold_path, self.prefix)
        self.assertFalse(model.training)
        self.assertTrue(model.config.use_cache)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(tokenizer.padding_side, 'right')

    def test_stale_modules_are_replaced(self):
        stale = {name: SimpleNamespace(__file__='/tmp/stale.py') for name in
                 ('ontology', 'linearize', 'align_events', 'evaluate', 'merge_sentence_predictions')}
        predictions = [{'id': row['id'], 'raw_output': '(ROOT)'} for row in self.rows]
        with patch.dict(sys.modules, stale):
            metrics = evaluation.score_predictions(self.rows, predictions, self.gold_path, self.prefix)
            self.assertEqual(len(metrics['official']), 18)
            for name in stale:
                self.assertEqual(Path(sys.modules[name].__file__).resolve(),
                                 (STAGE / 'evaluation-v2-20261003/scripts' / (name + '.py')).resolve())

    def test_invalid_and_limit_counts_are_preserved(self):
        model, tokenizer = Model(), Tokenizer()
        def limited(input_ids, **kwargs):
            return torch.cat([input_ids, torch.full((len(input_ids), 319), 9)], dim=1)
        with patch.object(model, 'generate', side_effect=limited), \
             patch.object(tokenizer, 'batch_decode', side_effect=lambda ids, **kw: ['broken' for _ in ids]):
            metrics = evaluation.evaluate_current_model(model, tokenizer, self.rows, self.gold_path, self.prefix)
        self.assertEqual(metrics['diagnostics']['invalid_sentences'], len(self.rows))
        self.assertEqual(metrics['diagnostics']['hit_max_new_tokens'], len(self.rows))
        self.assertEqual(len(read_rows(metrics['paths']['predictions'])), len(self.rows))

    def test_short_generation_batch_is_rejected(self):
        model, tokenizer = Model(), Tokenizer()
        with patch.object(tokenizer, 'batch_decode', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'batch ID coverage'):
                evaluation.evaluate_current_model(model, tokenizer, self.rows, self.gold_path, self.prefix)
        self.assertTrue(model.training)
        self.assertFalse(Path(str(self.prefix) + '.complete.json').exists())

    def test_context_overflow(self):
        model = Model()
        model.config = SimpleNamespace(use_cache=False, max_position_embeddings=320)
        with self.assertRaisesRegex(ValueError, 'truncation forbidden'):
            evaluation.evaluate_current_model(model, Tokenizer(), self.rows, self.gold_path, self.prefix)
        self.assertTrue(model.training)
        self.assertFalse(model.config.use_cache)

    def test_id_coverage(self):
        predictions = [{'id': row['id'], 'raw_output': '(ROOT)'} for row in self.rows]
        for bad in (predictions[:-1], predictions + predictions[:1],
                    [dict(p, id='unknown') for p in predictions], list(reversed(predictions))):
            with self.assertRaises(ValueError):
                evaluation.score_predictions(self.rows, bad, self.gold_path, self.prefix)
        with self.assertRaises(ValueError):
            evaluation.score_predictions(self.rows + self.rows[:1], predictions, self.gold_path, self.prefix)

    def test_historical_full_dev_scores(self):
        run = STAGE / 'runs/C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003/checkpoint-eval'
        rows = read_rows(STAGE / 'data/sentence-c-appendix-no-copy-20261002/dev.jsonl')
        predictions = read_rows(run / 'dev-checkpoint-2406.predictions.jsonl')
        saved = json.loads((run / 'dev-checkpoint-2406.metrics.json').read_text())
        metrics = evaluation.score_predictions(rows, predictions, STAGE / 'data/none/dev.jsonl', self.prefix)
        self.assertEqual(metrics['official'], saved['official'])
        self.assertEqual(metrics['alignment'], saved['alignment'])
        self.assertEqual(metrics['scope'], 'full')


if __name__ == '__main__':
    unittest.main()
