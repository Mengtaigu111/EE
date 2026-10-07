"""Frozen recipe and independent sampling/feedback verification gates."""
import copy
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch


class OnlineRecipeTests(unittest.TestCase):
    def test_frozen_recipe_and_input_hashes(self):
        from online_dev_config import build_manifest, validate_manifest
        m = build_manifest()
        validate_manifest(m)
        self.assertEqual(m['total_optimizer_steps'], 3208)
        self.assertEqual(m['eval_steps'], 401)
        self.assertEqual(m['warmup_steps'], 97)
        self.assertEqual(m['init_adapter'], None)
        self.assertEqual(m['checkpoint_steps'], list(range(401, 3209, 401)))
        missing = copy.deepcopy(m)
        missing['frozen_hashes'] = {}
        with self.assertRaises(ValueError):
            validate_manifest(missing)
        for key, value in [('total_optimizer_steps', 4010), ('eval_steps', 802),
                           ('init_adapter', 'old-adapter'), ('warmup_steps', 96)]:
            wrong = copy.deepcopy(m)
            wrong[key] = value
            with self.assertRaises(ValueError):
                validate_manifest(wrong)

    def test_lock_mismatch_is_rejected(self):
        from online_dev_config import verify_locks
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input'
            p.write_text('unchanged')
            with self.assertRaises(ValueError):
                verify_locks({str(p): 'wrong'})

    def test_sampling_partial_epoch_exact_prefix(self):
        from verify_sorted_first_online_dev import audit_sampling_prefix
        ids = [str(i) for i in range(7)]
        second = torch.randperm(7, generator=torch.Generator().manual_seed(43)).tolist()
        records = [{'epoch': e, 'index': i, 'id': ids[i], 'microbatch': j,
                    'optimizer_step_before': j // 2 if e == 0 else 4 + (j - 7) // 2}
                   for j, (e, i) in enumerate([(0, i) for i in range(7)] +
                                               [(1, i) for i in second[:4]])]
        result = audit_sampling_prefix(records, ids, global_step=6, accumulation=2,
                                       max_epochs=4)
        self.assertTrue(result['passed'])
        self.assertEqual(result['epoch_counts'], [7, 4])
        bad = copy.deepcopy(records)
        bad[-1] = bad[-2]
        with self.assertRaises(ValueError):
            audit_sampling_prefix(bad, ids, global_step=6, accumulation=2, max_epochs=4)

    def test_feedback_replay_detects_wrong_lr(self):
        from verify_sorted_first_online_dev import replay_feedback
        history = []
        for i in range(1, 8):
            history.append({'step': i * 401, 'score': .405,
                            'lr_before': .0001 if i <= 4 else .00005,
                            'lr_after': .0001 if i < 4 else .00005,
                            'new_best': True, 'reduce_lr': i == 4, 'stop': i == 7,
                            'best_score': .405, 'best_step': i * 401,
                            'anchor_score': .405, 'bad_count': i - 1,
                            'eval_count': i, 'last_eval_step': i * 401})
        state = replay_feedback(history, cadence=401, initial_lr=.0001)
        self.assertEqual(state['best_step'], 2807)
        self.assertEqual(state['bad_count'], 6)
        wrong = copy.deepcopy(history)
        wrong[4]['lr_before'] = .0001
        with self.assertRaises(ValueError):
            replay_feedback(wrong, cadence=401, initial_lr=.0001)

    def test_failed_training_verification_never_evaluates_test(self):
        from run_sorted_first_online_dev import run_after_training
        with tempfile.TemporaryDirectory() as d:
            with patch('verify_sorted_first_online_dev.verify', side_effect=ValueError('bad sampling')):
                with patch('appendix_fixed_eval.evaluate_checkpoint') as test_eval:
                    with self.assertRaisesRegex(ValueError, 'bad sampling'):
                        run_after_training(Path(d), {'smoke': False}, {})
                    test_eval.assert_not_called()

    def test_smoke_dev_contains_complete_abstracts(self):
        from train_sorted_first_online_dev import smoke_dev_rows
        rows = [{'id': f'{d}-{i}', 'document_id': d} for d in ['a', 'b', 'c'] for i in range(3)]
        docs = [{'document_id': d} for d in ['a', 'b', 'c']]
        selected, gold = smoke_dev_rows(rows, docs)
        self.assertEqual(selected, rows[:6])
        self.assertEqual(gold, docs[:2])

    def test_control_score_must_equal_official_ac(self):
        from verify_sorted_first_online_dev import replay_feedback
        row = {'step': 401, 'score': .5, 'official': {'AC_f1': .4},
               'lr_before': .0001, 'lr_after': .0001, 'new_best': True,
               'reduce_lr': False, 'stop': False, 'best_score': .5, 'best_step': 401,
               'anchor_score': .5, 'bad_count': 0, 'eval_count': 1, 'last_eval_step': 401}
        with self.assertRaises(ValueError):
            replay_feedback([row])

    def test_formal_dev_cannot_be_replaced_by_subset(self):
        from online_dev_config import build_manifest
        from verify_sorted_first_online_dev import validate_dev_inputs
        m = build_manifest()
        cfg = {'dev_file': m['data_dir']+'/dev.jsonl',
               'dev_document_file': m['document_data_dir']+'/dev.jsonl'}
        from online_dev_config import sha
        cfg.update(dev_file_sha256=sha(cfg['dev_file']),
                   dev_document_sha256=sha(cfg['dev_document_file']))
        validate_dev_inputs(m, cfg, False)
        with tempfile.TemporaryDirectory() as d:
            alternate = Path(d)/'dev.jsonl'
            alternate.write_text(Path(cfg['dev_file']).read_text().splitlines()[0]+'\n')
            cfg['dev_file'] = str(alternate)
            cfg['dev_file_sha256'] = sha(alternate)
            with self.assertRaises(ValueError):
                validate_dev_inputs(m, cfg, False)


if __name__ == '__main__':
    unittest.main()
