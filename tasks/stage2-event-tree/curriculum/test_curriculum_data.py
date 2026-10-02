"""Behavior tests for immutable curriculum splits and invalid joins."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

class CurriculumDataTests(unittest.TestCase):
    def module(self):
        self.assertTrue((HERE / 'build_mean_curriculum_data.py').exists(), 'curriculum builder is missing')
        spec = importlib.util.spec_from_file_location('curriculum_data', HERE / 'build_mean_curriculum_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_rejects_duplicate_missing_and_wrong_buckets(self):
        m = self.module()
        rows = [{'id':'p', 'n_events':1, 'completion':'(ROOT (ITT))'}, {'id':'r', 'n_events':0, 'completion':'(ROOT)'}]
        valid = [{'sample_id':'p','score_mean':'0.2','bucket':'B1'}, {'sample_id':'r','score_mean':'NA','bucket':'ROOT'}]
        self.assertEqual(m.validate_assignments(rows,valid,.3)['B1'], ['p'])
        for bad in [valid + valid[:1], valid[:1], [dict(valid[0],bucket='B2'), valid[1]], [valid[0],dict(valid[1],bucket='B1')]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.validate_assignments(rows,bad,.3)
        with self.assertRaises(ValueError):
            m.validate_assignments(rows+rows[:1],valid,.3)

    def test_real_split_budget_bytes_and_controls(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = m.build(Path(tmp))
            self.assertEqual([manifest[k] for k in ['B1_count','B2_count','root_total','root_stage1_count','stage1_total_count','stage2_total_count']], [5427,5277,2124,1077,6504,12828])
            self.assertLess(abs(manifest['stage1_root_ratio']-manifest['full_train_root_ratio'])*100,.2)
            self.assertEqual(manifest['stage2_root_ratio'],manifest['full_train_root_ratio'])
            self.assertEqual([manifest[k] for k in ['baseline_total_optimizer_steps','stage1_optimizer_steps','stage2_optimizer_steps']], [4010,2005,2005])
            source = Path(manifest['source_train_file']).read_bytes()
            self.assertEqual((Path(tmp)/'mean_curriculum_stage2.jsonl').read_bytes(),source)
            source_lines = source.splitlines(keepends=True)
            originals = {json.loads(line)['id']:line for line in source_lines}
            for line in (Path(tmp)/'mean_curriculum_stage1.jsonl').read_bytes().splitlines(keepends=True):
                self.assertEqual(line, originals[json.loads(line)['id']])
            root_ids = [json.loads(line)['id'] for line in source_lines if json.loads(line)['n_events']==0]
            sampled = random.Random(42).sample(root_ids,1077)
            self.assertEqual((Path(tmp)/'mean_curriculum_root_stage1_ids.txt').read_text().splitlines(), sampled)
            for name in ['event_count','random_half']:
                control = json.loads((Path(tmp)/f'{name}_control_config.json').read_text())
                self.assertFalse(control['auto_train'])
                self.assertEqual(control['root_stage1_ids'],sampled)
            self.assertEqual(json.loads((Path(tmp)/'random_half_control_config.json').read_text())['stage1_positive_count'],5427)
            m.build(Path(tmp))
            stage2_path = Path(tmp)/'mean_curriculum_stage2.jsonl'
            stage2_path.write_bytes(b'protected existing output\n')
            with self.assertRaisesRegex(FileExistsError,'different'):
                m.build(Path(tmp))
            self.assertEqual(stage2_path.read_bytes(),b'protected existing output\n')

if __name__ == '__main__':
    unittest.main()
