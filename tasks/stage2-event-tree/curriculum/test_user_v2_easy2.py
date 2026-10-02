"""Budget and immutable data tests for the fresh easy-two-epoch curriculum."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PARENT = HERE / 'user-v2-20261002/mean_curriculum_manifest.json'


class Easy2Tests(unittest.TestCase):
    def module(self):
        path = HERE / 'build_user_v2_easy2_config.py'
        self.assertTrue(path.exists(), 'new easy2 builder is missing')
        spec = importlib.util.spec_from_file_location('easy2_builder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_budget_exact_epochs_and_remaining_steps(self):
        module = self.module()
        self.assertEqual(module.budget(6504, 12828, 16, 4010), (407, 814, 3196))
        self.assertEqual(module.budget(17, 32, 16, 10), (2, 4, 6))
        for values in [(0, 32, 16, 10), (17, 32, 0, 10), (17, 32, 16, 4)]:
            with self.assertRaises(ValueError):
                module.budget(*values)

    def test_frozen_manifest_and_data_are_reused(self):
        module = self.module()
        before = PARENT.read_bytes()
        parent = json.loads(before)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'easy2'
            result = module.build(out)
            self.assertEqual(PARENT.read_bytes(), before)
            self.assertEqual(result['experiment'], 'C-CL-Mean-UserV2-Easy2')
            self.assertEqual([result[k] for k in ('stage1_optimizer_steps','stage2_optimizer_steps','baseline_total_optimizer_steps')], [814,3196,4010])
            self.assertEqual(result['stage1_epoch_target'], 2)
            self.assertEqual(result['effective_batch_size'], 16)
            self.assertEqual(result['initialization'], 'fresh_base_model')
            self.assertIsNone(result['init_adapter'])
            self.assertIsNone(result['resume_from_checkpoint'])
            self.assertEqual(result['root_stage1_ids'], parent['root_stage1_ids'])
            for key in ('stage1_data','stage2_data','source_train_file','dev_data','test_data','baseline_training_config'):
                self.assertEqual(result[key], parent[key])
            self.assertEqual((out/'prompt.md').read_bytes(), Path(parent['prompt_template_file']).read_bytes())
            self.assertEqual((out/'parent_manifest.json').read_bytes(), before)
            module.build(out)
            (out/'mean_curriculum_manifest.json').write_text('protected\n')
            with self.assertRaisesRegex(FileExistsError, 'different'):
                module.build(out)
            self.assertEqual((out/'mean_curriculum_manifest.json').read_text(), 'protected\n')


if __name__ == '__main__':
    unittest.main()
