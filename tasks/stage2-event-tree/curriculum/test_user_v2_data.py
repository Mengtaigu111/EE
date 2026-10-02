"""Regression checks for exact user prompt and frozen mean curriculum memberships."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PROMPT = PROJECT / 'task/prompt/1002-event-forest-v2.md'
OLD = HERE / 'mean_curriculum_manifest.json'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class UserV2DataTests(unittest.TestCase):
    def module(self):
        path = HERE / 'build_user_v2_curriculum_data.py'
        self.assertTrue(path.exists(), 'new user-v2 curriculum builder is missing')
        spec = importlib.util.spec_from_file_location('user_v2_builder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_exact_template_and_invalid_sentence_metadata(self):
        m = self.module()
        old = json.loads(OLD.read_text())
        row = rows(old['source_train_file'])[0]
        template = PROMPT.read_text()
        expected = template.replace('[S1] <sentence 1>\n[S2] <sentence 2>\n...', '\n'.join(f'[S{i+1}] {s}' for i,s in enumerate(row['sentences']))).replace('[S{k}]','[S1]').replace('<target sentence>',row['target_sentence'])
        converted = m.convert_row(row, template)
        self.assertEqual(converted['prompt'],expected)
        self.assertEqual(converted['mode'],'sentence-user-v2')
        self.assertEqual({k:v for k,v in converted.items() if k not in ('prompt','mode')}, {k:v for k,v in row.items() if k not in ('prompt','mode')})
        for bad in [dict(row,tokens=['bad']),dict(row,target_bounds=[1,34]),dict(row,sentence_index=100),dict(row,prompt=row['prompt'].replace('[S1] program','[S2] program'))]:
            with self.subTest(bad=bad['id']), self.assertRaises(ValueError):
                m.convert_row(bad, template)

    def test_real_all_splits_payload_ids_budgets_and_no_overwrite(self):
        m = self.module()
        old = json.loads(OLD.read_text())
        before = PROMPT.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            out, data = Path(tmp)/'curriculum', Path(tmp)/'data'
            manifest = m.build(out,data)
            self.assertEqual(PROMPT.read_bytes(),before)
            self.assertEqual((out/'prompt.md').read_bytes(),before)
            self.assertEqual(manifest['prompt_template_file'],str(out/'prompt.md'))
            self.assertEqual(manifest['experiment'],'C-CL-Mean-UserV2')
            self.assertEqual(manifest['baseline_training_config'],old['baseline_training_config'])
            self.assertEqual([manifest[k] for k in ('baseline_total_optimizer_steps','stage1_optimizer_steps','stage2_optimizer_steps')],[4010,2005,2005])
            self.assertEqual(manifest['root_stage1_ids'],old['root_stage1_ids'])
            prefix = before.decode().split('## FULL ABSTRACT')[0]
            for split, source in [('train',old['source_train_file']),('dev',old['dev_data']),('test',old['test_data'])]:
                original, changed = rows(source),rows(data/f'{split}.jsonl')
                self.assertEqual(len(original),len(changed))
                for a,b in zip(original,changed):
                    self.assertEqual({k:v for k,v in a.items() if k not in ('prompt','mode')},{k:v for k,v in b.items() if k not in ('prompt','mode')})
                    self.assertTrue(b['prompt'].startswith(prefix))
                    self.assertEqual(b['mode'],'sentence-user-v2')
                    self.assertNotIn('<target sentence>',b['prompt'])
            self.assertEqual((out/'mean_curriculum_stage2.jsonl').read_bytes(),(data/'train.jsonl').read_bytes())
            for name in ('stage1','stage2','root_stage1'):
                self.assertEqual((out/f'mean_curriculum_{name}_ids.txt').read_bytes(),(HERE/f'mean_curriculum_{name}_ids.txt').read_bytes())
            self.assertEqual([r['id'] for r in rows(manifest['stage1_data'])],(HERE/'mean_curriculum_stage1_ids.txt').read_text().splitlines())
            for name in ('event_count','random_half'):
                for suffix in ('assignments.csv','config.json'):
                    self.assertEqual((out/f'{name}_control_{suffix}').read_bytes(),(HERE/f'{name}_control_{suffix}').read_bytes())
            m.build(out,data)
            (data/'test.jsonl').write_text('protected\n')
            with self.assertRaisesRegex(FileExistsError,'different'):
                m.build(out,data)
            self.assertEqual((data/'test.jsonl').read_text(),'protected\n')


if __name__ == '__main__':
    unittest.main()
