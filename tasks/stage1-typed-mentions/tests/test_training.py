import importlib
import sys
import tempfile
import json
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class TrainingInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained('/opt/models/Qwen3-8B', local_files_only=True)

    def helpers(self):
        self.assertTrue(importlib.util.find_spec('model_utils'), 'training tokenization implementation is missing')
        return importlib.import_module('model_utils')

    def test_only_complete_assistant_json_and_eos_receive_labels(self):
        m = self.helpers()
        row = {'id': 'empty', 'prompt': 'Extract mentions from we.', 'completion': '{"mentions":[]}'}
        encoded = m.encode_training_row(row, self.tokenizer, 1024)
        labels = encoded['labels']
        first = next(i for i, label in enumerate(labels) if label != -100)
        self.assertTrue(all(x == -100 for x in labels[:first]))
        self.assertEqual(labels[first:], encoded['input_ids'][first:])
        self.assertEqual(self.tokenizer.decode(labels[first:]), row['completion'] + '<|im_end|>\n')
        self.assertIn(self.tokenizer.eos_token_id, labels[first:])

    def test_overlength_rejected_without_truncating_gold(self):
        m = self.helpers()
        row = {'id': 'long', 'prompt': 'x ' * 200, 'completion': '{"mentions":[]}'}
        with self.assertRaisesRegex(ValueError, 'long.*max_len'):
            m.encode_training_row(row, self.tokenizer, 20)

    def test_padding_never_contributes_to_loss(self):
        m = self.helpers()
        rows = [m.encode_training_row({'id': str(i), 'prompt': p, 'completion': '{"mentions":[]}'}, self.tokenizer, 1024)
                for i, p in enumerate(['Short.', 'Longer input. ' * 12])]
        batch = m.AssistantCollator(self.tokenizer.pad_token_id)(rows)
        self.assertTrue((batch['labels'][batch['attention_mask'] == 0] == -100).all())
        self.assertEqual(batch['attention_mask'].sum().item(), sum(len(r['input_ids']) for r in rows))

    def test_audit_reports_all_overlength_examples(self):
        m = self.helpers()
        rows = [{'id': 'a', 'prompt': 'x ' * 100, 'completion': '{"mentions":[]}'},
                {'id': 'b', 'prompt': 'x ' * 150, 'completion': '{"mentions":[]}'}]
        audit = m.audit_rows(rows, self.tokenizer, 30)
        self.assertEqual(audit['overlength_count'], 2)
        self.assertEqual([r['id'] for r in audit['overlength']], ['a', 'b'])
        self.assertEqual(audit['count'], 2)

    def test_few_shot_reads_builder_json_array(self):
        m = self.helpers()
        self.assertTrue(hasattr(m, 'read_examples'), 'few-shot JSON loader missing')
        rows = [{'id': str(i), 'split': 'train', 'completion': '{"mentions":[]}'} for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'few_shot.json'
            path.write_text(json.dumps(rows))
            self.assertEqual(m.read_examples(path), rows)


if __name__ == '__main__':
    unittest.main()
