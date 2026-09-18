import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from align_mentions import align_mentions, parse_output

class AlignerTests(unittest.TestCase):
    def align(self, text, mentions, start=10):
        return align_mentions(text.split(), mentions, start)

    def test_normal(self):
        a = self.align('we use a model', [{'text':'a model','type':'APP'}])
        self.assertEqual(a[0]['offsets'], [12,13])
        self.assertEqual(a[0]['method'], 'exact')

    def test_duplicate_text(self):
        a = self.align('model and model', [{'text':'model','type':'APP'}]*3)
        self.assertEqual([m['offsets'] for m in a], [[10],[12],None])

    def test_case(self):
        a = self.align('A Model works', [{'text':'a model','type':'APP'}])
        self.assertEqual(a[0]['offsets'], [10,11])
        self.assertEqual(a[0]['method'], 'case_insensitive')

    def test_discontinuous(self):
        a = self.align('our new and very useful model', [{'text':'our model','type':'APP'}])
        self.assertEqual(a[0]['offsets'], [10,15])
        self.assertEqual(a[0]['method'], 'subsequence')

    def test_unaligned(self):
        self.assertIsNone(self.align('we use a model', [{'text':'not here','type':'APP'}])[0]['offsets'])

    def test_same_span_multiple_types(self):
        a = self.align('model', [{'text':'model','type':'APP'}, {'text':'model','type':'MOD'}])
        self.assertEqual([m['offsets'] for m in a], [[10],[10]])

    def test_exact_priority(self):
        a = self.align('a different model then a model', [{'text':'a model','type':'APP'}])
        self.assertEqual(a[0]['offsets'], [14,15])

    def test_parser_fallback_braces_in_strings(self):
        raw = 'Output: {"mentions":[{"text":"a {b}","type":"APP"}]} trailing'
        parsed, invalid = parse_output(raw)
        self.assertFalse(invalid)
        self.assertEqual(parsed[0]['text'], 'a {b}')

    def test_parser_invalid(self):
        for raw in ['no json', '{"mentions": [', '{"mentions":[{"text":"x","type":"E-PUR"}]}', '{"mentions":null}', '{"mentions":[{"text":"","type":"APP"}]}']:
            self.assertEqual(parse_output(raw), ([], True))

    def test_first_complete_invalid_object_does_not_skip(self):
        self.assertEqual(parse_output('Output: {bad} {"mentions":[]}'), ([], True))

    def test_multiple_types_and_occurrences(self):
        mentions = [{'text':'model','type':t} for t in ['APP','MOD','APP','MOD']]
        self.assertEqual([m['offsets'] for m in self.align('model and model', mentions)], [[10],[10],[12],[12]])

    def test_valid_empty(self):
        self.assertEqual(parse_output('{"mentions":[]}'), ([], False))

if __name__ == '__main__': unittest.main()
