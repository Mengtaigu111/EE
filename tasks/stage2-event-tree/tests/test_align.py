import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from align_events import align_forest
from linearize import canonical_events
import utils

PROJECT = Path(__file__).resolve().parents[3]


class AlignTests(unittest.TestCase):
    def test_cursor_and_sentence_preference(self):
        tokens = 'we use a model . the model is good . we use a model again .'.split()
        sentences = ['we use a model .', 'the model is good .', 'we use a model again .']
        events = [{'type': 'WKS', 'trigger': 'use', 'args': [{'role': 'Content', 'text': 'model'}]},
                  {'type': 'FAC', 'trigger': 'is', 'args': [{'role': 'Subject', 'text': 'model'}]},
                  {'type': 'WKS', 'trigger': 'use', 'args': [{'role': 'Content', 'text': 'model'}]}]
        aligned, stats = align_forest(tokens, sentences, events)
        self.assertEqual([e['trigger']['offsets'] for e in aligned], [[1], [7], [11]])
        self.assertEqual([e['arguments'][0]['offsets'] for e in aligned], [[3], [6], [13]])
        self.assertEqual(stats['unaligned_arguments'], 0)

    def test_subevent_prefers_trigger(self):
        tokens = 'results show that x improves y'.split()
        events = [{'type': 'FAC', 'trigger': 'show', 'args': [{'role': 'Reason', 'text': 'improves'}]},
                  {'type': 'PUR', 'trigger': 'improves', 'args': []}]
        aligned, _ = align_forest(tokens, [' '.join(tokens)], events)
        self.assertEqual(aligned[0]['arguments'][0]['offsets'], aligned[1]['trigger']['offsets'])

    def test_unaligned_and_discontinuous(self):
        tokens = 'our new and useful model works'.split()
        events = [{'type': 'FAC', 'trigger': 'works', 'args': [{'role': 'Subject', 'text': 'our model'}, {'role': 'Object', 'text': 'nothing'}]},
                  {'type': 'FAC', 'trigger': 'absent', 'args': []}]
        aligned, stats = align_forest(tokens, [' '.join(tokens)], events)
        self.assertEqual(len(aligned), 1)
        self.assertEqual(aligned[0]['arguments'][0]['offsets'], [0, 4])
        self.assertEqual(stats['unaligned_triggers'], 1)
        self.assertEqual(stats['unaligned_arguments'], 1)

    def test_gold_roundtrip_dev(self):
        docs = json.loads((PROJECT / 'data/SciEvents/dev.json').read_text())
        golds, preds = [], []
        for d in docs:
            text_events = [{'type': e['type'], 'trigger': e['trigger'], 'args': [{'role': a['role'], 'text': a['text']} for a in e['args']]}
                           for e in canonical_events(d)]
            aligned, _ = align_forest(d['document'], d['sentences'], text_events)
            golds.append(d['events'])
            preds.append(aligned)
        scores = utils.calculate_F1(golds, preds, 'offset')
        self.assertGreater(scores[0], 0.985, 'TI roundtrip')
        self.assertGreater(scores[3], 0.94, 'AC roundtrip; ~5% ceiling = duplicate-trigger dict collisions + reverse-order + ambiguous text')
        self.assertGreater(scores[5], 0.90, 'EC roundtrip')
        print('\nGOLD ROUNDTRIP dev: TI=%.4f TC=%.4f AI=%.4f AC=%.4f EC=%.4f' % (scores[0], scores[1], scores[2], scores[3], scores[5]))


if __name__ == '__main__':
    unittest.main()
