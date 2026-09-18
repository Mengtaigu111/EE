from copy import deepcopy
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import data_loader
import main
from utils import decode as real_decode


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, bert_inputs, pieces2word, dist_inputs, document_length):
        return self.logits


class PredictionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = data_loader.Vocabulary()
        cls.vocab.load_ontology(str(REPO / "data/SciEvents/ontology.json"))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="exceeds-prediction-mask-")
        self.addCleanup(temporary.cleanup)
        self.output_dir = Path(temporary.name)

    @staticmethod
    def expected_events(length):
        events = [
            {"event_type": "FIN", "trigger": {"offsets": [0]},
             "arguments": [{"offsets": [2], "argument_type": "Content"}]},
            {"event_type": "FAC", "trigger": {"offsets": [2]},
             "arguments": [{"offsets": [1, 3], "argument_type": "Subject"}]},
        ]
        if length == 6:
            events.append({"event_type": "PRP", "trigger": {"offsets": [4]},
                           "arguments": [{"offsets": [5], "argument_type": "Content"}]})
        return events

    def run_path(self, method, lengths, use_cuda=False):
        samples = []
        expected = [self.expected_events(length) for length in lengths]
        for length, events in zip(lengths, expected):
            samples.append((
                torch.zeros(length + 2, dtype=torch.long),
                torch.zeros(length, length + 2, dtype=torch.bool),
                torch.zeros(length, length, dtype=torch.long),
                torch.zeros(length, length, self.vocab.label_num, dtype=torch.long),
                torch.ones(length, length, dtype=torch.bool), length, deepcopy(events),
            ))
        batch = data_loader.collate_fn(samples)
        logits = torch.full((len(lengths), 6, 6, self.vocab.label_num), -5.0)
        logits[..., 0] = 10.0  # NONE must remain false even with a positive logit.
        for index, length in enumerate(lengths):
            for row, column, label in (
                (0, 0, "FIN"), (2, 2, "FAC"), (2, 2, "Content"),
                (1, 3, "HTL"), (3, 1, "Subject"), (0, 2, "EAL"), (2, 1, "EAL"),
                (4, 4, "PRP"), (5, 5, "Content"), (4, 5, "EAL"),
            ):
                logits[index, row, column, self.vocab.label2id[label]] = 3.0
            if length < 6:
                logits[index, 0, 5, self.vocab.label2id["EAL"]] = 3.0
        unmasked = torch.zeros_like(logits, dtype=torch.bool)
        unmasked[..., 1:] = logits[..., 1:] > 0
        if use_cuda:
            logits = logits.cuda()
        captured = {}

        def capture_decode(predictions, vocab):
            captured["decoding_device"] = predictions.device.type
            captured["grid"] = predictions.clone()
            decoded = real_decode(predictions, vocab)
            captured["events"] = deepcopy(decoded)
            return decoded

        # These methods require model/vocab only; optimizer setup is unrelated.
        trainer = object.__new__(main.Trainer)
        trainer.model = FixedLogits(logits)
        trainer.vocab = self.vocab
        logger = logging.Logger("prediction-mask-test")
        logger.addHandler(logging.NullHandler())
        cuda_context = nullcontext() if use_cuda else patch.object(
            torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
        with cuda_context, \
             patch.object(main, "config", SimpleNamespace(output_dir=str(self.output_dir)), create=True), \
             patch.object(main, "logger", logger, create=True), \
             patch.object(main, "tqdm", lambda iterable: iterable), \
             patch.object(main.utils, "decode", side_effect=capture_decode):
            score = trainer.eval(0, [batch]) if method == "eval" else trainer.predict([batch])
        return score, captured, unmasked, batch[4], expected

    def assert_valid_region_unchanged(self, captured, unmasked, valid_mask):
        self.assertFalse(captured["grid"][..., 0].any().item())
        self.assertTrue(torch.equal(captured["grid"][valid_mask], unmasked[valid_mask]))

    def test_eval_excludes_padding_and_preserves_nested_event_structure(self):
        score, captured, unmasked, mask, expected = self.run_path("eval", (4, 6))
        self.assert_valid_region_unchanged(captured, unmasked, mask)
        self.assertFalse(captured["grid"][~mask].any().item(),
                         "Padded cells reached the real decoder in Trainer.eval")
        self.assertEqual(captured["events"], expected)
        self.assertEqual(score, (1.0, 1.0))

    def test_predict_persists_only_real_tokens_and_preserves_nested_events(self):
        lengths = (4, 6)
        score, captured, unmasked, mask, expected = self.run_path("predict", lengths)
        self.assert_valid_region_unchanged(captured, unmasked, mask)
        persisted = json.loads((self.output_dir / "test_predictions.json").read_text())
        for events, length in zip(persisted, lengths):
            for event in events:
                for nugget in [event["trigger"]] + event["arguments"]:
                    self.assertTrue(all(0 <= index < length for index in nugget["offsets"]),
                                    "Trainer.predict persisted an out-of-document nugget: {}".format(nugget))
        self.assertEqual(persisted, expected)
        self.assertEqual(captured["events"], expected)
        self.assertEqual(score, 1.0)

    def test_eval_without_padding_preserves_unmasked_predictions(self):
        score, captured, unmasked, mask, expected = self.run_path("eval", (6, 6))
        self.assertTrue(mask.all().item())
        self.assertTrue(torch.equal(captured["grid"], unmasked))
        self.assertEqual(captured["events"], real_decode(unmasked, self.vocab))
        self.assertEqual(captured["events"], expected)
        self.assertEqual(score, (1.0, 1.0))

    def test_predict_without_padding_preserves_unmasked_predictions(self):
        score, captured, unmasked, mask, expected = self.run_path("predict", (6, 6))
        self.assertTrue(mask.all().item())
        self.assertTrue(torch.equal(captured["grid"], unmasked))
        persisted = json.loads((self.output_dir / "test_predictions.json").read_text())
        self.assertEqual(persisted, real_decode(unmasked, self.vocab))
        self.assertEqual(persisted, expected)
        self.assertEqual(score, 1.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is needed for the transfer regression")
    def test_eval_decodes_cuda_outputs_on_cpu_without_changing_events(self):
        score, captured, _, _, expected = self.run_path("eval", (4, 6), use_cuda=True)
        self.assertEqual(captured["decoding_device"], "cpu")
        self.assertEqual(captured["events"], expected)
        self.assertEqual(score, (1.0, 1.0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is needed for the transfer regression")
    def test_predict_decodes_cuda_outputs_on_cpu_without_changing_saved_events(self):
        score, captured, _, _, expected = self.run_path("predict", (4, 6), use_cuda=True)
        self.assertEqual(captured["decoding_device"], "cpu")
        self.assertEqual(json.loads((self.output_dir / "test_predictions.json").read_text()), expected)
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
