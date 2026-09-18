from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import model

BASELINE_COMMIT = "c61ba9b8920eb95bf0c736bcd9da2fe45632446b"


class ControlledEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.embeddings = torch.nn.Embedding(17, hidden_size)
        self.forward_calls = 0

    def forward(self, input_ids, attention_mask):
        self.forward_calls += 1
        return (self.embeddings(input_ids),)


class ModelAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = subprocess.check_output(
            ["git", "show", BASELINE_COMMIT + ":model.py"], cwd=str(REPO), text=True
        )
        cls.legacy = ModuleType("exceeds_original_model")
        exec(compile(source, BASELINE_COMMIT + "/model.py", "exec"), cls.legacy.__dict__)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **changes):
        values = dict(bert_name="controlled-encoder", bert_hid_size=8, lstm_hid_size=8,
                      dropout_rate=0.25, dist_emb_size=2, grid_channels=4,
                      grid_refine_layers=2, grid_refine_dropout=0.1,
                      grid_refine_kernel=3, label_num=5)
        values.update(changes)
        return SimpleNamespace(**values)

    def build(self, config, implementation=model):
        torch.manual_seed(42)
        with patch.object(implementation.AutoModel, "from_pretrained",
                          side_effect=lambda *args, **kwargs: ControlledEncoder(config.bert_hid_size)):
            return implementation.EXCEEDS(config)

    def inputs(self):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 7, 8, 9, 6, 0]])
        lengths = torch.tensor([4, 3])
        pieces = torch.zeros(2, 4, 6, dtype=torch.bool)
        for sample, length in enumerate(lengths.tolist()):
            for word in range(length):
                pieces[sample, word, word + 1] = True
        distances = torch.full((2, 4, 4), 19, dtype=torch.long)
        return ids, pieces, distances, lengths

    def test_default_matches_original_parameters_and_output(self):
        original = self.build(self.config(), self.legacy).eval()
        default = self.build(self.config()).eval()
        explicit = self.build(self.config(use_contextual=True)).eval()
        expected_state = original.state_dict()
        for instance in (default, explicit):
            self.assertEqual(list(instance.state_dict()), list(expected_state))
            for key, tensor in instance.state_dict().items():
                torch.testing.assert_close(tensor, expected_state[key], rtol=0, atol=0)
            with torch.no_grad():
                torch.testing.assert_close(instance(*self.inputs()), original(*self.inputs()), rtol=0, atol=0)

    def test_contextual_bypass_has_no_context_weights_and_keeps_word_dropout(self):
        instance = self.build(self.config(use_contextual=False))
        self.assertIsNone(instance.lstm)
        self.assertIsNone(instance.cln)
        self.assertFalse(any(name.startswith(("lstm.", "cln.")) for name, _ in instance.named_parameters()))
        captured = {}

        def capture_dropout(module, arguments, output):
            captured["words"] = output.detach().clone()

        def capture_pairs(module, arguments):
            captured["pairs"] = arguments[0].detach().clone()

        dropout_hook = instance.dropout.register_forward_hook(capture_dropout)
        pairs_hook = instance.pair_mlp_feat.register_forward_pre_hook(capture_pairs)
        try:
            inputs = self.inputs()
            output = instance(*inputs)
        finally:
            dropout_hook.remove()
            pairs_hook.remove()
        mask = torch.arange(4)[None, :] < inputs[-1][:, None]
        words = captured["words"] * mask.unsqueeze(-1)
        torch.testing.assert_close(captured["pairs"][..., :8], words.unsqueeze(2).expand(-1, -1, 4, -1))
        torch.testing.assert_close(captured["pairs"][..., 8:16], words.unsqueeze(1).expand(-1, 4, -1, -1))
        self.assertEqual(instance.bert.forward_calls, 1)
        self.assertEqual(instance.dropout.p, 0.25)
        self.assertTrue(instance.dropout.training)
        self.assertEqual(tuple(output.shape), (2, 4, 4, 5))

    def test_zero_refinement_layers_still_mask_forward_and_gradient(self):
        refiner = model.GridRefiner(channels=3, num_layers=0)
        inputs = torch.ones(2, 3, 4, 4, requires_grad=True)
        mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
        valid = (mask.unsqueeze(1) & mask.unsqueeze(2)).unsqueeze(1).expand_as(inputs)
        output = refiner(inputs, mask=mask)
        torch.testing.assert_close(output, inputs * valid)
        output.sum().backward()
        torch.testing.assert_close(inputs.grad, valid.to(inputs.dtype))
        self.assertEqual(list(refiner.parameters()), [])
        self.assertIs(refiner(inputs), inputs)

    def test_both_variants_have_finite_cpu_forward_and_backward(self):
        for changes in ({"use_contextual": False}, {"grid_refine_layers": 0}):
            with self.subTest(changes=changes):
                instance = self.build(self.config(**changes))
                output = instance(*self.inputs())
                self.assertEqual(tuple(output.shape), (2, 4, 4, 5))
                self.assertTrue(torch.isfinite(output).all().item())
                output.square().mean().backward()
                for name in ("bert.embeddings.weight", "pair_mlp_feat.linear.weight", "classifier.weight"):
                    gradient = dict(instance.named_parameters())[name].grad
                    self.assertIsNotNone(gradient, name)
                    self.assertTrue(torch.isfinite(gradient).all().item(), name)
                    self.assertGreater(gradient.abs().sum().item(), 0, name)
                if changes.get("grid_refine_layers") == 0:
                    self.assertFalse(any(name.startswith("grid_refiner.") for name, _ in instance.named_parameters()))

    def test_contextual_bypass_rejects_mismatched_dimensions(self):
        with self.assertRaisesRegex(ValueError, "bert_hid_size.*lstm_hid_size"):
            self.build(self.config(use_contextual=False, lstm_hid_size=10))


if __name__ == "__main__":
    unittest.main()
