import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import AutoModel, RobertaConfig


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tasks" / "summarize_full_run.py"
sys.path.insert(0, str(REPO))

import model


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FullRunSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backbone_config = RobertaConfig(
            vocab_size=31, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16,
            max_position_embeddings=32, type_vocab_size=1,
        )
        cls.config = json.loads((REPO / "config/scievents.json").read_text())
        cls.config.update(seed=52, bert_hid_size=8, lstm_hid_size=8,
                          dist_emb_size=2, grid_channels=4)
        parameters = SimpleNamespace(**cls.config, label_num=5)
        with patch.object(model.AutoModel, "from_pretrained", side_effect=lambda *args, **kwargs:
                          AutoModel.from_config(cls.backbone_config)):
            instance = model.EXCEEDS(parameters)
        cls.valid_checkpoint = {key: value.clone() for key, value in instance.state_dict().items()}
        assert all(value.device.type == "cpu" for value in cls.valid_checkpoint.values())

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="exceeds-summary-regression-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.backbone_dir = self.root / "backbone-config-only"
        self.backbone_config.save_pretrained(self.backbone_dir)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.train_dir = self.run_dir / "20260905_000000000"
        self.train_dir.mkdir()
        config = dict(self.config, bert_name=str(self.backbone_dir))
        (self.run_dir / "config.json").write_text(json.dumps(config))
        self.ontology_path = self.data_dir / "ontology.json"
        self.ontology_path.write_text(json.dumps({"event_types": {"TEST": {"Role": []}}}))
        for split, count in (("train", 2007), ("dev", 251), ("test", 250)):
            (self.data_dir / (split + ".json")).write_text(json.dumps([{"events": []}] * count))
        self.manifest = {
            "seed": 52, "epochs": 20, "batch_size": 2,
            "source_sha256": {name: file_hash(REPO / name)
                              for name in ("model.py", "data_loader.py", "utils.py")},
            "data": {
                "directory": str(self.data_dir),
                "files": [{"file": path.name, "sha256": file_hash(path)}
                          for path in sorted(self.data_dir.glob("*.json"))],
            },
        }
        self.write_manifest()
        (self.run_dir / "train-exit-code.txt").write_text("0\n")
        (self.run_dir / "console.log").write_text("Synthetic summary regression fixture.\n")
        prefix = "[2026-09-05 00:00:00] - INFO - root - "
        log = [prefix + "Train - Epoch {} - Loss: 0.500000\n".format(epoch) for epoch in range(20)]
        log += [prefix + "Best epoch: 5\n", prefix + "Test\n\n"]
        log += ["| {} | 0.0000 | 0.0000 | 0.0000 |\n".format(name)
                for name in ("trg_I", "trg_C", "arg_I", "arg_C", "ec_I", "ec_C")]
        (self.train_dir / "train.log").write_text("".join(log))
        (self.train_dir / "test_predictions.json").write_text(json.dumps([[]] * 250))
        self.checkpoint = {key: value.clone() for key, value in self.valid_checkpoint.items()}
        self.checkpoint_path = self.train_dir / "best_model.state"
        self.write_checkpoint()

    def write_manifest(self):
        (self.run_dir / "manifest.json").write_text(json.dumps(self.manifest))

    def write_checkpoint(self):
        torch.save(self.checkpoint, self.checkpoint_path)

    def write_seeds(self, config_seed, manifest_seed):
        path = self.run_dir / "config.json"
        config = json.loads(path.read_text())
        config["seed"] = config_seed
        path.write_text(json.dumps(config))
        self.manifest["seed"] = manifest_seed
        self.write_manifest()

    def summarize(self):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--run-dir", str(self.run_dir)],
            cwd=str(self.root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
                 "TRANSFORMERS_OFFLINE": "1"},
        )
        report = json.loads((self.run_dir / "result.json").read_text())
        self.assertEqual(json.loads(result.stdout), report)
        return result, report

    def assert_rejected(self, stage):
        result, report = self.summarize()
        self.assertEqual(result.returncode, 1, {
            "status": report["status"], "checkpoint": report.get("checkpoint"),
            "stderr": result.stderr,
        })
        self.assertFalse(report["passed"])
        self.assertEqual(report["failed_check"], stage, report)
        return report

    def test_accepts_complete_checkpoint_and_records_verified_hashes(self):
        result, report = self.summarize()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(report["passed"])
        self.assertEqual(report["checkpoint"].get("sha256"), file_hash(self.checkpoint_path))
        self.assertEqual(report["checkpoint"].get("schema_device"), "meta")
        self.assertEqual(report["data"]["verified_sha256"].get("ontology"), file_hash(self.ontology_path))

    def test_accepts_seeds_42_and_43_and_reports_actual_seed(self):
        for seed in (42, 43):
            with self.subTest(seed=seed):
                self.write_seeds(seed, seed)
                result, report = self.summarize()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(report["configuration"]["seed"], seed)
                summary = (self.run_dir / "summary.md").read_text()
                self.assertIn("本次 seed {} 单次全量训练".format(seed), summary)
                self.assertIn("这是 seed {} 的单次全量训练验证".format(seed), report["limitations"])
                self.assertIn("作者 seed 52", summary)
                self.assertIn("选模协议不同", report["limitations"])
                self.assertIn("不能合称严格的三种子均值", report["limitations"])
                self.assertEqual(report["comparisons"]["TI"]["author_seed52_f1_percent"], 75.17)

    def test_rejects_seed_mismatch_between_config_and_manifest(self):
        self.write_seeds(42, 43)
        report = self.assert_rejected("inputs_and_exit_code")
        self.assertIn("Manifest seed", report["error"])

    def test_requires_explicit_seed_in_run_config(self):
        path = self.run_dir / "config.json"
        config = json.loads(path.read_text())
        del config["seed"]
        path.write_text(json.dumps(config))
        self.assert_rejected("inputs_and_exit_code")

    def test_rejects_non_integer_config_seed(self):
        for seed in (52.0, "52", True, None):
            with self.subTest(seed=seed):
                self.write_seeds(seed, seed)
                self.assert_rejected("inputs_and_exit_code")

    def test_rejects_non_integer_manifest_seed(self):
        self.write_seeds(52, 52.0)
        self.assert_rejected("inputs_and_exit_code")

    def test_rejects_four_one_element_tensors(self):
        self.checkpoint = {key: torch.zeros(1) for key in (
            "bert.embeddings.word_embeddings.weight", "lstm.weight_ih_l0",
            "classifier.weight", "classifier.bias",
        )}
        self.write_checkpoint()
        self.assert_rejected("checkpoint")

    def test_rejects_missing_parameter_beyond_original_four_keys(self):
        del self.checkpoint["grid_refiner.layers.0.conv.weight"]
        self.write_checkpoint()
        self.assert_rejected("checkpoint")

    def test_rejects_unexpected_parameter(self):
        self.checkpoint["unexpected.weight"] = torch.zeros(1)
        self.write_checkpoint()
        self.assert_rejected("checkpoint")

    def test_rejects_wrong_shape_with_unchanged_element_count(self):
        key = "grid_refiner.layers.0.conv.weight"
        self.checkpoint[key] = self.checkpoint[key].flatten()
        self.write_checkpoint()
        self.assert_rejected("checkpoint")

    def test_rejects_wrong_dtype(self):
        self.checkpoint["classifier.weight"] = self.checkpoint["classifier.weight"].double()
        self.write_checkpoint()
        self.assert_rejected("checkpoint")

    def test_rejects_nonfinite_checkpoint(self):
        self.checkpoint["classifier.weight"][0, 0] = float("nan")
        self.write_checkpoint()
        self.assert_rejected("checkpoint")

    def test_rejects_changed_ontology_before_loading_checkpoint(self):
        self.ontology_path.write_text(json.dumps({"event_types": {"TEST": {"OtherRole": []}}}))
        report = self.assert_rejected("data_and_predictions")
        self.assertIn("ontology.json", report["error"])
        self.assertNotIn("checkpoint", report["checks"])

    def test_rejects_missing_ontology_manifest_hash(self):
        self.manifest["data"]["files"] = [entry for entry in self.manifest["data"]["files"]
                                          if entry["file"] != "ontology.json"]
        self.write_manifest()
        report = self.assert_rejected("data_and_predictions")
        self.assertIn("ontology.json", report["error"])


if __name__ == "__main__":
    unittest.main()
