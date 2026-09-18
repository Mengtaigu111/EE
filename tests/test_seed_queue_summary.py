"""CLI regressions for complete, matched-seed experiment summaries."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
METRICS = ("TI", "TC", "AI", "AC", "EC")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class SeedQueueSummaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="exceeds-seed-summary-")
        self.addCleanup(temporary.cleanup)
        self.batch = Path(temporary.name)
        self.jobs = {"main_jobs": [], "ablation_jobs": [],
                     "expected_seeds": [42, 43, 52],
                     "contextual_definition": "Documented working definition.",
                     "legacy_unmasked_selection_seed52_included": False}
        for variant, delta in (("full", 0), ("no_contextual", -2), ("no_grid_refiner", 1)):
            for seed, base_f1 in ((42, 40), (43, 50), (52, 60)):
                directory = self.batch / "{}-seed{}".format(variant, seed)
                source = directory / "source"
                source.mkdir(parents=True)
                for name in ("main.py", "model.py", "utils.py"):
                    (source / name).write_text("# Synthetic fixture: {} {}\n".format(variant, name))
                config = {"seed": seed, "epochs": 20, "batch_size": 2,
                          "dataset": "SciEvents", "learning_rate": 0.001,
                          "use_contextual": variant != "no_contextual",
                          "grid_refine_layers": 0 if variant == "no_grid_refiner" else 2}
                manifest = {"seed": seed, "variant": variant, "epochs": 20, "batch_size": 2,
                            "protocol": "masked CPU decoding and matched dev selection",
                            "source_sha256": {p.name: digest(p) for p in source.iterdir()}}
                write_json(directory / "config.json", config)
                write_json(directory / "manifest.json", manifest)
                result = {"passed": True, "configuration": config, "variant": variant,
                          "checks": {"complete_training_log": True, "checkpoint": True},
                          "train_exit_code": 0,
                          "training": {"epochs_completed": 20, "best_epoch_one_based": 10,
                                       "all_epoch_losses_finite": True},
                          "data": {"predicted_documents": 250,
                                   "verified_sha256": {"test": "same-test-data"}},
                          "log_matches_recomputed_metrics_at_four_decimals": True,
                          "provenance": {"config_sha256": digest(directory / "config.json"),
                                         "manifest_sha256": digest(directory / "manifest.json")},
                          "metrics": {name: {"f1_percent": base_f1 + delta} for name in METRICS}}
                write_json(directory / "result.json", result)
                for name in ("train-exit-code.txt", "exit-code.txt"):
                    (directory / name).write_text("0\n")
                job = {"id": directory.name, "variant": variant, "seed": seed,
                       "directory": str(directory)}
                self.jobs["main_jobs" if variant == "full" else "ablation_jobs"].append(job)
        self.write_jobs()
        (self.batch / "queue-exit-code.txt").write_text("2\n")

    def write_jobs(self):
        write_json(self.batch / "jobs.json", self.jobs)

    def run_summary(self, script="summarize_three_seed_extension.py"):
        return subprocess.run([sys.executable, "-B", str(REPO / "tasks" / script),
                               "--batch-dir", str(self.batch)],
                              cwd=str(self.batch), text=True, capture_output=True)

    def change_input(self, filename, update):
        directory = self.batch / "full-seed52"
        path = directory / filename
        value = json.loads(path.read_text())
        value.update(update)
        write_json(path, value)
        if filename in ("config.json", "manifest.json"):
            result_path = directory / "result.json"
            result = json.loads(result_path.read_text())
            result["provenance"][filename.replace(".json", "_sha256")] = digest(path)
            write_json(result_path, result)

    def assert_rejected(self, message):
        process = self.run_summary()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(message.lower(), process.stderr.lower())
        self.assertFalse((self.batch / "combined-results.json").exists())
        self.assertFalse((self.batch / "combined-summary.md").exists())

    def test_three_seed_cli_computes_sample_statistics_and_preserves_inputs(self):
        before = {str(p): digest(p) for p in self.batch.rglob("*") if p.is_file()}
        process = self.run_summary()
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((self.batch / "combined-results.json").read_text())
        self.assertEqual(report["seeds"], [42, 43, 52])
        self.assertEqual(report["sample_count_per_variant"], 3)
        self.assertEqual(report["averages"]["full"]["TC"], {"mean": 50, "sample_std": 10})
        self.assertEqual(report["paired_ablation_minus_full_pp"]["no_contextual"]["52"]["EC"], -2)
        self.assertEqual(report["paired_ablation_minus_full_pp"]["no_grid_refiner"]["42"]["AC"], 1)
        self.assertFalse(report["legacy_seed52_included_in_averages"])
        self.assertIn("n=3", (self.batch / "combined-summary.md").read_text())
        self.assertEqual(before, {name: digest(Path(name)) for name in before})

    def test_existing_two_seed_entry_still_summarizes_two_seeds(self):
        for key in ("main_jobs", "ablation_jobs"):
            self.jobs[key] = [job for job in self.jobs[key] if job["seed"] != 52]
        del self.jobs["expected_seeds"]
        self.write_jobs()
        process = self.run_summary("summarize_seed_ablation_queue.py")
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        report = json.loads((self.batch / "combined-results.json").read_text())
        self.assertEqual(report["seeds"], [42, 43])
        self.assertEqual(report["averages"]["full"]["TC"]["mean"], 45)
        self.assertAlmostEqual(report["averages"]["full"]["TC"]["sample_std"], 50 ** 0.5)

    def test_missing_variant_seed_is_rejected(self):
        self.jobs["ablation_jobs"].pop()
        self.write_jobs()
        self.assert_rejected("seed coverage")

    def test_duplicate_variant_seed_is_rejected(self):
        self.jobs["main_jobs"].append(dict(self.jobs["main_jobs"][0]))
        self.write_jobs()
        self.assert_rejected("duplicate")

    def test_failed_training_cannot_be_hidden_by_a_passed_result(self):
        (self.batch / "full-seed52" / "train-exit-code.txt").write_text("1\n")
        self.assert_rejected("train-exit-code")

    def test_legacy_selection_protocol_is_rejected(self):
        self.change_input("manifest.json", {"protocol": "legacy unmasked selection"})
        self.assert_rejected("protocol")

    def test_training_config_difference_other_than_seed_is_rejected(self):
        self.change_input("config.json", {"learning_rate": 0.01})
        self.assert_rejected("configuration")

    def test_changed_frozen_source_is_rejected(self):
        (self.batch / "full-seed52" / "source" / "main.py").write_text("# changed selection\n")
        self.assert_rejected("source hash")

    def test_nonfinite_metric_is_rejected(self):
        self.change_input("result.json", {"metrics": {name: {"f1_percent": float("nan")}
                                                     for name in METRICS}})
        self.assert_rejected("metric")


if __name__ == "__main__":
    unittest.main()
