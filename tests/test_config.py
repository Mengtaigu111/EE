import argparse
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import Config


def main_parser():
    """Use the real CLI declarations without importing the training dependencies."""
    path = REPO / "main.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    entrypoint = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )
    setup = []
    for node in entrypoint.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "args"
            for target in node.targets
        ):
            break
        setup.append(node)
    else:
        raise AssertionError("Could not locate main.py argument parsing")
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=setup, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["parser"]


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = main_parser()
        cls.readme_path = REPO / "config" / "scievents.json"
        cls.readme_config = json.loads(cls.readme_path.read_text(encoding="utf-8"))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.config_path = Path(temporary.name) / "config.json"

    def load_config(self, updates=None, cli=(), path=None):
        if path is None:
            data = dict(self.readme_config)
            data.update(updates or {})
            self.config_path.write_text(json.dumps(data), encoding="utf-8")
            path = self.config_path
        args = self.parser.parse_args(["--config", str(path)] + list(cli))
        try:
            return Config(args)
        except KeyError as exc:
            self.fail("Valid configuration raised KeyError: {}".format(exc))

    def test_readme_config_initializes_with_defaults(self):
        config = self.load_config(path=self.readme_path)
        self.assertEqual(config.seed, 52)
        self.assertEqual(config.ontology, "./data/SciEvents/ontology.json")
        for key, value in self.readme_config.items():
            self.assertEqual(getattr(config, key), value)

    def test_missing_json_seed_uses_default(self):
        config = self.load_config({"ontology": "schema.json"})
        self.assertEqual(config.seed, 52)

    def test_explicit_cli_seed_works_without_json_seed(self):
        config = self.load_config({"ontology": "schema.json"}, ("--seed", "19"))
        self.assertEqual(config.seed, 19)

    def test_json_seed_is_used_when_cli_seed_is_omitted(self):
        for seed in (0, 17):
            with self.subTest(seed=seed):
                config = self.load_config({"ontology": "schema.json", "seed": seed})
                self.assertEqual(config.seed, seed)

    def test_explicit_cli_seed_overrides_json(self):
        for seed in (0, 23):
            with self.subTest(seed=seed):
                config = self.load_config(
                    {"ontology": "schema.json", "seed": 17}, ("--seed", str(seed))
                )
                self.assertEqual(config.seed, seed)

    def test_missing_ontology_defaults_to_dataset_path(self):
        config = self.load_config({"dataset": "CustomEvents", "seed": 17})
        self.assertEqual(config.ontology, "./data/CustomEvents/ontology.json")

    def test_explicit_json_ontology_is_preserved(self):
        config = self.load_config({"ontology": "custom/ontology.json", "seed": 17})
        self.assertEqual(config.ontology, "custom/ontology.json")

    def test_contextual_is_enabled_by_default(self):
        config = self.load_config()
        self.assertIs(getattr(config, "use_contextual", None), True)

    def test_contextual_ablation_flag_is_read_from_json(self):
        config = self.load_config({"use_contextual": False})
        self.assertIs(getattr(config, "use_contextual", None), False)


if __name__ == "__main__":
    unittest.main()
