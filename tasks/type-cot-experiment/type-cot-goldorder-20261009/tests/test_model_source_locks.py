"""Model contents must stay identical across the CoT/tree stage boundary."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('model_locks_config', HERE / 'config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    model = tmp_path / 'model'
    model.mkdir()
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json',
                 'merges.txt', 'generation_config.json', 'model-1.safetensors', 'model-2.safetensors'):
        (model / name).write_text('{}' if name.endswith('.json') else 'original contents')
    (model / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'layer.0': 'model-1.safetensors', 'layer.1': 'model-2.safetensors'}}))
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('train.jsonl', 'dev.jsonl', 'test.jsonl', 'manifest.json', 'verification.json'):
        (data / name).write_text('{}')
    reference, prompt = tmp_path / 'reference.json', tmp_path / 'prompt.txt'
    reference.write_text('{}')
    prompt.write_text('prompt')
    monkeypatch.setitem(config.SPEC, 'model_path', str(model))
    monkeypatch.setattr(config, 'COT_DATA', data)
    monkeypatch.setattr(config, 'REFERENCE', reference)
    monkeypatch.setattr(config, 'COT_PROMPT', prompt)
    monkeypatch.setattr(config, 'build_tree_reference', lambda seed: {'frozen_hashes': {}})
    monkeypatch.setattr(config, 'audit_alignment', lambda manifest: True)
    return config


def test_manifest_locks_all_loaded_model_files(config):
    manifest = config.build_manifest(42)
    model = Path(config.SPEC['model_path'])
    assert {str(path) for path in model.iterdir()} <= set(manifest['frozen_hashes'])


@pytest.mark.parametrize('name', ['model-1.safetensors', 'model-2.safetensors',
                                  'tokenizer.json', 'vocab.json', 'merges.txt',
                                  'generation_config.json'])
def test_replaced_contents_fail_before_next_stage(config, name):
    manifest = config.build_manifest(42)
    (Path(config.SPEC['model_path']) / name).write_text('replacement')
    with pytest.raises(ValueError):
        config.validate_manifest(manifest)
    with pytest.raises(ValueError, match='Frozen input/source changed'):
        config.verify_locks(manifest['frozen_hashes'])


def test_incomplete_historical_manifest_is_not_silently_upgraded(config):
    manifest = config.build_manifest(42)
    model = Path(config.SPEC['model_path'])
    legacy_names = {'config.json', 'tokenizer_config.json', 'model.safetensors.index.json'}
    manifest['frozen_hashes'] = {name: digest for name, digest in manifest['frozen_hashes'].items()
        if Path(name).parent != model or Path(name).name in legacy_names}
    saved = json.dumps(manifest, sort_keys=True)
    with pytest.raises(ValueError):
        config.validate_manifest(manifest)
    assert json.dumps(manifest, sort_keys=True) == saved


def test_added_tokenizer_sidecar_changes_inventory(config):
    manifest = config.build_manifest(42)
    (Path(config.SPEC['model_path']) / 'added_tokens.json').write_text('{}')
    with pytest.raises(ValueError):
        config.validate_manifest(manifest)


def test_missing_indexed_shard_is_rejected(config):
    (Path(config.SPEC['model_path']) / 'model-2.safetensors').unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        config.build_manifest(42)


@pytest.mark.parametrize('weight_map', [{}, {'layer': '../outside.safetensors'}])
def test_invalid_shard_inventory_is_rejected(config, weight_map):
    (Path(config.SPEC['model_path']) / 'model.safetensors.index.json').write_text(
        json.dumps({'weight_map': weight_map}))
    with pytest.raises(ValueError):
        config.build_manifest(42)
