# Complete model source locks (2026-10-12)

New manifests from `config.build_manifest(seed)` include SHA256 hashes for every safetensors shard referenced by the model index, the model/index/tokenizer configuration files, and available tokenizer vocabulary, merges, added/special tokens, chat template and generation configuration files. Missing indexed shards or invalid shard paths are rejected. Hashing streams the contents and does not cache digests; `validate_manifest()` recomputes the complete inventory and hashes. Stage-boundary `verify_locks()` therefore also detects modified shard/tokenizer contents.

The current Qwen3-8B source has five weight shards. Their index alone cannot prove that their contents are unchanged.

## Historical records and new runs

Historical manifests, plans, results and source snapshots remain unchanged. They do not acquire complete model provenance retrospectively. The updated configuration rejects their incomplete lock inventories; historical queue plans referencing these manifests also fail validation. Do not use those plans to launch another run or overwrite their manifests. In particular, cancelled seeds are not re-authorized by this change.

For a future authorized run, generate a new manifest at a new path with `config.build_manifest(seed)` using the updated code, then use that manifest with the existing `run.py` entry. A new queue plan must reference the new manifest and current source hashes. Historical verification should use its recorded source snapshot; it cannot certify the newly added model hashes for the past run.

Full hashing reads about 16.4 GB per model check. `validate_manifest()` compares recomputed hashes directly with the frozen dictionary, avoiding a redundant second verification pass. This fix does not change the model recipe, datasets, seeds, checkpoints or scores and does not launch training.

## Regression checks

`tests/test_model_source_locks.py` uses small local fixtures, never writes the real model, and verifies: complete coverage; rejection of replaced weights/tokenizer/merges/vocabulary/generation config; rejection of incomplete legacy manifests without mutation; added tokenizer sidecar detection; missing shard rejection; and invalid index/path rejection.

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m pytest -q tasks/type-cot-experiment/type-cot-goldorder-20261009/tests
```
