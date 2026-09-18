# Stage-3 Structure-Aware Event Tree Refinement v1

This directory implements the isolated Stage-3 protocol in
`task/0918-stage3-tree-refinement-v1.md`. It does not modify Stage-2 scripts.

## Current execution gate

The current `tmux stage2-c` Experiment C queue must finish before any Stage-3
GPU command starts. CPU tests and fixture audits are safe while it runs. Use
the fixed interpreter:

```bash
PY=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
$PY -m pytest -q tasks/stage3-tree-refinement/tests
```

After C is complete, the queue records every phase under `runs/` and requires
a named tmux session. D1/D2 dev output is a decision checkpoint before the
reward-model phases continue.

## Module map

- `common.py`: read-only Stage-2 import bridge, identities and artifact I/O.
- `diagnostics.py`: candidate labels and T0-relative diagnostics.
- `sample_pool.py`: D1 candidate normalization and support merge.
- `type_competition.py`: D2 prefix and subtree scoring primitives.
- `build_folds.py`, `build_reward_data.py`: leakage-safe reward data.

The remaining GPU entry points are added incrementally with test-first
coverage. No silent truncation is permitted.
