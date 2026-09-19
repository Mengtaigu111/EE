# Stage-3 Structure-Aware Event Tree Refinement v1

This directory implements the isolated Stage-3 protocol in
`task/0918-stage3-tree-refinement-v1.md`. It does not modify Stage-2 scripts.

## Current execution gate

The Experiment C queue `C-42-20260918` is complete. D1/D2 still require the
fixed C completion plus project-GPU-idle gate; F0/F1 are independent of C.
Every GPU phase must run in a named tmux session. Use the fixed interpreter:

```bash
PY=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
$PY -m pytest -q tasks/stage3-tree-refinement/tests
```

The first live run is `tmux stage3-d1-dev`, writing
`runs/D1-dev-42-20260919/`. `tmux stage3-d2-dev` runs
`scripts/run_d2_dev.sh`: it waits for D1, then runs D2 dev in both type-set
modes (`all`, `confusion`) into `runs/D2-dev-<mode>-42-<date>/`, followed by
`select_r1.py` (score_sum / score_mean), the Stage-2 `evaluate.py`, and
`diagnose.py` (§4.4 diagnostics).

Review fixes applied 2026-09-19 (Claude): the D2 forced prefix is now placed on
the assistant side of the chat template (`forced_prefix_ids`); the previous
version put the tree inside the user turn, so the model restarted a fresh
forest. Prior events are sliced from the stored greedy `raw_output`
(`greedy_prior_text`) and the constructed prefix is checked against it token by
token. Reward training keeps fp32 master weights under bf16 autocast. The
second synthetic negative keeps the gold type and drops one argument branch.
`run_queue.py --dependency-dir` lets a phase depend on another run directory. D1/D2 dev output is a decision checkpoint before
the reward-model phases continue. Generic resumable phases use:

```bash
tmux new-session -d -s stage3-f0 \
  "cd /home/myluo/EE/EXCEEDS && $PY tasks/stage3-tree-refinement/scripts/run_queue.py \
   --phase f0 --run-dir tasks/stage3-tree-refinement/runs/F0-42-20260919 \
   --gpu --command $PY tasks/stage2-event-tree/scripts/train_lora.py ..."
```

The queue writes `run.json`, phase state, logs, exit JSON, provenance, and
test-once markers. It refuses silent truncation and stale-attempt publication.

## Module map

- `common.py`: read-only Stage-2 import bridge, identities and artifact I/O.
- `diagnostics.py`: candidate labels and T0-relative diagnostics.
- `sample_pool.py`: D1 candidate normalization and support merge.
- `type_competition.py`: D2 prefix and subtree scoring primitives.
- `build_folds.py`, `build_reward_data.py`: leakage-safe reward data.
- `train_reward.py`, `score_candidates.py`: Qwen/RoBERTa multi-head scorer,
  exact masking, no-truncation audit, and validation metrics.
- `select_r1.py`, `select_r0.py`: type competition, candidate filtering,
  dev-only tau selection, and deterministic tie breaks.
- `run_queue.py`, `write_report.py`: gated/resumable phases and artifact-only
  report rendering.

The remaining GPU entry points are added incrementally with test-first
coverage. No silent truncation is permitted.
