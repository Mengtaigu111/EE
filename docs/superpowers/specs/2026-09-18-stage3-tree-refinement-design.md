# Stage-3 Structure-Aware Event Tree Refinement v1 Design

> This design is derived from the user-approved specification in
> `task/0918-stage3-tree-refinement-v1.md`.

## Goal

Test whether candidate competition and whole-tree scoring can correct Stage-2
event types and recover missed events without changing the existing Stage-2
pipeline.

## Isolation boundary

All new implementation lives below `tasks/stage3-tree-refinement/`. Stage-2
scripts remain read-only. Stage-3 imports only stable helpers from Stage-2:
`linearize.parse_forest`, `align_events.align_forest`, ontology/model helpers,
and the official evaluator. This keeps the running Experiment C queue and its
provenance reproducible.

The single RTX 4090 is currently occupied by C, so no Stage-3 GPU process may
start until C reports `completed`, its queue has exited, and `nvidia-smi` shows
no project GPU process. This is a resource gate for every GPU phase; the
logical data dependency on C applies specifically to D1/D2. F0/F1 have no C
data dependency but still wait for the GPU resource. CPU-only tests and
data-shape checks are safe while C is running. All GPU commands use
`/home/myluo/miniconda3/envs/qwen3meld-run/bin/python`, named tmux sessions,
saved logs/exit JSON, no package installation, and one model in GPU memory at
a time during training.

## Data flow

1. D1 reuses T0 greedy predictions as c0 and samples c1..c5 with
   `do_sample=True`, temperature 0.8, top-p 0.95, five return sequences,
   seed 42, and max-new-tokens 768. It aligns with the exact T0 parameters,
   deduplicates by `event_key`, records support 0..6, labels candidates, and
   reports support thresholds, diversity, and the exists-only oracle.
2. D2 creates forced-type continuations with the complete prior greedy prefix,
   fixed trigger, 160-token subtree stopping, dev all-10-type candidates and
   test greedy-plus-dev-confusion candidates. It scores sum and mean likelihood
   including forced type/trigger tokens, and includes the schema-filtered
   type-only control.
3. SHA256(doc_id) parity creates two document-disjoint folds. Each fold model
   predicts the other fold; those pools plus type-competition candidates and
   at-most-30% tagged synthetic negatives form reward data.
4. Qwen3-0.6B (and the RoBERTa-large 512-token no-context ablation) score
   `exists`, `type_ok`, and `event_score`; fixed formula
   `R=P(exists)*P(type_ok)*(0.5+0.5*event_score_hat)` drives R1, R0, and R01.
5. R1 reuses exactly D2 candidates. R0 scans dev AC over tau 0.10..0.90 by
   0.05 and applies the specified same-trigger retention rule; test uses only
   the frozen choice. Selection outputs are aligned SciEvents predictions and
   diagnostics. A single report is generated only from JSON artifacts under
   `runs/`.

## Component boundaries

- `common.py`: import bridge, identities, alignment/labels, provenance.
- `sample_pool.py`: deterministic greedy/sampled inference and pool merge.
- `type_competition.py`: prefix validation, forced subtree generation, and
  likelihood scoring.
- `build_folds.py` / `build_reward_data.py`: leakage-safe folds and labels.
- `train_reward.py` / `score_candidates.py`: multi-head reward model training
  and inference with explicit length audits.
- `select_r1.py` / `select_r0.py`: candidate selection and official-format
  output, with dev-only threshold tuning.
- `diagnostics.py`: T0-relative correction/recovery accounting.
- `run_queue.py` / `write_report.py`: resumable experiment orchestration and
  artifact-only reporting.

## Error and reproducibility policy

- No silent truncation. Any overlength input fails the audit and records the
  offending IDs.
- Dev determines confusion sets, threshold, and any selection choice; test is
  evaluated once per frozen configuration.
- Every run records command, seed, data/model/script hashes, status, exit code,
  and timings. Atomic state writes prevent stale completion markers.
- Existing T0 alignment defaults are never changed by this work.
- `data/confusion_sets.json`, `data/folds.json`, and `data/manifest.json` are
  frozen provenance artifacts; test never recomputes dev choices.
- D1 must apply its +6 trigger-recall / +3 low-diversity criteria, D2 must
  report prefix mismatch and type-correction/corruption counts, and all R
  phases must report official, complex-scenario, timing, peak-memory, and
  T0/D2/EXCEEDS comparison values.

## Verification strategy

The implementation starts with unit tests for labels, exact T0 alignment
parameters, pool support, forced-prefix token consistency, forced-token
likelihood accounting, parenthesis/schema handling, reward-data leakage and
synthetic caps, same-trigger selection, dev-only tau tuning, stale run-state
rejection, and artifact-only report generation. Before any GPU run, tests,
import compilation, gold/shape/length audits, official scorer fixtures, and
the C/GPU-idle gate are checked. D1 and D2 dev results are produced and
reported before later reward-model phases proceed. The final report must be
recomputable from `runs/*` JSON and must state that R2, multi-seed, and learned
tau/weights are out of scope.
