# Stage-3 Tree Refinement v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the isolated Stage-3 v1 diagnostic and refinement pipeline from D1/D2 through R1/R0/R01, without changing Stage-2 behavior.

**Architecture:** All new code is under `tasks/stage3-tree-refinement/`. Small pure helpers handle candidate identity, alignment, labels, diagnostics, folds, and selection; GPU-facing scripts call those helpers and record immutable run artifacts. Stage-2 parsing/alignment/model utilities are imported read-only through an explicit path bridge.

**Tech Stack:** Python 3.10, PyTorch/Transformers/PEFT from `qwen3meld-run`, pytest, JSONL/JSON, existing EXCEEDS official scorer, tmux for GPU queues.

---

## Guardrails for every task

- Do not modify `tasks/stage2-event-tree/scripts/`.
- Do not start Stage-3 GPU work while `stage2-c` is running. CPU unit tests and static audits are allowed.
- Use test-first cycles for pure logic: write the failing test, run it, implement, rerun the focused test, then run the full Stage-3 suite.
- Every generated artifact must be deterministic for a fixed seed and must reject silent truncation.

## Errors encountered

| Error | Attempt | Resolution |
|---|---|---|
| Stage-3 `run_queue` tests imported Stage-2 `run_queue` | Initial full-suite run | Changed the read-only Stage-2 path bridge from front insertion to append, keeping Stage-3 modules ahead and preserving explicit Stage-2 imports. |

### Task 1: Scaffold, import bridge, and provenance

**Files:**
- Create: `tasks/stage3-tree-refinement/README.md`
- Create: `tasks/stage3-tree-refinement/scripts/__init__.py`
- Create: `tasks/stage3-tree-refinement/scripts/common.py`
- Create: `tasks/stage3-tree-refinement/tests/conftest.py`
- Create: `tasks/stage3-tree-refinement/tests/test_common.py`

- [ ] Write failing tests for importing Stage-2 helpers, canonical event/trigger keys, atomic JSON writes, and script/data SHA256 provenance.
- [ ] Run `pytest -q tasks/stage3-tree-refinement/tests/test_common.py` and verify the expected missing-symbol failures.
- [ ] Implement the path bridge, JSONL readers/writers, stable identities, safe atomic state writes, and provenance helpers.
- [ ] Rerun the focused test and then compile all new scripts.

### Task 2: Candidate labels and relative diagnostics

**Files:**
- Create: `tasks/stage3-tree-refinement/scripts/diagnostics.py`
- Create: `tasks/stage3-tree-refinement/tests/test_labels.py`

- [ ] Add tests for exact `exists`, `type_ok`, argument P/R/F1 labels, same-trigger multiple gold events, and false-positive candidates.
- [ ] Run the tests red.
- [ ] Implement gold indexing, candidate labels, event score calculation, T0-relative correction/corruption/recovery counters, and per-pair counts.
- [ ] Verify focused tests and official metric input shape on synthetic rows.

### Task 3: D1 pool merge and support accounting

**Files:**
- Create: `tasks/stage3-tree-refinement/scripts/sample_pool.py`
- Create: `tasks/stage3-tree-refinement/tests/test_pool_merge.py`

- [ ] Test deterministic event-key deduplication, support counts across c0..c5, forest diversity, and preservation of source metadata.
- [ ] Run the tests red.
- [ ] Implement pool merge/labeling and a Qwen sampling CLI that reuses T0 greedy predictions and records no-truncation diagnostics.
- [ ] Fix the protocol to c0 greedy plus c1..c5 sampled with `do_sample=True`, temperature 0.8, top-p 0.95, `num_return_sequences=5`, seed 42, `max_new_tokens=768`, and `enable_thinking=False`.
- [ ] Write the exact per-document `pool.jsonl` layout: candidates, `event_key`, `trigger_key`, support 0..6, labels, and source sample IDs.
- [ ] Add dev/test summary JSON generation for support thresholds, trigger/(type,trigger) recall and precision, official TI/TC/AI/AC, exists-only oracle, duplicate-forest rate, and unique-forest diversity.
- [ ] Enforce the +6 trigger-recall criterion and conditional dev-only `num_return_sequences=10`, temperature 1.0 fallback when support≥1 gain is below 3 points.
- [ ] Run pure merge tests without loading a model.

### Task 4: D2 type competition and prefix safety

**Files:**
- Create: `tasks/stage3-tree-refinement/scripts/type_competition.py`
- Create: `tasks/stage3-tree-refinement/tests/test_prefix_tokens.py`
- Create: `tasks/stage3-tree-refinement/tests/test_type_competition.py`

- [ ] Test token-prefix equality against greedy output, forced-type candidate construction, balanced-parenthesis stopping, and invalid-schema role dropping.
- [ ] Run the tests red.
- [ ] Implement dev-derived confusion-set construction, forced subtree generation, sum/mean token log-likelihood scoring, and D2 replacement variants.
- [ ] Run dev with all 10 types and the confusion-set candidate set; run test only with greedy type plus the frozen dev confusion set.
- [ ] Require every forced prefix to contain all preceding greedy events and the fixed trigger text; log and reject an unclosed subtree at the 160-token cap.
- [ ] Add explicit 160-token/balanced-parenthesis stopping, forced type/trigger likelihood accounting, per-event batching/timing, and prefix mismatch fallback to greedy token IDs when mismatch exceeds 1%.
- [ ] Implement the illegal-schema “only replace type” control by dropping roles not allowed by the selected type, and record its invalid/drop rate.
- [ ] Emit the required D2 table, confusion-pair counts, prefix consistency, and `type_corrected > type_corrupted` dev criterion.
- [ ] Verify tests with a fake tokenizer/model fixture; do not use GPU.

### Task 5: Leakage-safe folds and reward data

**Files:**
- Create: `tasks/stage3-tree-refinement/scripts/build_folds.py`
- Create: `tasks/stage3-tree-refinement/scripts/build_reward_data.py`
- Create: `tasks/stage3-tree-refinement/tests/test_reward_data.py`

- [ ] Test SHA256 doc split parity, no document leakage, synthetic negative labeling, and the <=30% synthetic cap.
- [ ] Run the tests red.
- [ ] Implement fold manifests, held-out prediction pool assembly, type-competition source tagging, bounded synthetic negatives, and manifest hashes/counts.
- [ ] Train F0/F1 with `tasks/stage2-event-tree/scripts/train_lora.py`, max-len 2048, seed 42, five epochs, and fold-filtered `data/none/train.jsonl`; predict the opposite fold with greedy plus the same five-sample protocol, record held-out metrics, and apply the TC 3-point closeness warning rule against D1 dev.
- [ ] Run F-type only for `support>=1` greedy events: use each fold model to generate the same dev-frozen confusion-set type competitors for its held-out pool and tag candidates `source=type_competition`.
- [ ] Add exactly two synthetic negatives per gold event where possible: first confusable type with schema-mapped roles, then deterministic seeded random argument-branch deletion; cap total synthetic rows at 30% of real candidates and tag every row.
- [ ] Run tests against small fixture JSONL and verify fold disjointness.

### Task 6: Reward model training and scoring

**Files:**
- Create: `tasks/stage3-tree-refinement/scripts/train_reward.py`
- Create: `tasks/stage3-tree-refinement/scripts/score_candidates.py`
- Extend: `tasks/stage3-tree-refinement/scripts/common.py`
- Add: `tasks/stage3-tree-refinement/tests/test_reward_model.py`

- [ ] Test masked multi-head loss semantics, score formula `R`, output schema, and length-audit rejection using CPU fixtures.
- [ ] Run the tests red.
- [ ] Implement Qwen3-0.6B last-token representation with three heads and the RoBERTa-large `[CLS]` ablation, with the exact optimizer/loss masks from the spec.
- [ ] Build exact reward inputs with `Abstract`, greedy `Forest context` containing `[THIS]`, and `Candidate event` sections; derive Qwen p100 max length without truncation.
- [ ] Use the D1 dev pool as validation. Implement full-parameter bf16 Qwen training with lr 2e-5, cosine, warmup 0.03, 2 epochs, effective batch 32, seed 42; RB uses lr 1e-5, 3 epochs, `[CLS]`, 512-token no-context input, and reports overlength counts.
- [ ] Implement exact masked BCE/BCE/MSE training settings, Qwen last-token heads, and RoBERTa-large no-context input; report Qwen/RB dev AUC/Spearman.
- [ ] Implement score inference, AUC/Spearman summaries, and no-truncation audit artifacts.
- [ ] Run all CPU-compatible reward tests and compile the GPU entry points.

### Task 7: R1/R0/R01 selection and threshold tuning

**Files:**
- Create: `tasks/stage3-tree-refinement/scripts/select_r1.py`
- Create: `tasks/stage3-tree-refinement/scripts/select_r0.py`
- Extend: `tasks/stage3-tree-refinement/scripts/diagnostics.py`
- Create: `tasks/stage3-tree-refinement/tests/test_select.py`

- [ ] Test same-trigger retention rules, score threshold filtering, dev-only tau grid search, R1 candidate reuse, and oracle selection.
- [ ] Run the tests red.
- [ ] Implement R1 type replacement, R0 pool filtering, R01 composition, official prediction serialization, and diagnostics attachment.
- [ ] Ensure R1 consumes exactly the D2 candidate files, reports D2-vs-R1, T0-vs-R1, and candidate oracle rows; ensure R0 keeps same-trigger different types only when both exceed tau.
- [ ] Require the R1 best setting to be compared against and reported relative to the best D2 setting; do not imply R1 wins if it does not.
- [ ] Implement dev AC grid `0.10..0.90` step `0.05`, frozen test tau, R01 composition, and RB R1/R0 runs.
- [ ] Verify deterministic tie breaks, test mode cannot tune tau/confusion sets/candidate counts, and each test configuration is evaluated once.

### Task 8: Queue, reports, and resumability

**Files:**
- Create: `tasks/stage3-tree-refinement/scripts/run_queue.py`
- Create: `tasks/stage3-tree-refinement/scripts/write_report.py`
- Extend: `tasks/stage3-tree-refinement/README.md`
- Add: `tasks/stage3-tree-refinement/tests/test_queue_state.py`

- [ ] Test atomic `run.json` transitions, stale-attempt rejection, explicit tmux requirement for GPU phases, and report-only-from-artifacts behavior.
- [ ] Run the tests red.
- [ ] Implement phase-gated queue commands, per-stage logs/exit JSON, C completion/GPU-idle gate, D1/D2 dev stop point, checkpoint provenance, and resumable downstream stages.
- [ ] Implement report tables required by the spec, including complex test metrics and explicit v1 omissions.
- [ ] Include D1/D2/F/R tables, tau curve, complex metrics, T0/D2/EXCEEDS comparison, correction diagnostics, timing/peak memory, R1/R0 oracle rows, and v1 omissions.
- [ ] Run queue tests without starting GPU work.

### Task 9: Full pre-GPU verification and handoff

**Files:**
- Modify: `tasks/todo.md`
- Modify: `tasks/lessons.md` only if a new correction is received
- Create: `tasks/stage3-tree-refinement/verification/` artifacts as needed

- [ ] Run the complete Stage-3 unit suite.
- [ ] Run `python -m compileall` on Stage-3 scripts.
- [ ] Run data/path/model availability audits without GPU generation.
- [ ] Confirm `stage2-c` completion status and no project GPU process before any tmux launch.
- [ ] Update the single task checklist with implementation status, verification evidence, and the first queue command.
- [ ] Review the diff for accidental Stage-2 changes and untracked-output contamination.
- [ ] Hash every tracked Stage-2 script before and after implementation and run the existing Stage-2 test suite or documented baseline check.
- [ ] Run one fixture end-to-end path: parse → exact T0 alignment → labels → official scorer input → report JSON.
- [ ] Assert every document-level Stage-3 alignment call is `align_forest(tokens, sentences, events)` with no `trigger_scopes` and `argument_sentence_preference=True`.
- [ ] Pin and verify `tasks/stage2-event-tree/runs/C-42-20260918/run.json` status, no project process in `nvidia-smi`, the fixed conda interpreter, no package installation, and one GPU model/job at a time.
- [ ] Freeze hashes for confusion sets, tau, candidate counts, and seeds; reject test runs with mismatches or an existing test-evaluation marker.
- [ ] Invoke and preserve `eval_complex_quick_f1.py` output for each final test configuration.
- [ ] Enforce the fixed interpreter `/home/myluo/miniconda3/envs/qwen3meld-run/bin/python`, prohibit package installation, and refuse concurrent GPU jobs/models.

## Planned commits

1. `Add isolated Stage-3 foundations and candidate diagnostics`
2. `Implement Stage-3 sampling and type competition`
3. `Implement Stage-3 reward scoring and selection`
4. `Add Stage-3 queue reporting and verification`

## Acceptance criteria

- Stage-2 files are byte-for-byte unchanged by the implementation.
- Pure Stage-3 tests pass and all scripts compile.
- Gold/T0 fixture labels and selection results are reproducible.
- No silent truncation path exists in data, generation, or reward-model input.
- F0/F1, F-type, Qwen/RB reward models, R1/R0/R01, D1/D2 tables, oracle rows, tau curve, complex test metrics, and report JSON are all represented by verifiable run artifacts.
- D1 oracle selects the maximum `event_score` candidate per gold `(type, trigger_key)`; D2 reports unclosed-budget drops and R1 explicitly compares to the best D2 setting.
- GPU execution is refused until the current C run is complete and the GPU is idle, and is only launched through named tmux with run state and exit artifacts.
