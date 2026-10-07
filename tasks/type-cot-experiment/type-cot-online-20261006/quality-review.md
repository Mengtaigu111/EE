# Independent quality and scientific validity review

Reviewed 2026-10-06. Scope: new config/trainers/orchestrator/verifier/tests and inherited sampler, feedback, safe checkpoints and fixed scoring dependencies. Read-only review; no GPU launches or implementation mutations.

Verdict: no fatal training or import defect found. One medium-priority final-test verification gap should be closed before claiming the formal result verified. Smoke completion remains an independent launch prerequisite.

## Finding

**P2 — Final test verification does not bind offset predictions to generated raw outputs or dev-selected weights** (`verify.py`, `include_test` block). It recomputes four score modes, but offset scores consume the existing merged file while token scores consume raw generation independently. Two valid, same-coverage files from different runs can pass when their scores are recorded together. It also omits the prediction completion/hash check and an explicit equality between `run.best_checkpoint` and the checkpoint selected by independently replayed dev feedback. The orchestrator currently supplies correct paths, so this is a verification weakness, not evidence of wrong inference. Add raw-to-merged equality, completion/hash, selected-checkpoint/path checks and official saved artifact checks, following the historical independent verifier. Exact test sentence ID set/uniqueness is already enforced indirectly by raw-token scoring.

## Nonblocking observations

- Checkpoint LR verification uses `zip` without requiring equal scheduler/optimizer LR group counts or base LR equality. The inherited verifier checks both; retaining these assertions would prevent vacuous partial comparisons.
- Base model config/tokenizer config/index are hashed; tokenizer vocabulary and base weight shards are not. This is an inherited provenance limitation, with no observed altered model assets.
- Broad combined pytest collection has known same-name `build_dataset`/`evaluate` import collisions. Separate logs establish 54 online/core tests and 5 Type-CoT tests passing. Training and diagnostic GPU stages use separate processes, and fixed scoring explicitly loads origin-checked modules, so those collection failures do not establish a runtime collision.

## Verified by inspection/evidence

Fresh CoT base/new LoRA; complete shuffled assistant-only epoch; exact train-entry CoT tensor handoff; empty tree optimizer and fresh scheduler/controller; explicit dense checks; original sorted-first course; unchanged plateau rules and retained safe checkpoints; dev-only control; fixed final CoT diagnostic; test after independent training gate; exclusive tmux run/exit evidence and source snapshots. Existing smoke state shows CoT and diagnostic finished and tree training running, not completed.

Scientific interpretation is appropriately limited: single seed and changed feedback cadence do not isolate CoT causality. The report explicitly requires a same-schedule no-CoT comparison before attributing differences solely to CoT, without requesting an extra unapproved experiment.

## Closure review

Rereviewed the verifier fixes after the first smoke completed. The P2 finding is closed: final test now explicitly binds the selected checkpoint and exported best weights to replayed feedback, checks ordered complete sentence coverage and completion/hash, reparses raw outputs and requires exact offset-merged equality, and independently recomputes official metrics against both run state and saved metric artifact. The LR group/base-rate observation is also closed by nonempty equal group-count checks and fixed base-rate checks. Regression command independently rerun: `python -m pytest -q tasks/type-cot-experiment/type-cot-online-20261006/tests/test_verify.py` in `qwen3meld-run`; **3 passed**.

Quality verdict: pass, no outstanding blocking findings. No extra experiment scope recommended. Formal launch remains conditional on the planned fresh final-source two-stage GPU smoke passing; this closure does not claim that rerun has already completed.
