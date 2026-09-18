# Experiment C Implementation Plan

> **For agentic workers:** Implement this plan task-by-task with tests before production changes.

**Goal:** Implement document-conditioned, sentence-level Stage-2 event generation from `task/0918-experiment-C-short.md`, including dataset construction, target-scoped alignment, deterministic document merge, official evaluation, and a runnable C queue without changing T0 behavior.

**Architecture:** Keep the existing document-level Stage-2 pipeline intact. Add a sentence dataset builder that emits one row per document sentence with the full abstract as context and a local event forest as the target. Add a merge module that combines all sentence predictions, aligns triggers only inside their target sentence while aligning arguments over the full document, and emits document-level aligned predictions accepted by the existing scorer. Add a C-specific queue wrapper that audits lengths, trains with an explicit step budget, evaluates saved checkpoints on dev, selects the best checkpoint by AC F1, and evaluates test once.

**Tech Stack:** Python 3.10, JSONL, existing Qwen3/LoRA trainer and greedy generation, existing S-expression parser/linearizer, EXCEEDS `utils.calculate_F1`, unittest/pytest.

---

### Task 1: Record scope and identify test seams

**Files:**
- Modify: `tasks/todo.md`
- Modify: `findings.md`

- [ ] Record the approved Experiment C design, non-goals, and validation gates.
- [ ] Record that the implementation must preserve T0 behavior and exclude formal 8B training from this coding pass.

### Task 2: Add failing dataset-construction tests

**Files:**
- Create: `tasks/stage2-event-tree/tests/test_sentence_dataset.py`
- Test: `tasks/stage2-event-tree/scripts/build_sentence_dataset.py`

- [ ] Test one row per sentence, stable `document_id`/`sentence_index`, target-only event selection, `(ROOT)` for no-event sentences, and prompt markers.
- [ ] Test cross-sentence arguments remain in the target event completion and document token/sentence metadata is retained.
- [ ] Run the new test and verify it fails because the module does not exist.

### Task 3: Implement sentence dataset construction

**Files:**
- Create: `tasks/stage2-event-tree/scripts/build_sentence_dataset.py`
- Modify: `tasks/stage2-event-tree/scripts/prompts.py`

- [ ] Implement sentence boundary validation and event partitioning by trigger start offset.
- [ ] Emit train/dev/test JSONL rows with full abstract, target sentence, target bounds, local `completion`, original document metadata, and provenance manifest/statistics.
- [ ] Use the normal event schema and no Stage-1 hints.
- [ ] Make the CLI default to an untracked C data directory and support explicit project/input/output paths.
- [ ] Run dataset tests and verify they pass.

### Task 4: Add failing target-scoped alignment and merge tests

**Files:**
- Create: `tasks/stage2-event-tree/tests/test_sentence_merge.py`
- Test: `tasks/stage2-event-tree/scripts/align_events.py`
- Test: `tasks/stage2-event-tree/scripts/merge_sentence_predictions.py`

- [ ] Test repeated trigger text: only the occurrence in the target sentence may align.
- [ ] Test arguments may align in another sentence and same-text subevent arguments can reuse an aligned trigger offset.
- [ ] Test deterministic sentence-order merge, duplicate prevention, missing/duplicate sentence detection, and document IDs.
- [ ] Run the tests and verify they fail for the missing scope/merge behavior.

### Task 5: Implement target-scoped alignment and merge

**Files:**
- Modify: `tasks/stage2-event-tree/scripts/align_events.py`
- Create: `tasks/stage2-event-tree/scripts/merge_sentence_predictions.py`
- Modify: `tasks/stage2-event-tree/scripts/evaluate.py`
- Modify: `tasks/stage2-event-tree/scripts/predict.py`

- [ ] Extend alignment with optional per-event trigger scopes while preserving current default T0 behavior.
- [ ] Add document-wide argument alignment mode for C and retain global subevent trigger reuse.
- [ ] Merge parsed sentence predictions into one document record with aligned events, diagnostics, raw outputs, and truncation counts.
- [ ] Let evaluation consume pre-aligned C predictions without re-aligning triggers under document-level defaults.
- [ ] Preserve sentence metadata in prediction records and make `max_new_tokens` configurable.
- [ ] Run unit tests and verify all relevant existing Stage-2 tests remain green.

### Task 6: Add C queue and length/checkpoint controls

**Files:**
- Create: `tasks/stage2-event-tree/scripts/run_c_queue.py`
- Modify: `tasks/stage2-event-tree/README.md`

- [ ] Audit sentence-level training lengths before training and derive an explicit generation budget from the audit rather than reusing T0's constant.
- [ ] Support an explicit `--max-steps` budget so C can be compared fairly with T0; record row count, sentence count, and effective steps.
- [ ] Train with existing LoRA settings, evaluate every saved checkpoint on dev after sentence merge, select highest AC F1, then run test once.
- [ ] Enforce tmux for GPU execution, durable run metadata, no overwrite of existing outputs, and clear commands for build/audit/train/eval.

### Task 7: Gold roundtrip and minimal real-data verification

**Files:**
- Create/modify: `tasks/stage2-event-tree/tests/test_sentence_merge.py`
- Create: `tasks/stage2-event-tree/verification/experiment-c-gold-roundtrip.json` (generated, not committed if it is a large artifact)
- Modify: `tasks/todo.md`

- [ ] Run gold sentence trees → merge → official scorer on dev.
- [ ] Verify trigger scope, argument alignment, no-event rows, duplicate-trigger documents, and cross-sentence arguments.
- [ ] Run a small real-data build/audit smoke without loading model weights.
- [ ] Record exact commands, pass/fail evidence, and any known limitation; do not launch the formal 8B queue in this pass.
