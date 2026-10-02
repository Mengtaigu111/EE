# User-v2 prompt mean curriculum replacement

Goal: Use user-approved prompt task/prompt/1002-event-forest-v2.md for the already-specified single mean curriculum run. User explicitly requests NEW code/prompt without overwriting originals, same training data/schedule, and review before launch.

Scope: Stop superseded AppendixNoCopy curriculum GPU process safely, retain its code/prompt/data/logs/checkpoints. New builder, launcher, phase runner, verifier, tests under tasks/stage2-event-tree/curriculum/; new prompt snapshot and artifacts under curriculum/user-v2-20261002/; new all-split dataset data/sentence-c-user-v2-20261002/; new run C-CL-Mean-UserV2-42-20261002. Never edit old builders/runners/shared prompts or baseline data. Recipe reused from audited AppendixNoCopy baseline (raw Qwen3-8B,seed42,bf16,LoRA16/32/.05,batch1 accum16,AdamW1e-4,cosine4010,warmup.03,maxlen4096).

Data spec: original IDs, sentences, bounds, tokens, event counts, gold/targets/completion identical across train/dev/test; only prompt and metadata mode replaced. B1=5427 B2=5277 ROOT=2124; reuse exact existing Stage1 IDs6504 and ROOT1077 in same order, no resampling, no difficulty/threshold changes. Stage2 is all12828 and byte-identical newly-rendered full train. Training2005+2005 global4010, Stage2 fullstate resume, same cadence and selectiondevAC. Stage1-end dev and final/best metrics, besttest once. Controls only prepared, not trained.

Prompt: snapshot exact approved bytes; fill only full abstract sentence markers, target index and target text placeholders. No extra rules, grammar repairs, schema changes or labels. Training/eval always use new prompt for all splits.

Verification: TDD real-data transformation/gold preservation/ID membership/template injection. Reuse tested safe fullstate Trainer (read-only), new tiny phase runner smoke longest32 actual examples two+two steps (no research claims). New runner/report/verifier tests; independent reviewer before main launch. Verify input/source hashes and full token audit no truncation, rawlossfinite. Final verify independent official metrics recompute and exact steps/state/predictioncoverage. Preserve failed attempts and exits. Record package/GPU/log hashes. Newprompt+curriculum compared with historical AppendixNoCopy changes two factors; report does not attribute delta solely to curriculum, no extra baseline retraining authorized.

Tasks:
- [x] Stop superseded old GPU run, preserve all outputs, record stopped step and reason separately.
- [x] Build newprompt data and immutable manifest; verify exact gold/splits/ROOT/order.
- [x] Write new runner/phase/verifier/report; test and independent review.
- [x] Run all-split token audit and real longest-sample tmux GPU smoke.
- [x] Launch new main training/evaluation in tmux after checks; no old adapter resume.
- [ ] Main training/evaluation complete (currently running; user requests startup and ETA).
- [ ] Independently verify actual outputs and archive report/limitations.
