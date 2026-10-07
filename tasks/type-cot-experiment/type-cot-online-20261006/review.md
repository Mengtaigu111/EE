# Independent SPEC and code review

Reviewed 2026-10-06, read-only; no GPU use and no implementation edits.

Verdict: no blocking defect found in the reviewed implementation. This approves proceeding to the required GPU smoke, not a claim that GPU behavior has already passed.

## Recipe and control checks

- CoT explicitly constructs a fresh base model and new r16/alpha32/dropout .05 LoRA, trains only assistant tokens, uses seed42 shuffled coverage, one epoch/802 updates, LR1e-4 cosine/warmup25, and fixes selection to the final checkpoint. No historical adapter is loaded.
- Tree requires the same run CoT final adapter, verifies completion and its actual file hash, and at train entry compares every loaded adapter tensor against the source. Empty optimizer state, scheduler update zero/plateau scale one, and a pristine feedback controller are checked after trainer initialization.
- Frozen AppendixNoCopy data and Sorted1 sampler are retained. Dense checks are exactly 401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208. Warmup97, LR1e-4, min_delta .001, reduce at bad count3, stop at6, and no reset on reduction are retained.
- Evaluation supplies all18 official metrics and uses official offset dev AC_f1 to control training. The full1593 dev rows are independently checked, and the verifier recomputes official scores and feedback decisions/checkpoint LR states.
- Test runs after completed tree training verification and loads only the dev-best checkpoint. Test scores cannot enter the feedback closure.
- Smoke selects longest32 CoT rows for two updates/one epoch and longest32 sorted tree rows for four updates/two epochs. Dev is the first two complete abstracts. Reload uses checkpoint4 and requires exact raw generated output agreement.

## Independent evidence

Compared all train/dev/test CoT rows with the original Type-CoT source: every field except prompt is unchanged, including every completion. Counts12828/1593/1627 and current user-final output hashes match the dataset manifest. Reviewed the inherited sampler, online trainer, feedback scheduler, safe checkpoint implementation, dev scorer, fixed generation wrapper, and new CPU tests.

## Limits

CPU tests and GPU smoke execution belong to the root implementation workflow; this review did not run them. Metadata hashes cover the base-model config/tokenizer config/index, while source-adapter tensor identity is verified directly at handoff. Base-model shard hashes are not part of the new manifest, consistent with the existing recipe; this is a provenance limitation rather than an observed functional defect.
