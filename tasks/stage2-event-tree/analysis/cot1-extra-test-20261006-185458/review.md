# Supplementary checkpoint evaluation review

Read-only code and metadata review, 2026-10-06. No GPU launches or locked-source edits.

Verdict: no launch-blocking defect found. Runner evaluates only the requested existing checkpoints1804/1604/3208 into separate exclusive outputs; it does not retrain, rank checkpoints by test, or modify historical selection. Batch4, seed42, greedy bf16/no-thinking generation and max_new_tokens319 preserve the original test protocol.

Artifact verification binds generation to the exact checkpoint/hash and frozen test input; requires1627 ordered unique sentence IDs and250 complete documents; verifies metadata, completion/hash, repaired parsing and raw-to-merged consistency; recomputes all18 official metrics,7 complex metrics and four separate score modes; compares saved SciEvents/per-document artifacts. The tree/checkpoints adaptation and explicit batch-size parameter are correct.

Independently checked all136 manifest source/input/checkpoint locks successfully. Reference2205 verification records full coverage and all metric checks passed. Benchmark evidence supports retaining batch4: batch8 saved7.35% wall time but changed2/48 raw and parsed trees. Queue state was running checkpoint1804 with no completed results at review time; supplementary completion is not yet claimed.

Scientific limitation: supplementary test comparisons are exploratory and must remain separate from the original dev-selected2205 result. No extra experiment scope requested.
