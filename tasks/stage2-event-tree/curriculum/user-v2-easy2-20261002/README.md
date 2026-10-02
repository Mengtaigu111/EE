# UserV2 easy-two-epoch curriculum

Fresh seed-42 initialization from `/opt/models/Qwen3-8B`; no adapter or existing checkpoint is loaded in Stage 1. Stage 1 trains the unchanged 6,504 easy + ROOT records for two epochs: `2 * ceil(6504 / 16) = 814` optimizer updates. Stage 2 continues the full optimizer/scheduler/model/RNG state on the unchanged 12,828-record full train set for 3,196 further updates. Total cosine schedule remains 4,010 updates.

All training/dev/test data and frozen membership files are reused from `../user-v2-20261002/` without reserialization, filtering or resampling. `prompt.md` is a byte-identical snapshot of the approved prompt; `parent_manifest.json` preserves the prior UserV2 manifest bytes. The new manifest records the original source/output hashes plus this schedule. Builder reruns are idempotent and refuse files with different content.

Checkpoints retain global steps 802/814/1604/2406/3208/4010; select official dev AC F1 and evaluate test only for the selected model. Prior AppendixNoCopy comparison also changes the prompt, so it cannot isolate curriculum effects.
