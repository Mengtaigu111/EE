# UserV2 Easy2 Git update

- [x] Scope checked: curriculum code, approved prompt, configs, fixed IDs, difficulty analysis, lightweight evidence and required Stage2 prompt/parser changes.
- [x] Model/checkpoint files, raw training JSONL, caches, live logs and unrelated experiments excluded.
- [x] Stage2 + curriculum + difficulty CPU verification: 62 passed, 8 subtests passed. Command: `CUDA_VISIBLE_DEVICES='' /home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m pytest tasks/stage2-event-tree/tests tasks/stage2-event-tree/curriculum tasks/stage2-event-tree/difficulty-analysis -q`.
- [x] Runtime source hashes unchanged. Frozen CSV CRLF and approved prompt Markdown hard breaks preserved.
- [x] Required exact Appendix F2 reference added after dependency review.
- [ ] Commit and push current experiment branch; push currently blocked by missing GitHub HTTPS credentials (`gh auth status`: no login).

Initial test command did not hide the active GPU; three tiny resume tests touched CUDA RNG restoration and failed with GPU OOM. Isolating this test process with CUDA_VISIBLE_DEVICES resolved all failures without source changes or interrupting the main run. Frozen datasets/base model remain local prerequisites for full-data verification and training.
