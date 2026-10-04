# Sorted-first full-train curriculum implementation plan

Goal: raw Qwen3-8B + original AppendixNoCopy prompt; all train first epoch sorted from low to high mean difficulty, ROOT evenly interleaved, tied scores seeded shuffle; four later epochs shuffled. Same 4010 updates, same LoRA/optimizer/cosine/seed42, no existing sources overwritten.

Architecture: new byte-preserving data builder and manifest; custom epoch-aware Sampler used by one continuous safe CurriculumTrainer (no stage reload). First epoch Sequential over frozen sorted JSONL; later epochs seeded randperm. Actual collator records row IDs/epochs to prove order and coverage. Separate runner evaluates 802/1604/2406/3208/4010 on repaired full dev; select offset AC F1, later-step tie; selected test offset + raw-token official metric. Keep previous results immutable.

Files:
- curriculum/build_sorted_first_data.py: original row bytes reordered, ordered IDs, ROOT/tie/coverage/source hashes verification, original prompt snapshot, new sorted-first-20261003/manifest.json.
- curriculum/train_sorted_first.py + sorted_first_sampler.py + test_sorted_first_sampler.py: continuous training, real sampler test across Accelerator/Trainer epochs, observed-order audit, GPU smoke 32 rows/4 updates (epoch boundary2), final config/runtime/checkpoints.
- curriculum/run_sorted_first.py + verify_sorted_first_run.py + test_sorted_first_runner.py: pipeline, manifest/hash locks, metrics and selected raw-token results, completion verification.
- goal-24/run-{smoke,main}.sh, logs, exit markers, review/verification artifacts.

Contract: manifest keys experiment, seed, baseline_run_file, data_dir (original split JSONL), source_train_file, sorted_train_file, ordered_ids_file, prompt_template_file, total_optimizer_steps=4010, effective_batch_size=16, steps_per_epoch=802, checkpoint_steps=[802,1604,2406,3208,4010], source_hashes, output_hashes. Trainer CLI --data --output --baseline --total-steps --save-steps [--smoke]; smoke chooses longest32 then difficulty order preserving selected source order, maxsteps4 saves2. Runner --run-dir --manifest [--smoke] [--batch-size4]; smoke skips full inference but verifies checkpoint/order. Final verifier CLI --run-dir checks all source locks, observed first epoch sequential/later random permutation, budgets, checkpoints, selecteddev/test metrics, token source integrity.

Checks:
- [x] TDD builder tie/ROOT placement and full exact data equality by ID; tokenizer/data audit reuses frozen audited rows (no gold changes).
- [x] Epoch sampler TDD including actual CPU Trainer+Accelerate first and second epoch, every ID exactly once, first is sequential later differs.
- [x] Runner/verification TDD failure gates; old source preservation hash snapshot; independent spec and quality review.
- [x] GPU smoke tmux: originalmodel+same4096/bf16/r16 recipe, 4 steps crossing first epoch, finite loss, correct actual sequence, safe checkpoint verification.
- [x] Launch main in tmux only after all gates, confirm advancing global step and GPU allocation, report ETA from measured smoke/prior baseline.
- [x] Training/eval completion and independent final verifier: 2026-10-04 12:17 completed, main exit0, independent verifier passed.

Plan validation: no hard stage split, no random sampler overriding first order; batch1/accum16 unchanged. ROOT has no difficulty assigned, evenly interspersed with monotonic positive-score sequence. Sorting only first epoch. Token evaluation additionally reported; selection unchanged to avoid unapproved metric change. All external mutations excluded; user authorized implementation+tmux launch.
