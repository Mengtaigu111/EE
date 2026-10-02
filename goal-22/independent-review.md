# Independent Easy2 review

Result: no blocking implementation issue found. Review covers the fresh-start plan,
new builder/manifest, runner, verifier, launch scripts, and reused trainer/phase.
GPU smoke and final full-run verification remain required before their respective
success claims.

## Schedule and initialization

- Scratch Stage1 passes no `--resume`; unchanged phase code rejects any baseline
  initial adapter and builds raw Qwen3-8B with the original LoRA recipe.
- Independent executed assertions confirm `2 * ceil(6504/16) = 814`, remaining
  Stage2 updates3196, total/horizon4010, checkpoints802/814/1604/2406/3208/4010.
- Runner and verifier check actual phase step counts and Stage1 epoch2.0. Verifier
  additionally requires Stage1 start0 and configured resumeNone. Smoke deliberately
  exercises unequal2+4 updates rather than the full two-epoch data schedule.
- Stage2 resumes the new814 checkpoint, uses changed fulltrain with
  `ignore_data_skip=True`, and restores model/optimizer/scheduler/RNG. Its inherited
  Trainer epoch value is bookkeeping, not a phase-local coverage claim.
- Stage2 loss correction is reused from the existing phase script: Trainer summary
  loss is scaled by global_stop/(global_stop-global_start), hence4010/3196.

## Independent executed evidence

- `test_easy2_scratch_schedule.py`: CPU integration passed. A5-row dataset with
  accumulation2 trains exactly10 real examples in two complete permutations,
  flushes the partial final accumulation, and ends at epoch2.0/step6/horizon9.
  Changed7-row Stage2 resumes6→9, trains6 unique examples, restores all four state
  groups exactly, retains nonzero boundary LR, and seals valid checkpoint9.
- Separate source/data hash verification passed against every new manifest input
  and snapshot hash. The reused `train_user_v2_phase.py` and
  `curriculum_trainer.py` are byte-identical to the old UserV2 run source snapshot.
- `goal-22/tests.log` records6 passing tests, including the CPU integration above.

## Evaluation and remaining acceptance

Runner evaluates full dev at814, subsequently evaluates every retained checkpoint,
selects official dev AC_f1 with later-step tie break, and evaluates test once for
the selected checkpoint. Verifier checks all prediction IDs, recomputes official
metrics, checks checkpoint inventory/integrity, and verifies report score rows.
New canonical report location avoids overwriting the older experiment report.

The confirmed Transformers same-data mid-epoch partial-accumulation resume bug is
not entered: Stage1 starts from scratch, and Stage2 changes data without skipping.
No shared environment or trainer patch is required. Before launching main, the
new Qwen smoke must pass its independent verifier. Final4010-step results remain
pending until the queued final verifier exits successfully.
