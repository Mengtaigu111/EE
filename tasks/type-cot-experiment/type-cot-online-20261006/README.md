# Type-CoT1 → Sorted1 dense online dev

Entry point: run.py. Configuration: manifest.json. Approved recipe: plan.md.
Launch/status pointer: main-launch.json (session, absolute run/log/exit paths).

Phase1: fresh original Qwen3-8B/newLoRA, seed42, full12828 shuffled once,
802 updates, cosine LR1e-4/warmup25, user-final CoT prompt, original answers.
Fixed final adapter transfers to phase2. Full dev type-multiset evaluation
after phase1 is diagnostic only.

Phase2: fresh optimizer/scheduler/controller, same newly learned adapter,
original AppendixNoCopy/Sorted1 course, max3208/warmup97, dev AC controls LR
and stopping. Explicit checkpoints:
401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208.
An effective gain of .001 resets patience; bad3 halves LR, bad6 stops without
reset on reduction. Raw-best checkpoint ties choose later. Test is evaluated
only once after independently verified tree training and dev selection.

The tree TrainingArguments eval/save_strategy are `no` because periodic
modulo-based scheduling cannot express these steps. ExplicitEvaluationCallback
sets should_evaluate/should_save at the listed optimizer steps; evaluations
execute inside Trainer.train and affect the next update. This is online dev,
not post-training checkpoint evaluation. Raw data, all18 official dev metrics,
alignment, checkpoint full states and feedback records are retained.

Fresh handoff evidence is written on actual on_train_begin, not on object
construction: source tensors exactly equal loaded adapter tensors, empty
optimizer states, scheduler at0/plateau1, feedback empty/globalstep0. No earlier
experiment adapter or optimizer resume is used.

Validation: validation-summary.md; independent review.md and quality-review.md;
final-smoke-verification.json. Historical sources/prompts/gold/runs are retained.
Same-name legacy pytest modules are tested in separate processes, as actual
training/evaluation subprocesses are isolated too.
