# Lightweight CoT to event tree implementation plan

User approved on 2026-10-06. Implementation proceeds without another approval.

1. Freeze a new outer manifest with existing online-dev manifest nested as
   `tree_reference`, CoT source paths and user-final prompt hashes; lock all
   train/dev/test files and source dependencies. Preserve old manifests.
2. CoT: fresh Qwen3-8B/new LoRA, r16/alpha32/dropout.05, bf16, max4096,
   batch1/accum16, seed42, one full shuffled epoch=802 updates, LR1e-4 cosine
   warmup25, only assistant loss. Save final adapter and training evidence.
   Dev type multiset F1 at completion is diagnostic; fixed final weights used.
3. Tree: load only this run's CoT final adapter trainably on original base,
   verify matching hashes/weights, fresh optimizer/scheduler/feedback, seed42.
   Original AppendixNoCopy/Sorted1 course; horizon3208, warmup97, plateau LR.
   Validate at [401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208].
   Keep original min_delta.001/reduce3/stop6; no reset on reduction.
4. Test only dev-selected final tree after training verification, original
   repaired official offset evaluation plus separate diagnostics. Save all18
   metrics, raw generation and alignment, full safe optimizer checkpoints.
5. CPU: cadence boundaries, retention and earlystop, wrong adapter/source
   rejection, CoT/tree gold agreement. GPU smoke: longest32 samples in CoT
   two updates/one epoch, longest32 Sorted1 two epochs/four updates, eval2/4
   on two complete dev abstracts. Reload checkpoint4 independently and match
   raw outputs; check CoT weight handoff and optimizer reset.
6. Independent spec+quality review then smoke; no formal launch until all pass.
   Exclusive run dirs, source snapshots, tmux launcher exit files. Formal
   status confirmed from actual progress and GPU PID; runtime estimate from
   smoke and prior full-dev timings, not assumed linear speedup.

Files: config.py (frozen specs/data), train_cot.py (fresh first stage),
train_tree.py (second stage), dense_trainer.py (explicit check schedule),
run.py (two-stage orchestration), verify.py (independent checks), tests/.
Root handles config, first stage and orchestration; implementation subagent
handles tree trainer and explicit cadence; independent reviewer handles review.
