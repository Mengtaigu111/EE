# UserV2 two-easy-epoch curriculum implementation plan

> Execute task by task with independent review. User explicitly replaces checkpoint continuation with fresh training from raw Qwen3-8B. No shared code, old prompt/data/config/run overwritten.

**Goal:** Reduce exposure to easy subset while preserving total optimizer budget and exact task data.
**Architecture:** New run-local manifest and launcher/verifier under curriculum; reuse unchanged approved UserV2 phase recipe and safe Trainer, passing new phase boundary814. Fresh raw initialization, no existing checkpoint/adapter used. Stage2 full-state resume814 keeps total LR horizon4010.
**Stack:** Existing Python3.10/Torch2.4/Transformers4.52/PEFT runtime qwen3meld-run, singleRTX4090, tmux. No install/shared environment edits.

## Fixed specification

- Prompt snapshot exactly task/prompt/1002-event-forest-v2.md (hash8e6d508af5e9de15e001a8dba1da979d45353f334ef365d94edd87aa5254f7a2). Original five guidelines retained, no Region/SemanticDecision.
- Same train/dev/test12828/1593/1627, immutable gold/completions/IDs; no new filtering.
- Existing B1=5427, B2=5277, ROOT=2124; Stage1 saved IDs6504 with1077fixedROOT, Stage2all12828. Reuse all actual data bytes and ROOT IDs from UserV2 manifest, no resampling or formula/threshold edits.
- Effective batch16 =microbatch1*accum16. Stage1 updates/epoch=ceil6504/16=407 (last8 examples still get an update). Two epochs =814updates. Stage2=4010-814=3196 updates, about3.99 full passes.
- Original seed42, raw /opt/models/Qwen3-8B, LoRA16/32/.05 targets unchanged; assistant-only objective; bf16; AdamW1e-4, cosine horizon4010/warmup.03. No Stage1 trainedadapter init.
- Save baseline global cadence802 plus boundary814:802/814/1604/2406/3208/4010. Full dev at814, all retained checkpoints evaluated; select official AC F1 (laterstep tie). Test once selectedcheckpoint. Historical baseline prompt differs; report delta does not isolate curriculum.
- Stage2 resumes model/optimizer/scheduler/global/RNG from814; changed dataset uses ignore_data_skip=True, fresh shuffle. No LR/optimizer restart.

## Files / safe isolation

Create:
- curriculum/build_user_v2_easy2_config.py + test_user_v2_easy2.py
- curriculum/run_user_v2_easy2_curriculum.py
- curriculum/verify_user_v2_easy2_run.py
- curriculum/user-v2-easy2-20261002/{mean_curriculum_manifest.json,prompt.md,README.md}
- runs/C-CL-Mean-UserV2-Easy2-42-20261002/ (smoke separate)
- goal-22/run-{smoke,main}.sh and independent evidence
Read-only reuse old train_user_v2_phase.py, curriculum_trainer.py, evaluator, frozen UserV2 allsplit data. Newmanifest references same bytes, records oldmanifest/template/data hashes and newbudget. Controls remain prep-only. No git operations.

## Execution + acceptance checks

- [x] Read latest user instructions and validate old checkpoint inventory; pause obsolete2005-stepStage1 while planning. User subsequently overrides continuation: freshstart only.
- [x] Stop obsolete suspended parent/child while preserving old source/data/log/checkpoints; record superseded status separately. No resume802 or oldweights transfer.
- [x] TDD new config/boundary budget:2*ceil6504/16=814,3196remaining,4010total; exact dataset/prompt/root hashes remain equal.
- [x] New launcher/report/verifier remove hardcoded50/50 checks; positive arbitraryboundary validation; config and actual steps both checked. Training loss Stage2 corrected for remaining3196 denominator, not2005.
- [x] CPU integration: nondivisible dataset scratch Stage1 finishes exacttwoepochs and partial accumulation flushes; changed full-data resume continues correctglobalsteps and total LR. Checkpoints/metrics contain actual2.0 stage1epoch.
- [x] Independent review of config/data paths/phase boundary/fullstate resume/report.
- [x] New Qwen GPU smoke (4 or6totalupdates, unequal schedule) with exact prompt; verify model/optimizer/scheduler/global/RNG equality and nooverlength/NaN/OOM. Reuse unchanged allsplit token audit hash proof; no redundant new tokenizer audit needed unlessprompt/data changes.
- [x] Launch freshmain in namedtmux goal-22-user-v2-easy2-main only after tests/review/smoke pass. Confirm no resume/initadapterStage1 and first optimizersteps real.
- [x] Main queue trains814 +3196, evaluatesStage1/final/allcheckpoints, generates report and chains independent verifier. User asked tmux launch+ETA; hand back confirmedstartup, clearly finalmetrics pending.
- [ ] Final independent results (after main actualdone):checkpoint2.0epoch814,Stage2updates3196,global4010; exactstate recovery,sampler/datahash,predictioncoverage,officialmetricrecompute,devselection/report.

## Review/risk

InstalledTrainer has a confirmed SAME-dataset mid-epoch-resume synchronization issue when last accumulation batch ispartial; freshstart avoids this branch entirely. No thirdparty/coretrainer changes are needed for this run. Existing802checkpoint remains archived only. Stage2 changesdataset and starts fresh iterator as previously tested; verify actualupdatecounts regardless of inherited epoch display.

ReducingStage1 does not reduce totalwallbudget; it transfers about1191updates to fulltrain. Prior measured UserV2 speed~10s/update; budget4010 implies~11h train +2–3h eval. Freshstart atactualnewlaunch, expectedtotal13–15h, refined frominitialsteps.

Rollback:stop only newrun process, preserve all artifacts; old runs remain recoverable. No automatic extra baseline/ablation/seeds.

## Startup review
New config, independent review, CPU tests and Qwen GPU smoke passed. Fresh raw Qwen main started 2026-10-02 18:09 CST in goal-22-user-v2-easy2-main; actual optimizer progress >=5 verified (~10s/update). Final metrics remain pending; launcher chains independent verification after completion. Expected13–15h including evaluation (Oct3 07:10–09:10 CST). Old run artifacts and shared sources preserved.
