# UserV2 mean curriculum

User approved exact prompt: `prompt.md` snapshot of `task/prompt/1002-event-forest-v2.md`. No new guidelines/schema edits, Region, or SemanticDecision nodes. All train/dev/test prompts use this version; unchanged original sentence rows/gold/completions and original IDs/order except prompt/mode fields.

Stage1=5427B1+1077fixedROOT=6504; Stage2=all12828. Same tau=.4275338928340865/all_ties_easy assignments, same saved ROOT subset and Stage1 IDs (no resampling). Steps2005+2005 global4010, same seed42/rawQwen recipe. Stage2 restores model/Adam/cosine/global/RNG and starts fresh shuffled fulltrain. No extra epochs, weights-only warmstart, or hard-only Stage2. ROOT full16.5575% stage1 16.5590%.

Separate code: ../build_user_v2_curriculum_data.py, ../run_user_v2_curriculum.py, ../train_user_v2_phase.py, ../verify_user_v2_curriculum_run.py. Existing safe full-state Trainer and shared loss/tokenizer/evaluator reused read-only. New data: ../../data/sentence-c-user-v2-20261002/. Old prompt, code, datasets and superseded149-step run preserved. Controls prepared only; not automatically trained.

`byte_preserving_stage1` means exact subset bytes of NEW UserV2 fulltrain; old-vs-new gold/payload invariance is separately verified.

Historical AppendixNoCopy defines4010update budget/config and metrics reference. UserV2 changes prompt plus curriculum; delta cannot be attributed solely to curriculum. No newprompt baseline retraining is authorized.

Evidence: /home/myluo/EE/EXCEEDS/goal-21/{all-tests.log,data-verification.json,token-audit.log,gpu-smoke-verification.json}. Named tmux main `goal-21-user-v2-main`; launcher goal-21/run-main.sh chains training, dev selection/test and final independent verifier. Main is complete only when main.exit=0 and main-verification.json says passed. Output ../.. /runs/C-CL-Mean-UserV2-42-20261002/. Do not confuse smoke or startup with final metrics.
