# Mean-percentile curriculum on AppendixNoCopy

Baseline: `../runs/C-AppendixNoCopy-42-20261002/run.json` (user override of original task's older C prompt). Initial model is raw `/opt/models/Qwen3-8B`; never initialize from trained baseline. Frozen prompts and completion rows are reused byte-for-byte. No Region or SemanticDecision output nodes.

Data: B1=5427, B2=5277, ROOT=2124; root_stage1=round(5427*2124/10704)=1077 using random.Random(42).sample over original ROOT order. Stage1=6504; Stage2=12828 exact frozen train bytes. Boundary ties use existing fraction0.5/all_ties_easy assignments, tau0.4275338928340865. Event-count and random-half controls are config/assignment only, not trained.

Budget: baseline actual4010 optimizer steps; stop Stage1 at2005 with max_steps4010; Stage2 restores all state and finishes4010. Shared cosine horizon4010, warmup_ratio.03, unchanged effective batch16 and LoRA recipe. Stage2 starts fresh full-data shuffle (ignore_data_skip=True); Trainer epoch is inherited bookkeeping, local effective passes logged separately. bf16 does not use GradScaler.

Safe checkpoint hooks use JSON+safetensors for model/Adam/scheduler/Python+NumPy+Torch CPU/CUDA RNG, avoiding default Trainer4.52 optimizer pickle loading unsupported on existing Torch2.4. All checkpoint files hashed; exact state restoration checked before new optimizer updates. Raw loss is checked for NaN/Inf before updates.

Checkpoints:802/1604/2005/2406/3208/4010; mandatory Stage1-end dev evaluation at2005, all remaining checkpoint dev evaluations after training as baseline. Select maximum official dev AC F1, later-step tie break; evaluate selected checkpoint on test once. Six selection candidates versus baseline five is explicitly reported.

Commands (baseline environment; GPU commands in named tmux):

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/stage2-event-tree/curriculum/build_mean_curriculum_data.py
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m pytest tasks/stage2-event-tree/curriculum -q
bash tasks/stage2-event-tree/curriculum/run_c_mean_curriculum.sh --run-dir tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-42-20261002
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/stage2-event-tree/curriculum/verify_mean_curriculum_run.py --run-dir tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-42-20261002
```

Builder is idempotent for identical derived bytes and rejects different existing output. Launcher refuses existing run directories, validates all recorded input hashes, snapshots source as evidence and checks source integrity at phase/eval transitions. Each run keeps its own report; canonical report is published once only after all training/evaluation succeeds. Tests and longest-sample GPU smoke evidence live in `../../../goal-20/`. An experiment is complete only after full-run independent verification exits0; smoke metrics are not research results.
