#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS
export TOKENIZERS_PARALLELISM=false
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u '/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/type-cot-seed52-20261006/run.py' --manifest '/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/type-cot-seed52-20261006/manifest.json' --output '/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-52-20261006-235604-smoke' --smoke > '/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-52-20261006-235604-smoke.console.log' 2>&1
result=$?
echo "$result" > '/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-52-20261006-235604-smoke.exit'
exit "$result"
