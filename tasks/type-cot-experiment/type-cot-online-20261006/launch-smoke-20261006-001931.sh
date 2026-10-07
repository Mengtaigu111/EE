#!/bin/bash
set -u
cd /home/myluo/EE/EXCEEDS
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u /home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/type-cot-online-20261006/run.py --manifest /home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/type-cot-online-20261006/manifest.json --output /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-42-20261006-001931-smoke --smoke > /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-42-20261006-001931-smoke.console.log 2>&1
status=$?
printf "%s\n" "$status" > /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-42-20261006-001931-smoke.exit
exit "$status"
