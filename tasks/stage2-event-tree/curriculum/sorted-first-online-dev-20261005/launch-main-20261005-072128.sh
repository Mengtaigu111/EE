#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS
export TOKENIZERS_PARALLELISM=false
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum/run_sorted_first_online_dev.py --output /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128 > /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128.console.log 2>&1
run_status=$?
printf "%s\n" "$run_status" > /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128.exit
exit "$run_status"
