#!/bin/bash
set -u
cd /home/myluo/EE/EXCEEDS
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/analysis/cot1-extra-test-20261006-185458/run_evaluation.py > /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/analysis/cot1-extra-test-20261006-185458/console.log 2>&1
status=$?
printf "%s\n" "$status" > /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/analysis/cot1-extra-test-20261006-185458/exit.code
exit "$status"
