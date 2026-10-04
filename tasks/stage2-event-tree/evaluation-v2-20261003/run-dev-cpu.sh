#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS || exit 1
export CUDA_VISIBLE_DEVICES=''
OUT=tasks/stage2-event-tree/evaluation-v2-20261003/results
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/stage2-event-tree/evaluation-v2-20261003/run_reassessment.py --split dev > "$OUT/dev-run.log" 2>&1
STATUS=$?
printf '%s\n' "$STATUS" > "$OUT/dev-run.exit"
exit "$STATUS"
