#!/usr/bin/env bash
set -uo pipefail
cd /home/myluo/EE/EXCEEDS
PYTHON_BIN=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
RUN_DIR=tasks/stage2-event-tree/runs/C-CL-Mean-UserV2-Easy2-42-20261002-smoke1
"$PYTHON_BIN" tasks/stage2-event-tree/curriculum/run_user_v2_easy2_curriculum.py --run-dir "$RUN_DIR" --smoke > goal-22/gpu-smoke.log 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > goal-22/gpu-smoke-training.exit
if [[ "$TASK_EXIT" -eq 0 ]]; then
  "$PYTHON_BIN" tasks/stage2-event-tree/curriculum/verify_user_v2_easy2_run.py --run-dir "$RUN_DIR" --allow-smoke > goal-22/gpu-smoke-verification.json 2>&1
  TASK_EXIT=$?
fi
printf '%s\n' "$TASK_EXIT" > goal-22/gpu-smoke.exit
exit "$TASK_EXIT"
