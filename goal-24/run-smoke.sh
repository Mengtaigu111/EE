#!/usr/bin/env bash
set -uo pipefail
cd /home/myluo/EE/EXCEEDS
TASK_PYTHON=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
TASK_RUN=/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003-Smoke
"$TASK_PYTHON" tasks/stage2-event-tree/curriculum/run_sorted_first.py --run-dir "$TASK_RUN" --manifest tasks/stage2-event-tree/curriculum/sorted-first-20261003/manifest.json --smoke > goal-24/smoke-console.log 2>&1
TASK_EXIT=$?
printf '%s\n' "$TASK_EXIT" > goal-24/smoke-pipeline.exit
if [[ "$TASK_EXIT" -eq 0 ]]; then
  "$TASK_PYTHON" tasks/stage2-event-tree/curriculum/verify_sorted_first_run.py --run-dir "$TASK_RUN" > goal-24/smoke-verification.json 2>&1
  TASK_EXIT=$?
fi
printf '%s\n' "$TASK_EXIT" > goal-24/smoke.exit
exit "$TASK_EXIT"
