#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS || exit 1
export CUDA_VISIBLE_DEVICES=''
OUT=tasks/stage2-event-tree/evaluation-v2-20261003/results
PY=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
if [[ ! -f goal-22/main.exit ]] || [[ "$(cat goal-22/main.exit)" != 0 ]]; then
  printf 'Old pipeline has not finished successfully.\n' > "$OUT/test-run.log"
  printf '2\n' > "$OUT/test-run.exit"
  exit 2
fi
"$PY" tasks/stage2-event-tree/evaluation-v2-20261003/run_reassessment.py --split test > "$OUT/test-run.log" 2>&1
STATUS=$?
if [[ "$STATUS" -eq 0 ]]; then
  "$PY" tasks/stage2-event-tree/evaluation-v2-20261003/verify_results.py > "$OUT/verification.log" 2>&1
  STATUS=$?
fi
printf '%s\n' "$STATUS" > "$OUT/test-run.exit"
if [[ "$STATUS" -eq 0 ]]; then
  "$PY" tasks/stage2-event-tree/evaluation-v2-20261003/write_report.py > "$OUT/report-run.log" 2>&1
  STATUS=$?
fi
exit "$STATUS"
