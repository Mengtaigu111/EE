#!/usr/bin/env bash
set -u
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
QUEUE="$ROOT/queues/20260916-204747"
LOG="$ROOT/monitor-2h.log"

while true; do
  {
    date '+%F %T %Z'
    ps -eo pid,etime,stat,args | rg 'stage1-typed-mentions/scripts/(train_lora|predict|run_queue)' || true
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
    tail -3 "$ROOT/queue-0916-resume.log" 2>/dev/null || true
    find "$ROOT/runs" -name run.json -print0 | xargs -0 -r -n1 sh -c 'python3 -c "import json; d=json.load(open(\"$0\")); print(d.get(\"experiment\"), d.get(\"seed\"), d.get(\"status\"), d.get(\"stage\"))"'
    printf '\n'
  } >> "$LOG" 2>&1
  sleep 7200
done
