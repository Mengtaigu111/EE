#!/usr/bin/env bash
set -euo pipefail
STAGE2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE2_PYTHON="${STAGE2_PYTHON:-/home/myluo/miniconda3/envs/qwen3meld-run/bin/python}"
if [[ -z "${TMUX:-}" ]]; then
  echo 'Run this GPU queue inside tmux.' >&2
  exit 2
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
exec "$STAGE2_PYTHON" "$STAGE2_ROOT/scripts/run_queue.py" "$@"
