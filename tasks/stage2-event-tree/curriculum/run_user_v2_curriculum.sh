#!/usr/bin/env bash
set -euo pipefail
CURRICULUM_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
if [[ -z "${TMUX:-}" ]]; then
  echo 'Run UserV2 curriculum in named tmux.' >&2
  exit 1
fi
exec "$PYTHON_BIN" "$CURRICULUM_DIR/run_user_v2_curriculum.py" "$@"
