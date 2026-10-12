#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/myluo/EE/EXCEEDS
PYTHON=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
LAUNCH=/home/myluo/EE/EXCEEDS/tasks/action-experiment/launch-v2-20261012
SFT=/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012
RL=/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-v2-42-20261012
stage=starting
finish() {
  code=$?
  if [[ "$code" == 0 && "$stage" != completed ]]; then code=1; fi
  trap - EXIT
  printf '%s stage=%s exit=%s\n' "$(date -Is)" "$stage" "$code" >> "$LAUNCH/status.log"
  printf '%s\n' "$code" > "$LAUNCH/exit.code"
  exit "$code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export TOKENIZERS_PARALLELISM=false
printf '%s pid=%s session=%s\n' "$(date -Is)" "$$" "$TMUX" > "$LAUNCH/status.log"
printf '%s\n' "$$" > "$LAUNCH/pid"
stage=sft
printf '%s stage=%s\n' "$(date -Is)" "$stage" >> "$LAUNCH/status.log"
"$PYTHON" -u tasks/action-experiment/train_sft.py --output "$SFT" --max-new-tokens 768 --max-len 4096 --dev-batch-size 4 > "$LAUNCH/sft.log" 2>&1
stage=grpo-preflight
printf '%s stage=%s\n' "$(date -Is)" "$stage" >> "$LAUNCH/status.log"
"$PYTHON" -u tasks/action-experiment/preflight.py --stage grpo --init-adapter "$SFT/best-adapter" --output "$LAUNCH/grpo-preflight.json" > "$LAUNCH/grpo-preflight.log" 2>&1
stage=grpo
printf '%s stage=%s\n' "$(date -Is)" "$stage" >> "$LAUNCH/status.log"
"$PYTHON" -u tasks/action-experiment/train_grpo.py --init-adapter "$SFT/best-adapter" --output "$RL" > "$LAUNCH/grpo.log" 2>&1
stage=completed
