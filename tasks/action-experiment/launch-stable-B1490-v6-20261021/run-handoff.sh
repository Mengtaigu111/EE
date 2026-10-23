#!/usr/bin/env bash
cd /home/myluo/EE/EXCEEDS || exit 1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/action-experiment/launch-stable-B1490-v6-20261021/after_acceptance.py --verifier-pid "$1" --verifier-start "$2" > tasks/action-experiment/launch-stable-B1490-v6-20261021/handoff.log 2>&1
