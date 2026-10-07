#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS
export TOKENIZERS_PARALLELISM=false
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/type-cot-experiment/type-cot-fast-20261006/benchmark.py --manifest tasks/type-cot-experiment/type-cot-fast-20261006/benchmark-manifest.json --output tasks/type-cot-experiment/type-cot-fast-20261006/benchmark-results > tasks/type-cot-experiment/type-cot-fast-20261006/benchmark-console.log 2>&1
result=$?
echo "$result" > tasks/type-cot-experiment/type-cot-fast-20261006/benchmark.exit
exit "$result"
