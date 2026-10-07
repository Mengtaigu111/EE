#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS
export TOKENIZERS_PARALLELISM=false
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/type-cot-experiment/type-cot-fast-20261006/check_kernel_equivalence.py --output tasks/type-cot-experiment/type-cot-fast-20261006/kernel-equivalence.json > tasks/type-cot-experiment/type-cot-fast-20261006/kernel-equivalence.log 2>&1
result=$?
if [ "$result" -eq 0 ]; then
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/type-cot-experiment/type-cot-fast-20261006/benchmark_liger.py --manifest tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark-manifest.json --output tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark-results > tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark-console.log 2>&1
result=$?
fi
echo "$result" > tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark.exit
exit "$result"
