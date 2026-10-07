#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS
export TOKENIZERS_PARALLELISM=false
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/type-cot-experiment/type-cot-fast-20261006/check_kernel_equivalence.py --output tasks/type-cot-experiment/type-cot-fast-20261006/kernel-equivalence-v2.json > tasks/type-cot-experiment/type-cot-fast-20261006/kernel-equivalence-v2.log 2>&1
result=$?
if [ "$result" -eq 0 ]; then
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/type-cot-experiment/type-cot-fast-20261006/benchmark_liger.py --manifest tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark-manifest.json --output tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark-results > tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark-v2-console.log 2>&1
result=$?
fi
echo "$result" > tasks/type-cot-experiment/type-cot-fast-20261006/liger-benchmark-v2.exit
exit "$result"
