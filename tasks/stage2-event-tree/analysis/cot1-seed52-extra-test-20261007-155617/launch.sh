#!/usr/bin/env bash
set -u
cd /home/myluo/EE/EXCEEDS
export TOKENIZERS_PARALLELISM=false
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u '/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/analysis/cot1-seed52-extra-test-20261007-155617/run_evaluation.py' > '/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/analysis/cot1-seed52-extra-test-20261007-155617/console.log' 2>&1
result=$?
echo "$result" > '/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/analysis/cot1-seed52-extra-test-20261007-155617/exit.code'
exit "$result"
