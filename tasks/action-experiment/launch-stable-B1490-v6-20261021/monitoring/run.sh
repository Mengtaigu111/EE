#!/usr/bin/env bash
cd /home/myluo/EE/EXCEEDS || exit 1
exec /home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/action-experiment/launch-stable-B1490-v6-20261021/monitoring/scheduler.py > tasks/action-experiment/launch-stable-B1490-v6-20261021/monitoring/scheduler.log 2>&1
