# 实验与结果

共同设置：原始 `/opt/models/Qwen3-8B`，LoRA r16/alpha32/dropout0.05；bf16，batch1×累积16，AdamW 1e-4、cosine、warmup0.03、seed42、max_len4096、总4010步。不是从旧adapter续训。

| 简称 | Prompt | 训练安排 |
|---|---|---|
| 基线 | AppendixNoCopy | 全量随机约5轮 |
| UserV2-Easy2 | 用户新版树prompt | B1+部分ROOT 2轮，再全量约4轮 |
| Easy2 | 原AppendixNoCopy | 同上；与基线同prompt对照 |
| Sorted1 | 原AppendixNoCopy | 首轮全量难度升序，后4轮全量随机 |

四组最终选择均为 checkpoint-2406，按完整 dev offset AC F1，平局取较大step；test不选模。UserV2改了prompt，不能归因成纯课程效果。

## 修复后官方 offset（%）

| 实验 | dev AC | Test TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|---:|
| 基线 | 41.26 | 75.41 | 63.55 | 44.20 | 41.88 | 48.07 |
| UserV2-Easy2 | 40.72 | 74.16 | 62.68 | 43.11 | 40.73 | 48.44 |
| Easy2 | 40.53 | 74.62 | 62.62 | 43.44 | 41.14 | 47.93 |
| Sorted1 | 40.55 | 75.19 | 63.10 | 44.16 | 41.98 | 48.19 |

## 官方 raw token（%）

| 实验 | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| 基线 | 76.10 | 63.32 | 43.92 | 41.61 | 46.83 |
| Easy2 | 75.39 | 62.44 | 43.10 | 40.80 | 46.51 |
| Sorted1 | 75.91 | 62.84 | 43.68 | 41.52 | 46.88 |

Sorted1 相对基线：offset AC +0.10pp，token AC −0.09pp；相对Easy2：offset AC +0.84pp。当前基本追平基线，单seed没有稳定增益证据。UserV2未补算raw token。

## 原始证据

- [C-AppendixNoCopy-42-20261002](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-AppendixNoCopy-42-20261002/run.json)
- [C-CL-Mean-UserV2-Easy2-42-20261002](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-UserV2-Easy2-42-20261002/run.json)
- [C-CL-Mean-AppendixNoCopy-Easy2-42-20261003](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/run.json)
- [C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003/run.json)
- [修复后基线指标](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/results/baseline/test-checkpoint-2406.metrics.json)
- [UserV2修复后指标](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/results/curriculum/test-checkpoint-2406.metrics.json)
- [基线/Easy2 token完整指标](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/token-evaluation-20261003/results.json)
- [Sorted1最终独立校验](/home/myluo/EE/EXCEEDS/goal-24/main-verification.json)
