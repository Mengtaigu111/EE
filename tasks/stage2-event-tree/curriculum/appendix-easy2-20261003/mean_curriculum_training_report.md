# AppendixNoCopy Easy2 mean curriculum training report

沿用历史 AppendixNoCopy prompt、全部原样本与 gold；设置及预算参考 C-AppendixNoCopy-42-20261002。使用修复后的评估链路。
从原始 Qwen3-8B 初始化；无 Region / SemanticDecision 节点。单 seed，不提前假设 curriculum 有收益。

## Q1. Data construction

- B1_count: 5427
- B2_count: 5277
- root_total: 2124
- root_stage1_count: 1077
- stage1_total_count: 6504
- stage2_total_count: 12828

Stage2 JSONL 与原 AppendixNoCopy full train 字节一致；全部 gold、样本和课程 IDs 与原数据一致，ROOT subset 原样复用。
ROOT比例：full_train_root_ratio=16.557530%, stage1_root_ratio=16.559041%, stage2_root_ratio=16.557530%

## Q2. Training budget

Baseline=4010; Stage1=814; Stage2=3196; total=4010，严格相等。
Cosine horizon=4010，warmup_ratio=0.03，仅从 global step0 开始一次。

## Q3. Resume correctness

恢复证据：`stage2/resume_evidence.json`；optimizer/scheduler/RNG 采用 safetensors + JSON。
bf16 无 GradScaler（N/A）；不升级共享环境，不跳过新 full-train 数据迭代器的旧阶段 batch。
```json
{
  "checkpoint": "/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/stage1/checkpoints/checkpoint-814",
  "expected_step": 814,
  "model_exact": true,
  "optimizer_exact": true,
  "scheduler_exact": true,
  "scheduler_last_epoch": 814,
  "learning_rates": [
    9.236764662851797e-05,
    9.236764662851797e-05
  ],
  "scaler": "not used (bf16 / CPU)",
  "global_step": 814,
  "horizon": 4010,
  "data_skip": false,
  "rng_exact": true
}
```

## Q4. Dev results

| Run / Stage | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| AppendixNoCopy baseline (dev best) | 74.7175 | 62.3322 | 43.5531 | 41.2637 | 49.4118 |
| Curriculum Stage 1 end | 67.3344 | 53.9654 | 37.3037 | 35.2688 | 36.6322 |
| Curriculum final | 73.2303 | 60.6517 | 40.9340 | 38.4255 | 47.6481 |
| Curriculum best checkpoint | 74.6184 | 61.7707 | 43.1069 | 40.5298 | 48.6085 |

全部为 dev 百分比；EC 使用官方 EC_C_f1。baseline 与课程模型均采用修复后的同一官方评估口径。
Best: /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/stage2/checkpoints/checkpoint-2406, step=2406, selection=official dev AC_f1（平局较大step）。
对照5个checkpoint；curriculum 6个（额外包含强制Stage1边界）。

## Q5. Delta vs AppendixNoCopy

ΔTI=-0.0991 pp, ΔTC=-0.5615 pp, ΔAI=-0.4462 pp, ΔAC=-0.7339 pp, ΔEC=-0.8032 pp

两组 prompt 与训练预算相同，仅训练课程安排不同；单 seed 探索性对照，不作为统计显著性结论。官方评分器共享 trigger 覆盖限制仍保留。

## Q6. Training behavior

Stage1 loss=0.262099; Stage2 loss=0.081483（按本阶段updates校正Trainer汇总）。
边界最近loss: Stage1 last=0.0922; Stage2 first=0.1424。这是不同batch，单点跳变不作稳定性判断。
阶段collator读取: S1=13008 examples/2.0000 passes; S2=51125 examples/3.9854 passes（可能含一个prefetch batch）。
最佳checkpoint位于 Stage2。
主训练两阶段exit0；每个microbatch原始loss均检查finite，NaN/Inf会在optimizer更新前停止。未发生OOM；各阶段日志保留全部warnings。

## Provenance

- AppendixNoCopy baseline (dev best): `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/results/baseline/dev-checkpoint-2406.metrics.json`; SHA256 `8b128c69ad8d9bb133a630af0c10f06777c6225d4b3f14a8b4f84f5f23e8f4e9`
- Curriculum Stage 1 end: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/checkpoint-eval/dev-checkpoint-814.metrics.json`; SHA256 `8ee1f91f857d1e120dffbf873ae4ee61010581f2ccdf58f80fb3a35cc8cd72ba`
- Curriculum final: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/checkpoint-eval/dev-checkpoint-4010.metrics.json`; SHA256 `82bcf8c20f3a14524f067762d2b9999d08a61fed4cd69de1e4dcc2af9f8627a4`
- Curriculum best checkpoint: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/checkpoint-eval/dev-checkpoint-2406.metrics.json`; SHA256 `f85a457f97a6a6d6111b6dd5b61433a6b23881b6e94cb6443811f0db2e7623d8`
- Manifest: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum/appendix-easy2-20261003/mean_curriculum_manifest.json`
- Prompt: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum/appendix-easy2-20261003/prompt.md`; SHA256 `2bcc66773a71904973ae4f85b865e60eddfacf4230515618576d3217e6776b41`
- Run: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/run.json`
- Baseline run SHA256: `8bb96dea33168364c253932f871d022624011605c8d1033ea2e2b22a4aa6c1c5`

## Test (dev-best only)

| TI | TC | AI | AC | EC |
|---:|---:|---:|---:|---:|
| 74.6179 | 62.6181 | 43.4352 | 41.1416 | 47.9268 |

同 prompt baseline（修复后）：TI_f1=75.4139%, TC_f1=63.5479%, AI_f1=44.2034%, AC_f1=41.8797%, EC_C_f1=48.0687%
Test差值：ΔTI=-0.7960 pp, ΔTC=-0.9298 pp, ΔAI=-0.7681 pp, ΔAC=-0.7380 pp, ΔEC=-0.1418 pp
Test仅评估dev选出的一个checkpoint。
