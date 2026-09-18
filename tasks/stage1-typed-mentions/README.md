# Stage-1 Typed Mention Extraction

使用 `/opt/models/Qwen3-8B` 从目标句预测 10 类 argument mentions 的文本和类型。S_G 仅输入目标句；SD_G 加入完整 tokenized abstract。Stage 2 不在本轮范围。

## 复现

在项目目录运行，固定使用已有 conda 环境的 Python，避免系统 Python 与 CUDA 依赖不一致：

```bash
cd /home/myluo/EE/EXCEEDS
export STAGE1_PY=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
export STAGE1_ROOT=/home/myluo/EE/EXCEEDS/tasks/stage1-typed-mentions
"$STAGE1_PY" "$STAGE1_ROOT/scripts/build_dataset.py"
"$STAGE1_PY" -m unittest discover -s "$STAGE1_ROOT/tests" -v
```

队列按 E0 → E1 seed42 → E1-S seed42 → E1 seed43 → E1 seed44 执行；E0 仅为 dev 前 32 句的四例 sanity，不训练、不作为完整测试结果。脚本本身不创建 tmux，实验在外部 tmux 中启动：

```bash
tmux new-session -d -s stage1-typed 'cd /home/myluo/EE/EXCEEDS && bash tasks/stage1-typed-mentions/scripts/run_experiment.sh'
tmux attach-session -t stage1-typed
```

`run_experiment.sh` 默认创建新的队列目录，也可传 `--queue-dir /absolute/path/to/queue`。使用既有队列时先检查其状态、日志和失败阶段，避免同时启动两个 GPU 队列。运行目录与配置是具体调用参数、长度调整和退出状态的依据。

独立训练命令如下；正式训练不提供 `--limit` 或 `--max-steps`，使用固定最终第 3 epoch 的 adapter，不按 test 或 dev 选 checkpoint：

```bash
"$STAGE1_PY" "$STAGE1_ROOT/scripts/train_lora.py" \
  --data "$STAGE1_ROOT/data/SD_G/train.jsonl" \
  --output "$STAGE1_ROOT/runs/manual-E1-42/train" --seed 42 --max-len 1536
"$STAGE1_PY" "$STAGE1_ROOT/scripts/predict.py" \
  --data "$STAGE1_ROOT/data/SD_G/test.jsonl" \
  --adapter "$STAGE1_ROOT/runs/manual-E1-42/train/adapter" \
  --output "$STAGE1_ROOT/runs/manual-E1-42/test.predictions.jsonl" --seed 42 --batch-size 1
"$STAGE1_PY" "$STAGE1_ROOT/scripts/evaluate.py" \
  --data "$STAGE1_ROOT/data/SD_G/test.jsonl" \
  --predictions "$STAGE1_ROOT/runs/manual-E1-42/test.predictions.jsonl" \
  --output "$STAGE1_ROOT/runs/manual-E1-42/test.metrics.json"
```

手工运行的目录若要加入自动报告，需写入真实 `run.json` 元数据（`experiment`、`seed`、`status`），只有全部预期阶段成功后才能标 `completed`。队列自动维护此元数据。

```bash
"$STAGE1_PY" "$STAGE1_ROOT/scripts/write_report.py" --root "$STAGE1_ROOT"
```

生成的 [report.md](report.md) 读取 `runs/*/run.json`、`dev.metrics.json` 和 `test.metrics.json`；没有结果的实验显示待运行。E1 三 seed test 汇总仅使用完整、已完成的 seeds 42/43/44，报告均值与样本标准差。若同一 seed 有多次完成的运行，使用按路径排序最后的一次，同时保留全部运行记录。E1-S 只有 seed42，不能据此声称统计显著。

## 固定训练与推理配置

bf16、非量化；LoRA r16 / alpha32 / dropout0.05，作用于 q/k/v/o 和 gate/up/down projections。学习率 1e-4，cosine，warmup 0.03，3 epochs，microbatch1 × accumulation16，gradient checkpointing；loss 只覆盖 assistant tokens，`enable_thinking=False`。S 长度 1024，S+D 长度 1536。长度审计禁止静默截断 gold；若实际长度需要调整，必须在运行配置及日志中保留依据。OOM 时先保证 microbatch1，再评估长度调整。

推理 greedy、`max_new_tokens=400`、默认 batch1。先 strict JSON，再扫描首个完整 JSON 对象；仍失败则 invalid，预测为空。原始模型输出、解析结果、完成标志与文件 hash 保存在运行目录。

## Gold 与评测口径

类型取自 `data/SciEvents/{split}.json` 中 `events[].arguments[].nugget_type`。`tasks/llm-sft/normalized/*.documents.jsonl` 用于验证句边界；原有 `*.gold.jsonl` 没有 nugget type，不能直接用于 typed 监督。

- 只保留 OG / APP / MOD / FEA / TAK / DST / LIM / STR / WEA / DEG。排除纯 trigger 和 E-*，保留 trigger 兼任普通类型的标注。
- 用 `(offset 集合, type)` 去重，保留首次 annotation order；同位置不同类型仍是不同 typed gold。输出按 start、end、annotation order 排序。
- 跨句 mentions 整条排除，不能投影或截断为新标签。目标句包含的所有 gold 根据其自身位置归属，不根据所属 event 的 trigger 位置归属。保留空句。
- offsets 为文档 token positions；sentence start inclusive、end exclusive。乱序或重复 offset canonicalize 为排序去重集合，mention text 由对应原文 tokens 空格拼接；句内不连续 mentions 保留。原始差异与跨句排除记录在 `data/audit/`。
- `data/manifest.json` 保存来源和输出 hash、类别计数与固定四例 IDs。四例仅选自 train，贪心增加类型覆盖，输入顺序打破平局；正式 LoRA 不使用 few-shot。

预测按连续 exact token match → 大小写不敏感 exact → subsequence 对齐，同文重复匹配从左到右分配。评测是 offset 集合 exact match，typed 额外匹配类型；unaligned 计 FP，invalid 预测为空。Untyped 对重复类型的同位置 span 去重。FN 优先 TYPE_ONLY、其次 BOUNDARY、最后 MISSING；FP 分 BOUNDARY 和 OTHER。invalid rate 分母是句数，unaligned rate 分母是解析 mentions 数，平均预测数也包含无法对齐的已解析 mentions。

Gold-text roundtrip 可独立运行：

```bash
"$STAGE1_PY" "$STAGE1_ROOT/scripts/evaluate.py" \
  --data "$STAGE1_ROOT/data/S_G/test.jsonl" --oracle \
  --output "$STAGE1_ROOT/verification/oracle-test.json"
```

该结果只是文本到位置接口的 roundtrip 诊断，**不能视为模型性能上界**。重复同文、不连续 mention 和连续同文干扰会损失位置信息；模型也可能以不同输出顺序改变映射。评测器不覆盖既有输出文件，重跑时使用新路径并保留旧证据。
