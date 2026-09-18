# Stage-2 Schema-Guided Event Tree Generation

输入整篇 abstract，Qwen3-8B LoRA 一次生成该 abstract 的全部事件树（event forest），程序把文本对齐回 SciEvents token offsets，用 EXCEEDS 官方 `utils.calculate_F1` 计算 TI/TC/AI/AC/EC。方法论见 `task/0 方法论.md`。

## 树表示

```text
(ROOT (PRP (Trigger "propose") (Proposer "we") (Content "a new framework"))
      (CMP (Trigger "outperforms") (Arg1 "our method") (Dataset "SciEvents")))
```

- 事件按 trigger 首个 offset 排序（同 offset 按事件类型）；事件内 Trigger 第一，roles 按 ontology 顺序，同 role 多论元按位置。
- Nugget type 不进树。Sub-event 不建模，论元文本等于另一事件 trigger 时由对齐器优先绑定该 trigger 的 offsets，EC 自然恢复。
- 文本 = 按 offset 升序拼接的原 token；逆序 nugget（约 1%）因此无法精确回环。

## 对照实验

| 实验 | 训练 hints | 推理 hints | prompt 差异 |
| --- | --- | --- | --- |
| T0 | 无 | 无 | 只有 EventType→Roles schema |
| T1 | gold 加噪 | Stage-1 E1-42 预测 | 平铺 `"text" -> TYPE` 列表 |
| T2 | gold 加噪 | Stage-1 E1-42 预测 | hints 按 nugget type 分组 + Role[allowed types] schema |
| T1/T2 oracle | 同上 adapter | gold typed mentions | 衡量 Stage-1 质量上界，不训练 |

训练 hints 不用 Stage-1 的 train 预测（过拟合，与 dev/test 分布不符），而是对 gold hints 按 Stage-1 dev 误差画像加噪：逐类召回、BOUNDARY/TYPE_ONLY/MISSING 比例、每句约 0.92 条伪 mention。画像写在 `data/manifest.json`。

## 上界

gold 树经线性化→解析→对齐后，dev 官方指标 TI 99.2 / TC 98.6 / AI 94.8 / AC 94.8 / EC 97.2。AI/AC 的约 5 个点损失来自：官方评分按 trigger offset 建 dict，41 个同 trigger 异类型事件互相覆盖；66 个逆序 nugget 的 offset 串顺序敏感；95 个论元文本在文档内歧义或不连续。前两项 EXCEEDS 同样承受。

## 复现

```bash
cd /home/myluo/EE/EXCEEDS
export S2_PY=/home/myluo/miniconda3/envs/qwen3meld-run/bin/python
export S2=/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree
"$S2_PY" "$S2/scripts/build_dataset.py"            # 生成 data/{none,flat,typed,flat-oracle,typed-oracle}
"$S2_PY" -m unittest discover -s "$S2/tests" -v     # 14 项测试
tmux new-session -d -s stage2 "cd /home/myluo/EE/EXCEEDS && bash $S2/scripts/run_experiment.sh --experiments smoke,T0,T2,T1 --seeds 42"
```

训练：bf16 LoRA r16/alpha32，7 个投影，lr 1e-4 cosine，5 epochs，microbatch 1 × accumulation 16，max-len none 2048 / flat,typed 2816（最长样本 2646 token，峰值显存 20.9 GiB）。固定用最终 epoch 的 adapter，所有 epoch checkpoint 保留。推理 greedy，`max_new_tokens=768`（gold 树最长 638）。

队列可续跑：`run_experiment.sh --queue-dir <旧队列> --resume`。多 seed：`--seeds 42,43,44`。每个 run 目录含 `run.json`、训练/推理/评测日志、`*.metrics.json`、`*.metrics.scievents.json`（官方格式预测）、`*.metrics.docs.jsonl`（逐文档分数与原始输出）。test 额外调用官方 `eval_complex_quick_f1.py` 得到 Consecutive/Overlap/Reverse/Subevent 子集。报告：`report.md`。
