# SciEvents 原始 C 树：Curriculum Difficulty Analysis

## 数据与口径

使用原始 C 的 `checkpoint-2406` dev-best 预测；未训练模型、未重新推理。分析 1,593 个目标句，覆盖 251 篇摘要；gold/pred 分别有 2,445/2,328 个事件。无效预测 0。

官方 TI/TC/AI/AC 通过项目 scorer 对每个目标句的完整事件列表打分后 pooled 汇总；argument offsets 仍保留文档级位置。type-only 是每句事件类型多重集合，不看 trigger。事件类型混淆用 type-agnostic 一对一对齐，依次使用 trigger exact、trigger overlap、normalized trigger text、argument span/role overlap；事件类型不参与配对。

**Metric verification:** full-dev official score is compared with the stored checkpoint metrics; absolute differences: TI_f1=0, TC_f1=0, AI_f1=0, AC_f1=0.

## 总体表现

Type-only P/R/F1 = **75.90%/72.27%/74.04%**；整句 type inventory exact = **55.87%** (890/1593)。只看事件数量、忽略类型的句子准确率为 65.73%。

官方 pooled 指标：TI_f1=73.87%, TC_f1=61.63%, AI_f1=42.32%, AC_f1=39.98%

## Q1–Q3：schema difficulty 与事件数量

### Event count slices

| slice | samples | exact | type_p | type_r | type_f1 | TI_f1 | TC_f1 | AI_f1 | AC_f1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 271 | 77.86% | NA | NA | NA | NA | NA | NA | NA |
| 1 | 571 | 61.82% | 73.64% | 75.83% | 74.72% | 66.32% | 57.61% | 37.43% | 35.23% |
| 2 | 501 | 50.30% | 80.41% | 74.55% | 77.37% | 78.84% | 65.84% | 45.83% | 42.66% |
| 3+ | 250 | 29.60% | 81.64% | 67.32% | 73.79% | 77.57% | 63.01% | 43.46% | 41.96% |

### Schema recoverability（positive samples，按 sample-min train-weighted recoverability 分位分桶）

| slice | samples | exact | type_f1 | TI_f1 | TC_f1 | AI_f1 | AC_f1 |
|---|---|---|---|---|---|---|---|
| Low | 442 | 40.72% | 71.83% | 73.36% | 60.38% | 38.24% | 35.93% |
| Medium | 484 | 50.00% | 76.08% | 78.34% | 64.83% | 46.11% | 43.07% |
| High | 396 | 64.90% | 80.49% | 73.94% | 63.85% | 46.02% | 44.72% |

### Schema margin（positive samples，按 sample-min margin 分位分桶）

| slice | samples | exact | type_f1 | TI_f1 | TC_f1 | AI_f1 | AC_f1 |
|---|---|---|---|---|---|---|---|
| Low | 456 | 36.40% | 71.87% | 76.03% | 59.89% | 39.13% | 35.68% |
| Medium | 429 | 60.37% | 78.40% | 69.39% | 61.27% | 41.33% | 38.39% |
| High | 437 | 58.12% | 77.98% | 79.29% | 68.17% | 48.86% | 48.08% |

### Event count × schema margin（margin 由高到低；type inventory exact 为主）

| slice | samples | exact | type_f1 | TC_f1 | AC_f1 |
|---|---|---|---|---|---|
| 1 events / High margin | 197 | 62.44% | 75.35% | 63.40% | 43.96% |
| 1 events / Medium margin | 272 | 67.65% | 78.36% | 56.18% | 32.36% |
| 1 events / Low margin | 102 | 45.10% | 63.21% | 48.70% | 18.39% |
| 2 events / High margin | 192 | 57.29% | 81.07% | 71.47% | 50.64% |
| 2 events / Medium margin | 109 | 50.46% | 79.61% | 64.55% | 39.62% |
| 2 events / Low margin | 200 | 43.50% | 72.56% | 61.02% | 35.42% |
| 3+ events / High margin | 48 | 43.75% | 73.87% | 66.67% | 47.71% |
| 3+ events / Medium margin | 48 | 41.67% | 76.70% | 66.19% | 44.44% |
| 3+ events / Low margin | 154 | 21.43% | 72.98% | 61.14% | 39.32% |

判断：比较表中同 event-count 行和同 margin 列的变化，不预设单调关系。分桶按样本-min 分数的经验三分位，边界相同的 tie 留在同一组，故各组样本数不保证完全相等。

## Q4–Q5：事件类型错误

混淆配对表中，wrong-type 是匹配到同一个 trigger/argument 证据但类型不同；MISSED 是无可配对预测；SPURIOUS 是无可配对 gold。该分析和 type-only multiset 分数互补。

| Gold type | gold n | correct | wrong-type | missed | spurious as type |
|---|---:|---:|---:|---:|---:|
| PUR | 395 | 247 | 50 | 98 | 89 |
| ITT | 190 | 132 | 10 | 48 | 35 |
| RWS | 70 | 17 | 12 | 41 | 10 |
| RWF | 201 | 135 | 9 | 57 | 55 |
| PRP | 235 | 218 | 10 | 7 | 7 |
| WKS | 354 | 199 | 82 | 73 | 40 |
| MDS | 303 | 159 | 75 | 69 | 68 |
| FIN | 192 | 159 | 9 | 24 | 44 |
| CMP | 192 | 137 | 27 | 28 | 9 |
| FAC | 313 | 226 | 24 | 63 | 34 |

Top confusion pairs：

| Gold → Pred | Count | Rate among gold type |
|---|---:|---:|
| WKS → MDS | 52 | 14.69% |
| MDS → PUR | 36 | 11.88% |
| MDS → WKS | 32 | 10.56% |
| PUR → MDS | 27 | 6.84% |
| CMP → FAC | 25 | 13.02% |
| FAC → CMP | 14 | 4.47% |
| WKS → PRP | 11 | 3.11% |
| WKS → PUR | 11 | 3.11% |
| PRP → WKS | 7 | 2.98% |
| PUR → WKS | 7 | 1.77% |

Top Appendix schema similarity pairs：

| Type pair | Weighted Jaccard | Unweighted Jaccard | Also top-10 confusion pair? |
|---|---:|---:|---|
| RWS ↔ MDS | 0.714 | 0.833 | no |
| PRP ↔ WKS | 0.236 | 0.333 | yes |
| RWS ↔ FAC | 0.211 | 0.444 | no |
| ITT ↔ MDS | 0.200 | 0.600 | no |
| PRP ↔ FIN | 0.195 | 0.250 | no |
| ITT ↔ WKS | 0.186 | 0.600 | no |
| WKS ↔ FIN | 0.182 | 0.167 | no |
| RWF ↔ FAC | 0.156 | 0.444 | no |
| PUR ↔ ITT | 0.144 | 0.500 | no |
| ITT ↔ RWS | 0.143 | 0.500 | no |

Spearman ρ between directed off-diagonal schema similarity and gold-normalized model confusion: 0.2560597807259421 (n=90 directed pairs). This is descriptive, not causal.

## Q6：论元结构复杂度

按 gold event 的几何特征分片；这些分片允许重叠。AI/AC 是对每个结构 gold event 与其一对一匹配预测单独调用官方 scorer 后 pooled 的分数，missed gold 保留为空预测；未匹配 spurious prediction 不归入 gold 复杂度切片。

| complexity | gold_events | gold_arguments | matched_event_pairs | AI_f1 | AC_f1 |
|---|---|---|---|---|---|
| normal_contiguous | 2257 | 5074 | 1795 | 46.63% | 43.95% |
| same_span_reuse | 2 | 14 | 1 | 43.48% | 34.78% |
| nested | 166 | 452 | 128 | 39.04% | 37.26% |
| shared | 678 | 1870 | 557 | 49.08% | 47.49% |
| discontinuous_2frag_gap_le_3 | 89 | 290 | 70 | 35.76% | 35.34% |
| discontinuous_2frag_gap_4_102 | 55 | 167 | 42 | 28.26% | 26.81% |
| discontinuous_2frag_gap_gt_102 | 11 | 29 | 8 | 28.57% | 28.57% |
| discontinuous_3plus | 33 | 87 | 24 | 20.98% | 20.98% |
| reverse | 1526 | 4171 | 1246 | 48.00% | 46.15% |
| cross_sentence | 282 | 883 | 220 | 38.67% | 37.36% |

定义：nested 为同摘要内不同论元 span 的严格 token-offset 包含关系（相等 span 排除）；shared 为至少两个不同事件使用完全相同 offset span；same-span reuse 是同一事件内部重复 span；reverse 指论元整体位于 trigger 前；cross-sentence 指至少一个论元 token 不在 trigger 所属句。句法距离等不能从这些标注字段推断。

## Q7：No-event false positives

Gold no-event 样本数：271。trigger-like lexicon 仅从 train gold 的 trigger token sequence 建立，按目标句内大小写归一后的连续精确匹配计数。

| trigger_like_bin | samples | false_positive_samples | false_positive_rate | predicted_events | predicted_event_count_distribution | predicted_type_counts |
|---|---|---|---|---|---|---|
| N1:0 | 3 | 0 | 0.00% | 0 | {"0": 3} | {} |
| N2:1 | 9 | 0 | 0.00% | 0 | {"0": 9} | {} |
| N3:2+ | 259 | 60 | 23.17% | 92 | {"0": 199, "1": 37, "2": 14, "3": 9} | {"CMP": 2, "FAC": 9, "FIN": 1, "ITT": 11, "MDS": 13, "PRP": 4, "PUR": 12, "RWF": 23, "RWS": 1, "WKS": 16} |

该 lexicon 只表示训练中出现过同一 trigger surface，不代表句中真有事件；因此分层结果是触发词表命中与误报的关联诊断。

## Q8：Curriculum 结论

结果解释（与单一的逐级难度假设区分）：recoverability 从 Low→Medium→High 时，type-only F1 为 71.83→76.08→80.49%，句级 type inventory exact 为 40.72→50.00→64.90%，呈单调上升，因此它是本次数据中最清楚的 curriculum 维度。Schema margin 的 exact 为 Low 36.40%、Medium 60.37%、High 58.12%；Low 明显最难，但 Medium/High 不单调，不能宣称完整线性阶梯。

事件数主要降低整句 inventory exact：1/2/3+ 事件分别为 61.82/50.30/29.60%；但对应 type-only F1 是 74.72/77.37/73.79%，并不随事件数单调下降。固定 margin 后 exact 在各档都随事件数增加而下降；margin 在固定事件数时 Low 也始终落后，2-event 与 3+-event 的 High→Medium→Low 呈单调下降，单事件组则 Medium(67.65%) 高于 High(62.44%)，Low(45.10%) 最差。故建议把“多事件组合”和“低 margin”作为两种独立难度轴。

类型错误的例子：WKS→MDS 有 52 个、MDS→PUR 36 个、MDS→WKS 32 个、PUR→MDS 27 个、CMP→FAC 25 个。Appendix schema 中 PRP↔WKS 同时是 top-10 相似 pair 和 top-10 confusion pair；整体 schema similarity 与 confusion 的 Spearman ρ=0.256（90 个有向类型对），关联偏弱，schema 重叠只能解释部分混淆。RWS 的 70 个 gold 中 41 个漏检、12 个错分；WKS 则 82 个错分、73 个漏检，故不同类型的主要错误机制不同。

结构复杂度的结果支持把连续论元放在前、结构困难样本逐步混入：普通连续组 n=2,257，AI/AC=46.63/43.95%；跨句组 n=282 为 38.67/37.36%；嵌套组 n=166 为 39.04/37.26%；两片段且 gap 4–102 的组 n=55 为 28.26/26.81%；三片段以上 n=33 为 20.98/20.98%。但 shared 组 n=678 的 AI/AC=49.08/47.49%，反而较高；reverse n=1,526 也没有明显变差，因此不应把所有非标准结构都视作同等困难。gap>102 仅 n=11，只作提示。

No-event 结果：271 个 gold-empty 句中有 60 个误报（22.14%），全部发生在 train trigger lexicon 命中至少 2 次的组（60/259=23.17%，对应 92 个预测事件）；0/1 次命中组是 0/3 与 0/9，样本太少不能据此断言安全。课程上可先用高 recoverability、单事件样本，再增加高/中 margin 多事件组合，之后加低 margin 与跨句、嵌套、断续论元；shared/reverse 混合贯穿各阶段，并在最终阶段混入全部分布。该顺序是基于切片的设计建议，不是已验证的训练收益；要证明收益仍需做受控训练实验。

## 可复现产物

- `analyze_difficulty.py`：全流程脚本；`python3 tasks/stage2-event-tree/difficulty-analysis/analyze_difficulty.py` 重跑。
- `event_level_features.csv`、`sample_level_features.csv`：gold difficulty 特征。
- `schema_similarity.csv`、`event_type_confusion.csv`：schema 和模型 type diagnostics。
- `difficulty_slices.csv`、`argument_complexity.csv`、`no_event_breakdown.csv`：分层结果。
- `event_alignment.csv`：type-agnostic event pairs 和匹配层级。
- `verification.json`：输入计数、官方指标复核及输出文件 hash。

### Limitations

Event-type confusion depends on event alignment. Exact trigger duplicates are ambiguous in the official offset scorer (its span dictionaries retain one event per identical trigger key); duplicate-trigger sample IDs are listed in `analysis_summary.json`. Prediction argument offsets are reconstructed from output text by the existing aligner and may be ambiguous. Gold structural properties use raw gold offsets. Semantic coreference and discourse-level difficulty are not recoverable from current spans.
