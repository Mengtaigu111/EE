# Stage-1 Typed Mention Extraction 实验报告

自动汇总时间：2026-09-17T02:13:20.869863+00:00。所有数值来自现存 JSON 结果；缺失或未写完的文件标为待运行。

模型：`/opt/models/Qwen3-8B`。E1 为 S+D LoRA 主实验；E1-S 为 S LoRA 对照；E0 为不训练的 S+D 四例格式检查。

## 数据与口径

| split | 文档 | 句子 | 句内 typed mentions | 空句 | 跨句排除 | 句内不连续 | 重复标注去除 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2007 | 12828 | 36065 | 2121 | 691 | 1204 | 3217 |
| dev | 251 | 1593 | 4520 | 265 | 66 | 117 | 410 |
| test | 250 | 1627 | 4536 | 268 | 102 | 115 | 455 |

来源、输出 SHA256 与规则：[data/manifest.json](data/manifest.json)。四例仅来自 train，覆盖类型：APP, DEG, DST, FEA, LIM, MOD, OG, STR, TAK, WEA。

Gold 类型仅取原始 `events[].arguments[].nugget_type` 的 10 类；按 offset 集合和类型去重，同 span 多类型保留。纯 trigger 和 E-* 排除，trigger 兼任普通类型保留。跨句整条排除，空句保留；不连续和乱序标注按原文 token 顺序输出，审计见 `data/audit/`。

评测为文档 token offset 集合 exact match；typed 额外要求类型相同。unaligned 计 FP，invalid 预测为空。invalid rate 分母是句数，unaligned rate 分母是解析出的 mentions 数量。P/R/F1、recall 和 rate 均以百分数展示。

## E0 / E1 / E1-S 主结果

E0 仅评估 dev 前 32 句的 sanity 子集，不能与完整 dev/test 分数直接比较。smoke 仅验证执行链。未完成的 run 即使已有部分指标，也不会进入正式结论或三 seed 汇总。

| 实验 | seed | split | 状态 / 范围 | 句数 | Untyped P | R | F1 | Typed P | R | F1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 42 | dev | completed / subset | 32 | 22.67 | 41.46 | 29.31 | 13.25 | 24.10 | 17.09 |
| E1 | 42 | dev | completed / full | 1593 | 57.17 | 60.99 | 59.02 | 48.50 | 51.53 | 49.97 |
| E1 | 42 | test | completed / full | 1627 | 57.81 | 62.67 | 60.14 | 49.83 | 53.90 | 51.78 |
| E1 | 43 | dev | completed / full | 1593 | 57.49 | 61.50 | 59.43 | 48.84 | 52.01 | 50.37 |
| E1 | 43 | test | completed / full | 1627 | 57.60 | 62.70 | 60.04 | 49.39 | 53.64 | 51.43 |
| E1 | 44 | dev | completed / full | 1593 | 56.76 | 61.57 | 59.07 | 48.37 | 52.30 | 50.26 |
| E1 | 44 | test | completed / full | 1627 | 58.46 | 63.80 | 61.02 | 50.37 | 54.85 | 52.52 |
| E1-S | 42 | dev | completed / full | 1593 | 57.05 | 61.19 | 59.05 | 48.40 | 51.70 | 49.99 |
| E1-S | 42 | test | completed / full | 1627 | 56.80 | 62.45 | 59.49 | 48.82 | 53.57 | 51.09 |
| smoke | 42 | dev | completed / subset | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| smoke | 42 | test | completed / 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

## 类型召回与格式诊断

### E0-42-20260916-204747 / dev

[runs/E0-42-20260916-204747/dev.metrics.json](runs/E0-42-20260916-204747/dev.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50.00 | 52.94 | 9.09 | 7.14 | 22.22 | 66.67 | 0.00 | 0.00 | 33.33 | 0.00 |

invalid rate：0.00%；unaligned rate：1.32%；平均预测 mentions/句：4.72。

错误分类：FN BOUNDARY=27, TYPE_ONLY=14, MISSING=22；FP BOUNDARY=36, OTHER=95。

逐句错误：[runs/E0-42-20260916-204747/dev.metrics.errors.jsonl](runs/E0-42-20260916-204747/dev.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `206628c44ef09a294a61fc2bd0568d59:0`：current state - of - the - art neural dialogue models learn from human conversations following the data - driven paradigm .

  错误：[{"side": "FP", "category": "OTHER", "offsets": [8, 9, 10], "type": "APP"}, {"side": "FP", "category": "OTHER", "offsets": [13, 14], "type": "DST"}, {"side": "FP", "category": "OTHER", "offsets": [17, 18, 19, 20], "type": "APP"}]。

- `206628c44ef09a294a61fc2bd0568d59:1`：as such , a reliable training corpus is the crux of building a robust and well - behaved dialogue model .

  错误：[{"side": "FN", "category": "BOUNDARY", "offsets": [26, 27, 28], "type": "DST"}, {"side": "FP", "category": "BOUNDARY", "offsets": [27, 28], "type": "DST"}, {"side": "FP", "category": "OTHER", "offsets": [40, 41], "type": "APP"}]。

### E1-42-20260916-204747 / dev

[runs/E1-42-20260916-204747/dev.metrics.json](runs/E1-42-20260916-204747/dev.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94.16 | 57.63 | 43.32 | 43.16 | 38.35 | 56.00 | 41.30 | 54.70 | 43.19 | 45.87 |

invalid rate：0.00%；unaligned rate：0.06%；平均预测 mentions/句：3.01。

错误分类：FN BOUNDARY=973, TYPE_ONLY=429, MISSING=789；FP BOUNDARY=1004, OTHER=1469。

逐句错误：[runs/E1-42-20260916-204747/dev.metrics.errors.jsonl](runs/E1-42-20260916-204747/dev.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `206628c44ef09a294a61fc2bd0568d59:0`：current state - of - the - art neural dialogue models learn from human conversations following the data - driven paradigm .

  错误：[{"side": "FP", "category": "OTHER", "offsets": [8, 9, 10], "type": "APP"}]。

- `206628c44ef09a294a61fc2bd0568d59:1`：as such , a reliable training corpus is the crux of building a robust and well - behaved dialogue model .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [26, 27, 28], "type": "DST"}]。

### E1-42-20260916-204747 / test

[runs/E1-42-20260916-204747/test.metrics.json](runs/E1-42-20260916-204747/test.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 92.63 | 60.02 | 35.75 | 46.35 | 43.36 | 67.14 | 43.42 | 53.65 | 46.64 | 42.42 |

invalid rate：0.00%；unaligned rate：0.12%；平均预测 mentions/句：3.02。

错误分类：FN BOUNDARY=944, TYPE_ONLY=396, MISSING=751；FP BOUNDARY=948, OTHER=1514。

逐句错误：[runs/E1-42-20260916-204747/test.metrics.errors.jsonl](runs/E1-42-20260916-204747/test.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `180fbfb92cce987edc29809639e29c2e:0`：we present fiesta , a model selection approach that significantly reduces the computational resources required to reliably identify state - of - the - art performance from large collections of candidate models .

  错误：[{"side": "FN", "category": "BOUNDARY", "offsets": [12, 13, 14], "type": "FEA"}, {"side": "FP", "category": "BOUNDARY", "offsets": [12, 13], "type": "FEA"}]。

- `180fbfb92cce987edc29809639e29c2e:1`：despite being known to produce unreliable comparisons , it is still common practice to compare model evaluations based on single choices of random seeds .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [38, 39], "type": "WEA"}, {"side": "FN", "category": "MISSING", "offsets": [48, 49], "type": "FEA"}, {"side": "FN", "category": "MISSING", "offsets": [52, 53, 54, 55, 56], "type": "APP"}]。

### E1-43-20260916-204747 / dev

[runs/E1-43-20260916-204747/dev.metrics.json](runs/E1-43-20260916-204747/dev.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 95.77 | 59.15 | 41.94 | 44.87 | 38.35 | 55.00 | 39.47 | 56.84 | 46.01 | 37.61 |

invalid rate：0.00%；unaligned rate：0.12%；平均预测 mentions/句：3.02。

错误分类：FN BOUNDARY=958, TYPE_ONLY=432, MISSING=779；FP BOUNDARY=976, OTHER=1487。

逐句错误：[runs/E1-43-20260916-204747/dev.metrics.errors.jsonl](runs/E1-43-20260916-204747/dev.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `206628c44ef09a294a61fc2bd0568d59:0`：current state - of - the - art neural dialogue models learn from human conversations following the data - driven paradigm .

  错误：[{"side": "FP", "category": "OTHER", "offsets": [8, 9, 10], "type": "APP"}]。

- `206628c44ef09a294a61fc2bd0568d59:1`：as such , a reliable training corpus is the crux of building a robust and well - behaved dialogue model .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [26, 27, 28], "type": "DST"}]。

### E1-43-20260916-204747 / test

[runs/E1-43-20260916-204747/test.metrics.json](runs/E1-43-20260916-204747/test.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 94.02 | 59.44 | 35.23 | 44.43 | 41.60 | 68.57 | 46.97 | 56.65 | 42.86 | 42.42 |

invalid rate：0.00%；unaligned rate：0.12%；平均预测 mentions/句：3.03。

错误分类：FN BOUNDARY=953, TYPE_ONLY=409, MISSING=741；FP BOUNDARY=952, OTHER=1541。

逐句错误：[runs/E1-43-20260916-204747/test.metrics.errors.jsonl](runs/E1-43-20260916-204747/test.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `180fbfb92cce987edc29809639e29c2e:0`：we present fiesta , a model selection approach that significantly reduces the computational resources required to reliably identify state - of - the - art performance from large collections of candidate models .

  错误：[{"side": "FN", "category": "BOUNDARY", "offsets": [12, 13, 14], "type": "FEA"}, {"side": "FP", "category": "OTHER", "offsets": [5, 6, 7], "type": "APP"}, {"side": "FP", "category": "BOUNDARY", "offsets": [12, 13], "type": "FEA"}]。

- `180fbfb92cce987edc29809639e29c2e:1`：despite being known to produce unreliable comparisons , it is still common practice to compare model evaluations based on single choices of random seeds .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [38, 39], "type": "WEA"}, {"side": "FN", "category": "MISSING", "offsets": [48, 49], "type": "FEA"}, {"side": "FN", "category": "MISSING", "offsets": [52, 53, 54, 55, 56], "type": "APP"}]。

### E1-44-20260916-204747 / dev

[runs/E1-44-20260916-204747/dev.metrics.json](runs/E1-44-20260916-204747/dev.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 95.77 | 59.62 | 43.32 | 43.95 | 38.35 | 58.50 | 40.08 | 54.27 | 44.60 | 48.62 |

invalid rate：0.00%；unaligned rate：0.14%；平均预测 mentions/句：3.07。

错误分类：FN BOUNDARY=988, TYPE_ONLY=419, MISSING=749；FP BOUNDARY=1015, OTHER=1508。

逐句错误：[runs/E1-44-20260916-204747/dev.metrics.errors.jsonl](runs/E1-44-20260916-204747/dev.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `206628c44ef09a294a61fc2bd0568d59:0`：current state - of - the - art neural dialogue models learn from human conversations following the data - driven paradigm .

  错误：[{"side": "FP", "category": "OTHER", "offsets": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "type": "APP"}]。

- `206628c44ef09a294a61fc2bd0568d59:1`：as such , a reliable training corpus is the crux of building a robust and well - behaved dialogue model .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [26, 27, 28], "type": "DST"}]。

### E1-44-20260916-204747 / test

[runs/E1-44-20260916-204747/test.metrics.json](runs/E1-44-20260916-204747/test.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 92.23 | 61.42 | 38.86 | 47.46 | 43.46 | 74.76 | 43.84 | 57.51 | 43.28 | 41.67 |

invalid rate：0.00%；unaligned rate：0.10%；平均预测 mentions/句：3.04。

错误分类：FN BOUNDARY=905, TYPE_ONLY=405, MISSING=738；FP BOUNDARY=897, OTHER=1554。

逐句错误：[runs/E1-44-20260916-204747/test.metrics.errors.jsonl](runs/E1-44-20260916-204747/test.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `180fbfb92cce987edc29809639e29c2e:0`：we present fiesta , a model selection approach that significantly reduces the computational resources required to reliably identify state - of - the - art performance from large collections of candidate models .

  错误：[{"side": "FN", "category": "BOUNDARY", "offsets": [12, 13, 14], "type": "FEA"}, {"side": "FP", "category": "BOUNDARY", "offsets": [12, 13], "type": "FEA"}]。

- `180fbfb92cce987edc29809639e29c2e:1`：despite being known to produce unreliable comparisons , it is still common practice to compare model evaluations based on single choices of random seeds .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [38, 39], "type": "WEA"}, {"side": "FN", "category": "MISSING", "offsets": [48, 49], "type": "FEA"}, {"side": "FN", "category": "MISSING", "offsets": [52, 53, 54, 55, 56], "type": "APP"}]。

### E1-S-42-20260916-204747 / dev

[runs/E1-S-42-20260916-204747/dev.metrics.json](runs/E1-S-42-20260916-204747/dev.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 95.37 | 59.04 | 42.86 | 43.68 | 37.08 | 55.50 | 40.89 | 55.13 | 43.19 | 46.79 |

invalid rate：0.00%；unaligned rate：0.19%；平均预测 mentions/句：3.03。

错误分类：FN BOUNDARY=956, TYPE_ONLY=429, MISSING=798；FP BOUNDARY=977, OTHER=1515。

逐句错误：[runs/E1-S-42-20260916-204747/dev.metrics.errors.jsonl](runs/E1-S-42-20260916-204747/dev.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `206628c44ef09a294a61fc2bd0568d59:0`：current state - of - the - art neural dialogue models learn from human conversations following the data - driven paradigm .

  错误：[{"side": "FP", "category": "OTHER", "offsets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "type": "APP"}]。

- `206628c44ef09a294a61fc2bd0568d59:1`：as such , a reliable training corpus is the crux of building a robust and well - behaved dialogue model .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [26, 27, 28], "type": "DST"}]。

### E1-S-42-20260916-204747 / test

[runs/E1-S-42-20260916-204747/test.metrics.json](runs/E1-S-42-20260916-204747/test.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 92.83 | 59.32 | 37.31 | 44.57 | 42.43 | 64.76 | 42.38 | 60.52 | 46.64 | 44.70 |

invalid rate：0.00%；unaligned rate：0.22%；平均预测 mentions/句：3.06。

错误分类：FN BOUNDARY=945, TYPE_ONLY=401, MISSING=760；FP BOUNDARY=955, OTHER=1592。

逐句错误：[runs/E1-S-42-20260916-204747/test.metrics.errors.jsonl](runs/E1-S-42-20260916-204747/test.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `180fbfb92cce987edc29809639e29c2e:0`：we present fiesta , a model selection approach that significantly reduces the computational resources required to reliably identify state - of - the - art performance from large collections of candidate models .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [2], "type": "APP"}, {"side": "FN", "category": "BOUNDARY", "offsets": [12, 13, 14], "type": "FEA"}, {"side": "FP", "category": "OTHER", "offsets": [5, 6, 7], "type": "APP"}]（该句其余错误见原文件）。

- `180fbfb92cce987edc29809639e29c2e:1`：despite being known to produce unreliable comparisons , it is still common practice to compare model evaluations based on single choices of random seeds .

  错误：[{"side": "FN", "category": "MISSING", "offsets": [38, 39], "type": "WEA"}, {"side": "FN", "category": "MISSING", "offsets": [48, 49], "type": "FEA"}, {"side": "FN", "category": "MISSING", "offsets": [52, 53, 54, 55, 56], "type": "APP"}]。

### smoke-42-20260916-204747 / dev

[runs/smoke-42-20260916-204747/dev.metrics.json](runs/smoke-42-20260916-204747/dev.metrics.json)

| OG | APP | MOD | FEA | TAK | DST | LIM | STR | WEA | DEG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| N/A (gold=0) | N/A (gold=0) | N/A (gold=0) | N/A (gold=0) | N/A (gold=0) | 0.00 | N/A (gold=0) | N/A (gold=0) | N/A (gold=0) | N/A (gold=0) |

invalid rate：0.00%；unaligned rate：0.00%；平均预测 mentions/句：1.00。

错误分类：FN BOUNDARY=1, TYPE_ONLY=0, MISSING=0；FP BOUNDARY=1, OTHER=1。

逐句错误：[runs/smoke-42-20260916-204747/dev.metrics.errors.jsonl](runs/smoke-42-20260916-204747/dev.metrics.errors.jsonl)。以下按文件顺序展示最多两条有错误的句子，非随机抽样。

- `206628c44ef09a294a61fc2bd0568d59:0`：current state - of - the - art neural dialogue models learn from human conversations following the data - driven paradigm .

  错误：[{"side": "FP", "category": "OTHER", "offsets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "type": "APP"}]。

- `206628c44ef09a294a61fc2bd0568d59:1`：as such , a reliable training corpus is the crux of building a robust and well - behaved dialogue model .

  错误：[{"side": "FN", "category": "BOUNDARY", "offsets": [26, 27, 28], "type": "DST"}, {"side": "FP", "category": "BOUNDARY", "offsets": [27, 28], "type": "DST"}]。

## E1 test 三 seed 汇总

每个 seed 取路径排序最后的 completed/complete、完整 test run；保留上方所有运行记录。均值和样本标准差使用 seeds 42、43、44（n=3，标准差分母 n−1），不是置信区间。

| 指标 | 均值 | 样本标准差 |
| --- | ---: | ---: |
| untyped precision | 57.96 | 0.45 |
| untyped recall | 63.06 | 0.64 |
| untyped f1 | 60.40 | 0.54 |
| typed precision | 49.86 | 0.49 |
| typed recall | 54.13 | 0.64 |
| typed f1 | 51.91 | 0.56 |

使用运行：[runs/E1-42-20260916-204747/run.json](runs/E1-42-20260916-204747/run.json)、[runs/E1-43-20260916-204747/run.json](runs/E1-43-20260916-204747/run.json)、[runs/E1-44-20260916-204747/run.json](runs/E1-44-20260916-204747/run.json)。

## Gold-text roundtrip 诊断

将 gold text/type 送入同一确定性对齐器，检查文本接口的信息损失。它不是模型性能的数学上界：模型可能输出不同顺序或文本而改变映射。重复文本、exact-first 干扰及不连续跨度会导致 roundtrip 错误。

| split | Untyped P | R | F1 | Typed P | R | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [verification/oracle-train.json](verification/oracle-train.json) | 99.07 | 99.05 | 99.06 | 99.05 | 99.05 | 99.05 |
| [verification/oracle-dev.json](verification/oracle-dev.json) | 98.73 | 98.71 | 98.72 | 98.72 | 98.72 | 98.72 |
| [verification/oracle-test.json](verification/oracle-test.json) | 98.94 | 98.92 | 98.93 | 98.92 | 98.92 | 98.92 |

## Q1–Q3

Q1：E1 seed42 完整 test typed F1=51.78%，untyped F1=60.14%。是否满足应用所需的“高质量”仍取决于召回要求，不能仅凭单个总分认定。

Q2：seed42 中 S+D 相对 S 的 typed F1 差为 +0.70 个百分点。E1-S 只有一个 seed，该差异仅是观察结果，不能判断统计显著性或稳定性。

Q3：E1 seed42 的 FN 中 BOUNDARY 最多（944 条）。分类优先顺序是 TYPE_ONLY → BOUNDARY → MISSING；需结合逐句错误和 roundtrip 诊断解释来源。

## 运行证据

- [runs/E0-42-20260916-204747/run.json](runs/E0-42-20260916-204747/run.json)：completed。配置、训练日志、推理原文与错误 JSONL 位于同一运行目录。
- [runs/E1-42-20260916-204747/run.json](runs/E1-42-20260916-204747/run.json)：completed。配置、训练日志、推理原文与错误 JSONL 位于同一运行目录。
- [runs/E1-43-20260916-204747/run.json](runs/E1-43-20260916-204747/run.json)：completed。配置、训练日志、推理原文与错误 JSONL 位于同一运行目录。
- [runs/E1-44-20260916-204747/run.json](runs/E1-44-20260916-204747/run.json)：completed。配置、训练日志、推理原文与错误 JSONL 位于同一运行目录。
- [runs/E1-S-42-20260916-204747/run.json](runs/E1-S-42-20260916-204747/run.json)：completed。配置、训练日志、推理原文与错误 JSONL 位于同一运行目录。
- [runs/smoke-42-20260916-204747/run.json](runs/smoke-42-20260916-204747/run.json)：completed。配置、训练日志、推理原文与错误 JSONL 位于同一运行目录。
