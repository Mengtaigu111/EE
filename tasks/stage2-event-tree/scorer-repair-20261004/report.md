# 评测修复结果（2026-10-04）

同一批已有预测，CPU离线重算；原官方代码、gold、模型和历史结果均保留。三组全部19组dev/test评估的旧官方offset分数精确复现。新增实例一对一评分器是诊断口径，不替代官方benchmark。

## 问题与修复

- 官方字典按trigger/argument表达覆盖实例，漏记同trigger不同事件、同论元不同role，也可能一条预测重复命中多个gold。新评分保留重复实例、1:1配对，TI/TC/AI/AC/EC_I/EC_C全量输出P/R/F1和TP/FP/FN。
- 同(type, trigger)事件通过最大权匹配配对，依次最大化AC、AI、EC_C、EC_I；并非每项独立任意重配。EC沿用官方“论元表达属于文档trigger集合”的定义，token模式仍不能识别同文异位的真实事件边。
- 旧文本树编码排序并去重offset：95个gold论元的原始顺序或重复项已在编码时丢失。新可选lossless_forest协议每个mention带`(At 全局零基token位置...)`，严格保存并核验offset顺序与重复项；dev/test gold全量往返各项100%。未改现有prompt或重训，不能补回旧模型没有输出的位置。
- 新ambiguity_resolver完整、有界地枚举连续及不连续同token候选；多候选明确标记ambiguous，截断标记truncated，绝不把猜测当唯一答案。它不读取gold来选择实际预测位置。全量gold诊断中，旧算法漏枚举的40处位置全部进入新候选，但40处均ambiguous（最大141个候选、默认256未截断）。旧预测仍按历史merge对齐，方便同批对比。

## Test F1：旧官方与新实例评分

token表使用原始模型文本，保留无法对齐输出；offset表沿用原已对齐预测。两表预测集合略不同，不能将差值全部解释成位置误差。

| 指标 | 基线旧offset | 基线新offset | Sorted1旧offset | Sorted1新offset |
|---|---:|---:|---:|---:|
| TI | 75.41% | 74.80% | 75.19% | 74.46% |
| TC | 63.55% | 63.14% | 63.10% | 62.57% |
| AI | 44.20% | 44.17% | 44.16% | 43.95% |
| AC | 41.88% | 41.87% | 41.98% | 41.75% |
| EC_I | 48.07% | 47.60% | 48.19% | 47.73% |
| EC_C | 48.07% | 47.60% | 48.19% | 47.73% |

| 指标 | 基线旧raw token | 基线新raw token | Sorted1旧raw token | Sorted1新raw token |
|---|---:|---:|---:|---:|
| TI | 76.10% | 74.91% | 75.91% | 74.62% |
| TC | 63.32% | 63.29% | 62.84% | 62.79% |
| AI | 43.92% | 44.76% | 43.68% | 44.53% |
| AC | 41.61% | 42.45% | 41.52% | 42.34% |
| EC_I | 46.83% | 47.74% | 46.88% | 48.17% |
| EC_C | 46.83% | 47.74% | 46.88% | 48.17% |

## 新口径全部P/R/F1（%）

| 实验 | 模式 | 指标 | P | R | F1 |
|---|---|---|---:|---:|---:|
| baseline | offset | TI | 76.0338 | 73.6111 | 74.8028 |
| baseline | offset | TC | 64.1772 | 62.1324 | 63.1382 |
| baseline | offset | AI | 45.3473 | 43.0471 | 44.1673 |
| baseline | offset | AC | 42.9888 | 40.8081 | 41.8701 |
| baseline | offset | EC_I | 46.7391 | 48.4962 | 47.6015 |
| baseline | offset | EC_C | 46.7391 | 48.4962 | 47.6015 |
| baseline | raw_token | TI | 76.1282 | 73.7337 | 74.9118 |
| baseline | raw_token | TC | 64.3189 | 62.2958 | 63.2911 |
| baseline | raw_token | AI | 45.9007 | 43.6768 | 44.7611 |
| baseline | raw_token | AC | 43.5294 | 41.4203 | 42.4487 |
| baseline | raw_token | EC_I | 46.9880 | 48.5075 | 47.7356 |
| baseline | raw_token | EC_C | 46.9880 | 48.5075 | 47.7356 |
| Easy2 | offset | TI | 73.0525 | 75.0817 | 74.0532 |
| Easy2 | offset | TC | 61.3275 | 63.0310 | 62.1676 |
| Easy2 | offset | AI | 43.8606 | 42.9246 | 43.3876 |
| Easy2 | offset | AC | 41.5192 | 40.6332 | 41.0714 |
| Easy2 | offset | EC_I | 46.2455 | 48.6216 | 47.4038 |
| Easy2 | offset | EC_C | 46.2455 | 48.6216 | 47.4038 |
| Easy2 | raw_token | TI | 73.2221 | 75.2859 | 74.2397 |
| Easy2 | raw_token | TC | 61.5415 | 63.2761 | 62.3968 |
| Easy2 | raw_token | AI | 44.4504 | 43.5718 | 44.0067 |
| Easy2 | raw_token | AC | 42.0949 | 41.2629 | 41.6748 |
| Easy2 | raw_token | EC_I | 46.7223 | 48.7562 | 47.7176 |
| Easy2 | raw_token | EC_C | 46.7223 | 48.7562 | 47.7176 |
| Sorted1 | offset | TI | 71.4179 | 77.7778 | 74.4623 |
| Sorted1 | offset | TC | 60.0150 | 65.3595 | 62.5733 |
| Sorted1 | offset | AI | 41.9140 | 46.1956 | 43.9507 |
| Sorted1 | offset | AC | 39.8191 | 43.8867 | 41.7540 |
| Sorted1 | offset | EC_I | 43.5818 | 52.7569 | 47.7324 |
| Sorted1 | offset | EC_C | 43.5818 | 52.7569 | 47.7324 |
| Sorted1 | raw_token | TI | 71.5088 | 78.0229 | 74.6239 |
| Sorted1 | raw_token | TC | 60.1647 | 65.6454 | 62.7857 |
| Sorted1 | raw_token | AI | 42.3301 | 46.9652 | 44.5274 |
| Sorted1 | raw_token | AC | 40.2491 | 44.6563 | 42.3383 |
| Sorted1 | raw_token | EC_I | 44.0660 | 53.1095 | 48.1669 |
| Sorted1 | raw_token | EC_C | 44.0660 | 53.1095 | 48.1669 |

## TI位置差异的直接证据

固定同一批已对齐预测和新匹配器，token比offset额外匹配的trigger：

- baseline：offset TP=1802，token TP=1805，同文异位额外命中3个，跨句1个。
- Easy2：offset TP=1838，token TP=1843，同文异位额外命中5个，跨句3个。
- Sorted1：offset TP=1904，token TP=1910，同文异位额外命中6个，跨句3个。

真实例子：文档`fd4526d8ccd6b35ae6a285c2e34721f3`的`fine - tuning`，gold位置[94, 95, 96]、预测位置[79, 80, 81]。token认作相同文本，offset认作不同位置。不能仅凭token较高断言模型更好。完整例子见[actual-examples.json](actual-examples.json)。

原gold自评旧offset：TI100、TC99.7549、AI/AC98.6356、EC98.9975%；旧token：TI100、TC98.7745、AI96.7116、AC96.6941、EC95.5224%。新实例评分的dev/test gold两模式18项均100%。自评损失不能直接加到模型分数。

## 验证与后续使用

- 三组dev重选仍为checkpoint-2406；未用test选模。
- 41项CPU测试通过（实例匹配、runner闸门、无损格式、候选枚举）；新组件源文件见[final-source-manifest.json](final-source-manifest.json)。
- 19组独立原预测重构复算通过；1000次最优匹配穷举oracle通过，dev/test两模式gold自评18项均100%。[verification.json](verification.json)记录实际检查数量。
- 历史源码hash独立核验、全部句子/文档覆盖检查、已对齐文本与源offset一致检查通过。
- [evaluation-results.json](evaluation-results.json)：全checkpoint、全18项、计数和hash；[gold-roundtrip-report.md](gold-roundtrip-report.md)：表示及回填损失逐例报告；[independent-review.md](independent-review.md)：独立复算结论。
- 新训练若要消除位置猜测，需要显式位置输出，必须另开实验；仅文本的多位置歧义在旧输出中无法无损恢复。官方和新实例口径并列保存，不能混比。
