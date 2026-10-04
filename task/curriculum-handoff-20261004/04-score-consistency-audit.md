# 2026-10-04 评分一致性核查

问题是测试评测是否低估模型绝对抽取质量，不是课程学习是否胜过基线。核查未改变模型、预测、数据或官方评分器。

## 直接证据：原始 gold 对自身评分

使用 `/home/myluo/miniconda3/envs/qwen3meld-run/bin/python` 调用原 `utils.calculate_F1`，输入 `data/SciEvents/test.json` 的250篇原始 `events`，gold与prediction使用同一列表。进程退出0。

```python
import json
from utils import calculate_F1
docs = json.load(open('data/SciEvents/test.json'))
gold = [d['events'] for d in docs]
for mode in ('offset', 'token'):
    print(mode, calculate_F1(gold, gold, mode)[:6])
```

| gold自评（%） | TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|
| offset | 100.000000 | 99.754902 | 98.635648 | 98.635648 | 98.997494 | 98.997494 |
| token | 100.000000 | 98.774510 | 96.711562 | 96.694070 | 95.522388 | 95.522388 |

offset有24篇、token有62篇的gold自评AC低于100%。这直接证明评分匹配在完全正确输入上也会损失分数，不能将所有低分归因于模型。

## 代码原因与真实案例

`utils.py` 的 `gold_expr2trg` / `predict_expr2trg` 只按trigger表达建字典，重复键保留最后一个事件；argument字典也仅按表达建键。事件类型、角色及实例身份没有被完整保留。遍历原始事件时，匹配目标已经被覆盖。

- offset：`fd4526d8ccd6b35ae6a285c2e34721f3`，标题 *Making Pre-trained Language Models Better Few-shot Learners*。第10和15号事件（0-based）同为FAC、trigger均为`achieving`、offset均为`[192]`，但前一个Extent为`30 %`、后一个Extent为`11 %`，其它论元也不同。前者匹配到后者，原样自评AC只有95.454545%。
- token：`180fbfb92cce987edc29809639e29c2e`，标题 *FIESTA: Fast IdEntification of State-of-The-Art models using adaptive bandit algorithms*。第5和11号事件同为PUR、trigger文本均为`evaluate`，位置分别为`[109]`和`[151]`，Aim分别是`each model`和`fiesta`。token字典把不同位置的事件合并，原样自评AC只有97.058824%。

## 排除流水线差异的控制

固定同一份baseline已对齐预测 `evaluation-v2-20261003/results/baseline/test-checkpoint-2406.metrics.scievents.json`，按doc_id与原gold配对，仅切换评分模式：offset AC41.879652%，token AC41.659944%。仍有差异，说明两种模式的差别并不只来自两条后处理流水线。

交接表的baseline raw token AC41.611459%，与上述已对齐token也不同，因为raw token保留未对齐预测，而offset链路包含位置回填、过滤及事件去重。因此原表并不是“固定相同事件，只替换比较字段”的实验。

## 结论边界

### 全指标补充核查

用户要求检查所有指标。对同一份baseline已对齐预测，两种模式的完整F1如下；所有指标的precision/recall、输入与源码SHA-256另存于 `04-score-consistency-metrics.json`。

| 同一baseline预测（%） | TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|
| offset | 75.413907 | 63.547920 | 44.203353 | 41.879652 | 48.068670 | 48.068670 |
| token | 76.111686 | 63.329886 | 43.968453 | 41.659944 | 46.886447 | 46.886447 |

TC、AI、AC、EC在gold自评中均出现不满分，证明缺陷并不限于AC。TI自评100%不能证明TI匹配正确：评分器只检查键存在，重复文本的不同位置可能被视为命中，且同一预测可以被多个gold实例复用。EC还会按argument表达是否出现在trigger字典中决定子事件关系，token同文异位可能改变关系识别。不同指标偏差方向不必相同；baseline同预测中token TI较高、其它指标较低，均须逐实例核验。

已经证实评分器会低估某些完全正确答案，用户的怀疑有直接依据。尚未量化实际模型预测中净损失多少；字典覆盖也可能造成错误实例被误计为正确，不能保证更正后所有模型指标单向上升。不能把gold自评距离100%的差值直接加到模型F1。

token与offset在文本—位置一一对应、事件实例和角色均无冲突、两边预测集合相同的前提下应一致；本数据和现有评分器不满足这些前提。后续应独立实现保留实例的一对一匹配与gold往返诊断，并保留官方指标作对照，本轮未执行该修改。
