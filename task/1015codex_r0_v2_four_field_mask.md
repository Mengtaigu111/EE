# Codex 执行任务：重做 R0-v2，并接入四字段选择性策略更新

> **直接检查本地代码、实现、测试；环境允许时，在现有 RTX 4090 上串行运行 A/B 短程对照。不要只回复设计建议。**
>
> 本文是本轮完整执行规范，**替代 `codex_r0_structural_mask_v1.md` 中的旧奖励、旧匹配和两字段捆绑 mask 方案**。不需要读聊天记录才能执行。
>
> 本轮目标：**用文本建立结构内对应，分别评价 Trigger、Type、Argument、Role；同一份匹配结果同时用于答案奖励和字段 mask。** 不增加 rollout，不做前缀分支，不设计新的局部优势。

## 0. 先确认变更范围

| 旧要求 | 本轮替换为 |
|---|---|
| 严格 R0 不动 | 训练主方案改为本文 R0-v2；旧奖励已实现则保留为历史模式 |
| `(Type, Trigger)` 相同才允许事件配对 | **完整 Trigger token 文本相同**即可候选配对，Type 可以不同 |
| 父事件 Type 错，下属论元全部失去得分资格 | Type 不限制已对应事件内的 Argument、Role 得分 |
| Role 必须属于预测 Type 的合法角色集合 | **不以此决定奖励或 mask**；Role 只与对应 Gold Role 比较 |
| Type 与 Trigger、Role 与 Argument 各共用标签 | 四类数据字段分别使用自己的命中标签 |
| 旧严格解析失败则奖励 0 | 区分动作语法错误与语义标签错误；只有真正的语法/结构解析失败才整份归零 |
| 旧 mask 的 A=0 数据位置保留 | 新 `field_v2` 在 A=0 时数据位置 mask=0；控制/格式仍保持原样，KL 不变 |

**不再做“旧 R0 占 80% + 辅助分占 20%”。也不同时增加边际贡献、符号翻转、熵筛选、额外奖励或控制动作规则。**

## 1. 当前项目与固定条件

项目为 Qwen3-8B 科学事件抽取。旧 Type-CoT 热身和 Action SFT 已完成。使用现有 LoRA、GRPO，G=4。输出仍是：

```text
NEW(TYPE)
TRG("trigger text")
ARG(ROLE, "argument text")
END
STOP
```

历史定位线索为 `Mengtaigu111/EE`、分支 `review/action-sft-grpo-v2-20261012`、目录 `tasks/action-experiment/`。**这些不是服务器当前 HEAD 的核验结果。先检查工作区，不擅自切分支、reset、覆盖未提交修改或删除已有 run。**

以下保持不变：

| 项目 | 要求 |
|---|---|
| 初始化 | A/B 都从同一份已完成 Action SFT 的最佳 `best-adapter` 独立启动 |
| Reference | 同一份完整 Action SFT adapter，冻结；不是纯 Base，也不是 A 训练后的模型 |
| Prompt | 沿用该 SFT 的 `row['prompt']`、manifest/provenance 和已锁定 v2 模板；不开 thinking |
| 动作语言 | 保留 NEW/TRG/ARG/END/STOP；不删 END，不增加 PATCH 或坐标输出 |
| 数据 | 原 train/dev/test、None 样本、拆词规则及数据顺序规则不变 |
| 采样 | G=4；原 temperature/top-p/top-k/长度上限/采样概率口径不变 |
| GRPO | 原组内优势、clip、KL、loss 分母、梯度累积、优化器和调度不变 |
| 正式评测 | 原 TI/TC/AI/AC/EC、presence 评分、解析器和 dev 选模规则不变 |

旧执行稿记录的配置参考：`lr=1e-5, clip=0.2, beta=0.01, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=768, seed=42, LoRA dropout=0`，优势使用组内样本标准差。

**先读取本地已验证配置。若与上述旧文字不同，记录实际差异，A/B 统一使用同一份配置；不要顺手调参或升级依赖。** 本文不要求重跑 SFT、重建数据、迁移到 TRL/verl、修改生成引擎或增加模型前向。

## 2. 实施前检查与源码边界

检查训练实际使用的奖励、解析、Gold 转换、completion IDs、logprob、loss 入口及来源校验。优先查看下列位置，路径以本地为准：

```text
tasks/action-experiment/action_rewards.py
tasks/action-experiment/train_grpo.py
tasks/action-experiment/grpo_core.py
tasks/action-experiment/action_common.py
tasks/action-experiment/action_evaluation.py
tasks/action-experiment/action_update_mask.py   # 若已存在
tasks/action-experiment/tests/
```

先简短报告实际入口和已有进度，再继续实施。旧 R0 或旧 mask 尚未实现，不构成必须先把旧方案做完的理由。

旧稿将 `action_evaluation.py`、`scripts/score_tree_presence.py`、`scripts/action_language.py` 列为需要保持的评测/来源锁文件。**保留这些文件及其原有行为；先核对本地来源校验，不绕过 hash 检查。**

新训练解析、匹配和奖励可以放进独立 `action_reward_v2.py`；字段到 token 的映射可复用或扩展训练侧 `action_update_mask.py`。不要为放宽训练奖励而修改正式评测器，或伪造旧 SFT 的 manifest。新模块纳入本次 RL 的源码快照与 hash。

## 3. 训练解析：只把语法失败作为整份奖励为零的条件

### 3.1 保持动作结构严格

训练奖励侧保留原动作语法和状态顺序：

```text
(NEW(TYPE) -> TRG("text") -> ARG(ROLE,"text")* -> END)* -> STOP
```

必须有正确的动作顺序、完整字符串和最终 STOP。空字段、引号/转义、行格式等沿用现有动作语言的语法约束；不要自行支持另一套语法。

以下属于真正的语法/结构失败：无法读取动作、必要括号或引号缺失、NEW 后没有 TRG、未 END 就打开新事件、缺 END/STOP、STOP 后又有实质输出等。此时 `syntax_valid=false, reward=0`；不修复、不截取一个合法子集评分。

模型 EOS 与动作 STOP 不相同。预算用尽前已有合法 STOP、但未生成模型 EOS，仍可按完整动作评分；缺 STOP 的截断输出不合法。loss 对第一个 EOS、padding、EOS 后位置的处理沿用旧实现。

### 3.2 语义标签错误不得提前清零或删记录

以下信息能按动作文法解析时，保留原始字段和完整实例，交给后续精确匹配：

- Type 与 Gold 不同，或标签语法合法但 Type 不在已知枚举中。
- Role 与 Gold 不同，或标签语法合法但 Role 不在已知枚举中。
- Role 不属于**预测 Type**允许的角色集合。
- Trigger/Argument 文本未命中 Gold，或未通过旧的源文本出现性检查。

这些情况会使相应字段不命中，**但不能先返回全句 0、删除 ARG、改写 Role、纠正 Type 或清空事件**。训练文法仍要求合法的标签写法和字符串结构，不是允许任意损坏文本。

关键验收：预测 `WKS + uses + BaseComponent + BERT`，对应 Gold 为 `MDS + uses + BaseComponent + BERT` 时，必须保留 ARG，并得到 0.75。

若旧解析器把 schema 检查与语法解析绑定，新建训练专用解析路径。不能先调用会过滤字段的旧解析器，再对过滤后的结果计算新奖励。生成侧、训练样本筛选侧也不得用旧的 schema-invalid 标记提前阻断 R0-v2。

建议区分 `syntax_valid` 与可选的 `schema_compatible` 诊断字段；后者不作为奖励乘数、mask 门槛或样本过滤条件。正式 dev/test 仍走原评测路径。

Gold 从原始标注及其 token 字段读取；保持 Gold 不变。Gold 本身损坏属于数据问题，应报错定位，不能伪装成预测语法错误。

## 4. R0-v2 的唯一计分定义

### 4.1 单位和文本相等

```text
Event instance = (内部实例索引, Type, Trigger token 文本元组, Argument 列表)
Argument       = (内部实例索引, Role, Argument token 文本元组)
```

匹配使用项目原有的**有序 token 文本**，不是 Qwen BPE IDs，也不是源句 offset。Gold 优先使用原 `tokens`；预测用原有纯文本拆词规则。不要额外 lowercase、排序、词形还原、模糊边界、同义替换或语义相似度。

保留重复事件、重复论元和原始顺序。内部实例索引只定位记录，不直接当作 Gold/预测相等条件。不同事件允许复用同一串源文本；不消费源 token。

### 4.2 四类命中

| 字段 | 命中条件 |
|---|---|
| Trigger | 该预测事件通过完整 Trigger 文本与一个 Gold 事件一对一配对 |
| Type | 已配对事件中，预测 Type 与该 Gold Type 相同 |
| Argument | 已配对事件内部，该预测论元通过完整文本与一个 Gold 论元一对一配对 |
| Role | 已配对论元中，预测 Role 与该 Gold Role 相同；**不检查预测 Type 的角色集合** |

Trigger 不对应时，该事件四类字段都不能跨事件找分。Argument 文本不对应时，该 ARG 的 Argument、Role 均不命中，不能仅凭 Role 名称给分。

Type 未命中，不影响已配对事件内的 Argument/Role 命中；Role 未命中，不影响该 Argument 文本命中。

### 4.3 分母与主公式

对一个完整候选答案，记：

```text
D_E = 预测事件总数 + Gold 事件总数
D_A = 预测论元总数 + Gold 论元总数

C_T = Trigger 命中数
C_Y = Type 命中数
C_A = Argument 文本命中数
C_R = Role 命中数
```

当 `D_E>0` 且 `D_A>0`：

```text
F_trigger  = 2*C_T / D_E
F_type     = 2*C_Y / D_E
F_argument = 2*C_A / D_A
F_role     = 2*C_R / D_A

R0_v2 = (F_trigger + F_type + F_argument + F_role) / 4
```

分母包含**全部**预测/Gold 实例，包括未配对、额外、重复及漏掉的实例。不能只对已配对部分算分类 accuracy，也不能先去重。

只输出一个标量奖励给 GRPO。**先合成上述标量，再按旧算法计算一次组内优势；不得把四项各自归一化后合并优势。**

### 4.4 空值与非法输出

| 条件 | 奖励与分项 |
|---|---|
| `syntax_valid=false` | 奖励 0；分项标为 NA，匹配 trace 不可用；不评分合法前缀 |
| 合法 STOP 且两侧均无事件 | 奖励 1；计数均 0，四分项标为 NA，case=`both_empty` |
| `D_E>0, D_A=0` | `R0_v2=(F_trigger+F_type)/2`；Argument/Role 两项为 NA，不参与平均 |
| `D_A>0`，只有一侧有论元 | Argument/Role 分项均 0；仍按四项平均 |
| 只有一侧有事件 | 奖励 0；有效分项按上述规则为 0，不赠送空论元分 |

NA 是日志/返回值的显式 `None` 或 active 标记，不是进入 loss 的 NaN。奖励始终是有限标量。不能把缺失的两项填 0 后仍除以 4，也不能填 1 赠分。

## 5. 同一次最优匹配产生计数和字段标签

### 5.1 候选边

事件候选边只允许 `pred.trigger_tokens == gold.trigger_tokens`。按 Trigger 文本分组，组内做一对一匹配。**分组键不包含 Type。**

不同 Trigger 组不互借论元。相同 Trigger 的多个事件不按列表顺序贪心配对，不先锁死 Type 相同的配对。

### 5.2 给每个事件候选对计算论元配对

对于候选事件对 `(p,g)`：

1. 在这两个事件内部，按完整 Argument 文本分组。
2. 每个相同文本组内，先尽量匹配 Role 相同的实例，再将剩余实例按固定顺序配对。
3. 每个预测/Gold ARG 最多使用一次。得到具体论元对应、文本命中数 `a_pg` 和 Role 命中数 `r_pg`。

该方法在不减少文本配对数的情况下最大化 Role 命中数。Role 比较不受预测 Type 限制。未选中的事件候选边的论元配对不能进入最终 trace。

### 5.3 事件边权：必须优化最终 F1 组合

当 `D_A>0`，对一个 Trigger 相同的候选事件对：

```text
t_pg = int(pred.Type == gold.Type)

w_pg = (1+t_pg)/(2*D_E) + (a_pg+r_pg)/(2*D_A)
```

最大化选中边的 `sum(w_pg)`，等价于最大化 R0-v2。

可直接使用以下等价整数权重，避免浮点比较：

```text
W_pg = D_A*(1+t_pg) + D_E*(a_pg+r_pg)
# sum(W_pg) / (2*D_E*D_A) == R0_v2
```

**D_E、D_A 均是整份候选答案的总数，不是当前事件或 Trigger 分组的局部数。** 不要用 `1+t_pg+a_pg+r_pg` 代替权重；那可能选出不同结果。

当 `D_A=0, D_E>0`，仅按 `W_pg=1+t_pg` 做组内最优匹配，最终奖励为 `(C_T+C_Y)/D_E`。双方无事件的情况在匹配前处理。

复用现有可用的最大权指派实现；需要补齐时采用高效实现。允许 dummy 未匹配项，权重为 0。不要在正式训练里枚举所有排列。相同 Trigger 组内真实边权均为正，最优指派会用尽可配对实例容量。

同分采用确定性规则，保留原实例索引，重复调用结果可复现。次级同分规则不得改变主目标；不要随机抽取，也不要用可能改变主目标的浮点扰动。输入换序不应改变最优奖励；等价重复实例的具体 occurrence 标签可以随固定同分规则变化。

### 5.4 返回同一份 trace

返回结构可按本地代码适配，但至少包含：

```python
result = {
    "reward_version": "r0_v2",
    "reward": ...,                         # 有限 float
    "syntax_valid": ...,
    "reward_case": ...,                    # ordinary/head_only/both_empty/syntax_invalid
    "components": {"trigger": ..., "type": ..., "argument": ..., "role": ...},
    "counts": ...,                         # 四类命中和两侧完整实例数
    "event_pairs": [(pred_e, gold_e), ...],
    "argument_pairs": [(pred_e, pred_a, gold_e, gold_a), ...],
    "event_hits": [                         # 按预测原始事件顺序
        {"trigger": True, "type": False},
    ],
    "argument_hits": [                      # 按预测事件、预测 ARG 原始顺序
        [{"argument": True, "role": True}],
    ],
}
```

先把所有预测字段标签置 False，再由**最终选中的配对**赋值。未配对/超过容量的实例保持 False。四类 True 数必须与 `C_T,C_Y,C_A,C_R` 分别一致。

奖励和 mask 共用该 trace；不为 mask 另写一套贪心匹配，不因为启用 mask 改变奖励。类型或组合不合法但语法完整的样本仍有正常 trace，不触发 mask 回退。

## 6. 四字段 mask：只选择位置，不创造或翻转优势

### 6.1 字段单位

```text
NEW(MDS)                  -> Type 数据内容
TRG("uses")               -> Trigger 引号内文本
ARG(BaseComponent,"BERT")  -> Role 数据内容，以及 Argument 引号内文本
```

每个完整字段使用一个命中标签，再映射到该字段的生成 token。不是逐词打分；不再把 Type/Trigger 或 Role/Argument 捆绑。

对原 GRPO 算出的候选优势 `A_i` 和字段命中 `h`：

```text
A_i > 0  -> m = int(h)
A_i < 0  -> m = 1 - int(h)
A_i == 0 -> m = 0
```

读取真实优势符号，不用 `reward>0.5` 等阈值，不分别计算四套优势，不设新的近零阈值。A 非有限时按训练错误处理，不当作零。

本轮固定 A=0 时数据位置为 0；策略项本来已为零。控制/格式位置仍为 1；KL 不被 mask。**不因为同分组新增重采样、跳过组或跳过 KL 的规则；原训练器已有的组处理需保持并记录。**

### 6.2 必须出现的处理结果

Gold 是 `MDS / uses / BaseComponent / BERT`；预测是 `WKS / uses / BaseComponent / BERT`：

| 字段 | 命中 | A>0 | A<0 |
|---|---:|---:|---:|
| WKS | 0 | 0 | 1 |
| uses | 1 | 1 | 0 |
| BaseComponent | 1 | 1 | 0 |
| BERT | 1 | 1 | 0 |

若 Role 改成 Content，则 Content 命中为 0，BERT 仍为 1。不要因父 Type 错而把两个下游字段全部屏蔽或全部判错。

正确字段允许在错误标签构成的实际前缀下接受正优势，这是本实验明确接受的近似；不要自行加入第三状态或新的依赖保护。

### 6.3 控制与格式不另改

NEW/TRG/ARG 动作名、END、STOP、括号、引号、逗号、换行和原本有效的首个 EOS 保持原有全局策略信号。仅四类数据内容接受新规则。

原 prompt、padding、EOS 之后的位置仍由原有效 completion mask 排除。漏掉的字段没有对应 token；不为缺失字段插入 Gold，不对 END/STOP 新增独立奖惩。

## 7. 从原始字段区间映射到真正生成的 token

复用旧工程中已验证的定位能力，但**不能复用依赖预测 Type/schema 合法性的筛选条件**。

### 7.1 保留原始生成历史

保存原始 completion IDs、原始解码文本、原有效 completion mask 和行为概率。current/old/reference logprob、teacher forcing 和 loss 仍使用真正采样的 IDs。

Gold 只用于训练侧匹配；不进入 prompt，不修改或重排采样答案。不能把匹配后重序列化的动作拿去训练。

### 7.2 原文区间

训练解析/定位返回 Type、Trigger、Role、Argument 数据内容在**原始 completion 文本**中的 `[start,end)` 字符区间，绑定原事件及 ARG 实例索引。

Type/Role 区间不含动作名、标点或周围空白；Trigger/Argument 区间是字符串引号内的原始表达，包括其中的转义序列。评分读取解码后的字段值，但位置必须可追溯到原始文本。

不要只用 `find("BERT")` 或同名 Role 的第一次出现位置。重复表达、引号转义和多事件均按 occurrence 定位。定位前不得无映射地 strip、合并空格或重新排版。

这些是**生成答案内部的 loss 定位区间**，不是要求模型输出源句 offset。

### 7.3 BPE 对齐与边界 token

优先复用可信的原始 token→字符映射。若通过额外 tokenize 取得 offsets，必须验证对应非 special IDs 与真正采样的 IDs 逐项相等。这次分词只用于定位，不替换训练 IDs。

固定保守规则：

- token 完全落在一个数据字段内部：采用该字段 mask。
- token 同时跨数据/格式或多个字段：mask=1，计为 mixed-boundary token。
- special token 或没有可靠字符范围的控制 token：保持旧处理。

不要拆分 BPE token，也不要凭重叠一点就屏蔽整 token。mixed token 不触发整句回退。

mask 对应的是**目标生成 token 的 logprob**；核对 causal shift，不能错乘到前一个或后一个预测位置。padding、候选展平/排序、microbatch 和梯度累积后，IDs、A、trace、mask 必须仍对应同一候选。

### 7.4 失败处理

语法无效、trace 无法追溯或 token/区间对齐无法验证时，该候选 mask 回退全 1，沿用原策略更新，并记录原因。

- 语法无效：奖励已按第 3 节为 0。
- **仅定位失败**：奖励保持原来算出的 R0-v2，不能改成 0。
- Type–Role 组合不合法但可解析：是正常评分/定位样本，不得因此回退。
- 合法空 STOP：没有数据字段，mask 全 1，是正常样本而非失败。

不要吞掉 shape 错误、CUDA 错误、NaN 等程序异常。若几乎所有真实样本回退或所有字段都被当成 mixed，不能报告 mask 已有效实现；先修正定位路径，不增加新方法。

## 8. loss 接线：只改策略项分子

按原代码计算 token 策略损失，包括原 ratio、advantage 和 clipping，再乘 mask。`baseline_reduce` 指当前训练器的原始聚合方式，不假定它属于某个框架。

```python
# pg_token_loss: 原逐 token 策略项，已完成 ratio/advantage/clipping。
# valid_mask: 原有效 completion mask。
# field_mask: 无梯度的字段选择张量，shape 与对应 token loss 一致。

if field_mask is None:              # --policy-update-mask off
    policy_loss = baseline_reduce(pg_token_loss, valid_mask)
else:
    policy_loss = baseline_reduce(
        pg_token_loss * field_mask.detach(),
        valid_mask,                # 分母仍由原有效位置决定
    )

loss = policy_loss + original_kl_loss
```

必须保持：

- KL 的计算、符号、beta、有效位置和聚合不变。不要把 mask 乘到已经合并 KL 的总 token loss。
- 不把 `valid_mask*field_mask` 传给以 mask.sum() 作分母的 masked mean；不按保留 token 数放大梯度。
- 原来按序列长度或其他规则聚合，继续原规则；不改成新的 batch mean。
- 不修改 old/current/reference logprob、采样分布、ratio、EOS mask、优化器时机或梯度累积。
- 不在 generation、softmax 或 attention mask 中使用字段 mask。
- mask=off 只关闭位置选择，**不会把明确指定的 R0-v2 切回旧奖励**。

若当前实现把策略项和 KL 合在一起，最小拆分后做上述接线；给定相同 logprob、优势和全 1 mask，新 loss/梯度必须与原数学形式一致。

采用原分母意味着策略项可能变弱，KL 相对影响可能增强。本轮不加补偿系数。这不是无偏的原始 GRPO 改写；只验证这套启发式选择是否有效。

## 9. CLI、版本和修改范围

建议增加或复用以下等价参数，名称可以按本地风格适配，但必须进入真实训练路径：

```text
--reward-version r0_v2
--policy-update-mask off
--policy-update-mask field_v2
```

`field_v2` 仅与 R0-v2 的四字段 trace 搭配。旧 `structural_v1` 若已存在，保持其历史含义，不用同一个名字悄悄换实现。错误的版本组合应明确报错。

旧命令未显式选择新奖励时，保持原默认行为，避免误改历史任务；本轮两个启动命令都必须显式选择 `r0_v2`。旧 R0 未实现则无需专门补做旧训练。

run config 至少保存真实生效的 reward version、匹配版本、mask mode、四项等权、无论元规则、语法/schema 分离规则和既有训练超参数。禁止只写一个不起作用的配置键。

| 文件/职责 | 本轮必要修改 |
|---|---|
| 新训练侧奖励模块 | 语法解析、纯文本转 token、实例最优匹配、四项计分、trace |
| 现有奖励路由 | 按显式版本调用新模块；已有旧分支不破坏 |
| 训练侧 mask 模块 | 四字段标签、原文区间、BPE 对齐、二值 mask 和回退 |
| train_grpo.py | 奖励/trace/A/mask 传递，参数、日志和新 run provenance |
| grpo_core.py | 原策略损失位置支持可选 mask，KL/分母不变 |
| 测试与启动脚本 | 小型单元测试、真实 smoke、串行 A/B 命令 |

允许合并小模块以减少重复代码。不要引入奖励模型、Critic、SFT 回放、额外候选、动态权重、新训练框架或大型重构。

## 10. 最小验收测试：明确预期，不增加研究实验矩阵

### 10.1 计分数值

默认 Gold：一个 `MDS / uses` 事件，含一个 `BaseComponent / BERT` 论元。所有用例先在记录级验证，再对关键语法样例验证训练解析入口。除注明外，其余字段与 Gold 相同。

| 测试 | R0-v2 预期 |
|---|---:|
| 全部正确 | 1 |
| Type 改为 WKS，保留 BaseComponent 与 BERT | 0.75 |
| 仅 Role 改为 Content，文本正确 | 0.75 |
| Type=WKS 且 Role=Content，文本正确 | 0.50 |
| Argument 文本错误，但 Role 名相同 | 0.50 |
| Trigger 文本不对应 | 0 |
| 不输出唯一论元，事件头正确 | 0.50 |
| 只找到 Trigger，Type 错且不输出论元 | 0.25 |
| 多输出一个完全重复论元 | 5/6 |
| Gold 一个事件，预测复制整个正确事件为两个 | 2/3 |
| Gold 两个不同 Trigger 的正确事件，只预测其中一个，各含一个论元 | 2/3 |
| 两个不同 Trigger 的事件头正确，但把两者的不同论元文本互换 | 0.50 |
| 双方均无论元，事件头正确 | 1 |
| 双方均无论元，只有 Type 错 | 0.50 |
| Gold 无论元，预测增加一个论元，事件头正确 | 0.50 |
| 合法 STOP，双方无事件 | 1 |
| 只有一侧有事件 | 0 |
| 动作语法不完整/缺必要 END 或 STOP | 0 |
| Type 为语法可读的 UNKNOWN_TYPE，其余正确 | 0.75 |
| Role 为语法可读的 UNKNOWN_ROLE，其余正确 | 0.75 |

另一个必须覆盖的两论元样例：

```text
Gold:
NEW(MDS)
TRG("leverages")
ARG(BaseComponent, "adversarial domain adaptation framework")
ARG(Target, "introduce")
END
STOP

Prediction:
NEW(WKS)
TRG("leverages")
ARG(Content, "adversarial domain adaptation framework")
ARG(Target, "introduce")
END
STOP
```

预期四项为 `(1, 0, 1, 0.5)`，奖励 **0.625**。这是构造数值验收，不代表正式评测分数。

### 10.2 匹配与结构测试

必须检查：

- 同 Trigger 的多个实例、不同 Type、相同/重复论元、跨事件复用文本；不去重，不跨父事件借用。
- 同文本论元先匹配相同 Role。例如同一事件内 `R1/x, R2/x` 对 `R2/x, R1/x` 应两条 Role 都命中。
- 预测/Gold 实例换序后最优奖励不变；同一有序输入重复执行，配对及 trace 确定。
- `C_Y<=C_T`，`C_R<=C_A`，计数不超过两侧相应实例容量；四类 hit 标签的求和等于四类计数。
- 同一次 trace 同时驱动奖励与 mask；mask 开关不改变该候选的奖励/计数。

**权重正确性专用小例子（记录级符号标签即可）：**

```text
Gold:
  G0 = Type A, Trigger t, ARG(R, a)
  G1 = Type B, Trigger t, ARG(R, b)

Prediction:
  P0 = Type A, Trigger t, ARG(R, b)
  P1 = Type B, Trigger t, ARG(R, a)
```

此时 `D_E=4, D_A=4`，交换配对得 0.75，高于按相同 Type 配对的 0.50。不能先锁死 Type 配对。

然后仅给 P0 增加 6 个 Gold 中没有的不同论元文本 `x0...x5`。此时 `D_A=10`，按相同 Type 配对得 **0.50**，交换配对得 **0.45**，最优选择应反转。

若直接最大化命中总数，会在后一例误选交换配对。这个测试可检查是否真正使用全局分母对应的权重。小规模暴力枚举可作为测试 oracle，不能用于正式训练。

### 10.3 解析、mask 与损失测试

| 检查 | 预期 |
|---|---|
| 预测 Type 下非法、但等于 Gold 的 Role | 训练解析不丢 ARG；Role 计分，按第 6 节更新 |
| 旧解析会做 schema 过滤的样本 | 新入口不先经过该过滤；raw 字段/实例完整 |
| Type 错、其余对；Role 错、文本对 | 四字段标签不同，正负优势分别产生对应 mask |
| 未配对/超容量实例 | 对应字段为未命中，不因文本在别处出现而全数保护 |
| A=0 与同分组 | 不创造策略优势；有效数据 mask=0；控制/格式原样；KL 独立 |
| 真正语法失败 vs 仅对齐失败 | 前者奖励 0；后者保留奖励；二者 mask 回退可区分 |
| 重复文本、转义、多事件、真实 tokenizer | mask 落在正确 occurrence，采样 IDs 不被替换 |
| mixed-boundary token | 保留并记录，不错误地整句回退 |
| causal shift、padding、候选展平 | mask 与目标 token loss 对齐，不错位 |
| 合法 STOP 但无模型 EOS；第一个 EOS 后 padding | 奖励与有效 token 范围按规定处理 |
| mask off 或传入全 1 | 同一组输入张量和优势下，与原 loss/梯度一致，容许原精度数值误差 |
| 被屏蔽位置 | 对独立逐 token 策略变量的直接策略导数为 0；不是要求总模型梯度为 0 |
| KL 与归一化分母 | on/off 的 KL 值和覆盖位置相同，分母保持原始值 |
| 正式评测回归 | 同一批 raw predictions 的原 TI/TC/AI/AC/EC 修改前后完全一致 |

R0-v2 与旧 R0 **本来就允许不同**，不要用“新旧奖励数值完全一样”作为验收条件。对旧奖励模式的兼容回归与本轮新奖励测试分开。

保留约 8–12 个可人工检查的样例，包含原动作、四项分数、真实/测试用 A、配对、字段标签、token keep/drop、回退原因。人工 A 只用于单测，不用于正式训练。

## 11. 日志：够定位问题即可

保留原训练日志，在其上增加或更新：

```text
reward_version / matching_version / policy_update_mask_mode
reward_r0_v2
reward_trigger_f1 / reward_type_f1 / reward_argument_f1 / reward_role_f1
matched_trigger_count / matched_type_count / matched_argument_count / matched_role_count
pred_event_count / gold_event_count / pred_argument_count / gold_argument_count
syntax_valid / reward_case
positive_data_keep_ratio / negative_data_keep_ratio
mask_fallback_rate / fallback_reason_counts
mixed_boundary_token_count
```

四项奖励名不要冒用原 dev 的 TI/TC/AI/AC。语法无效、both-empty 或未激活分项记 NA；分项均值只对该分项实际有效的样本求平均，并记录样本数。总 reward 均值仍包含非法输出的 0 分及合法空样本的 1 分，不能过滤后报高分。

保留率按 A>0、A<0 分开统计。分母为可靠定位且完全落入数据字段的有效 token；A=0、控制/格式、mixed 和整条回退样本不混入数据保留率。空分母记 NA，按累积计数求比值，不简单平均样本百分比。

可同时报告全有效 completion token 的总保留率；它不能代替字段保留率。A 组 mask=off 不必为了 mask 日志额外做 BPE 定位，该类日志可记 NA。必要的四字段覆盖数放在小型抽查中即可，不需要全量逐 token 日志。

复用已有输出长度、截断率、事件/论元数、KL、clip fraction、梯度范数、吞吐、时长和 dev 记录。schema 不兼容只做可选诊断，不能由此重新过滤奖励。零优势组比例已有则沿用，不新增采样机制。

## 12. 执行顺序与两组短程实验

### 12.1 单测与真实 tokenizer

先完成计分、匹配、语法/schema 分离、mask 与 loss 测试，检查冻结文件和来源锁，展示少量真实 tokenizer 的字段定位结果。

本稿的关键分数与边权关系已用独立记录级算例核对；**这不代表项目代码、训练解析器、Qwen 映射或 GPU 路径已经通过测试。Codex 必须在项目中实际执行验收。**

### 12.2 真实模型 smoke

用现有 smoke 入口、真实模型和新奖励，完成少量真实 G=4 rollout、loss、backward 与 optimizer 调用。优先复用一次真实 rollout 的 IDs/trace 对照 off/on 的 loss，不做同前缀分支。

检查 Reference 冻结、原采样 logprob/ratio 口径、loss 与梯度有限、字段 mask 进入真实策略损失。若真实 smoke 未覆盖非零优势或某种符号，照实记录；单测可以覆盖符号，但不能伪造真实训练 A 或声称已覆盖。

不要用假模型通过测试后宣称真实 GPU smoke 完成。仅 mask 定位失败时不能把整个训练样本丢掉；按第 7 节回退并报告。

### 12.3 正式比较仅 A/B

| 实验 | 初始模型 | 训练奖励 | 策略更新 |
|---|---|---|---|
| A：新奖励基线 | 原 Action SFT best-adapter | R0-v2 | off，原全有效位置 |
| B：新奖励 + mask | **同一份** Action SFT best-adapter | **同一个** R0-v2 | field_v2 |

A/B 共用相同的训练奖励解析器、匹配器、四项权重、数据、超参数、seed、预算、dev 频率和选模规则。唯一算法差异是字段 mask。不能 A 用旧严格解析、B 用新训练解析。

在一张 4090 上串行运行。每组启动都重置到原 SFT checkpoint 和相同初始配置/RNG；不从 A 的结果接着训练 B，也不从 smoke 更新后的模型启动任一组。

使用两个新输出目录，不覆盖旧 run，不杀掉用户的进程，不并发挤占显存。若 GPU 被占用或资料缺失，完成可执行部分，给出准确阻塞和可运行脚本，不谎报完成。

### 12.4 预算固定，不自动扩展实验

先复用项目已有短程预算。若没有，则固定为 **2 个当前 dev 评测间隔，且不超过原一次 train 遍历预算**。步数计数、数据曝光和梯度累积口径从真实配置读取；明确 step 是 optimizer step、prompt group 还是其他单位。

旧文档的 dev 间隔有历史版本差异，因此不要照搬旧数值。A/B 开跑前锁定同一预算和间隔，不根据中途得分临时延长某一组。预算不足完整间隔时，按相同规则做结束 dev。

不为凑步数额外重复数据；不自动启动多 seed、随机 mask、HTMR 对照、权重搜索或旧奖励训练。若旧 R0 的可比结果已存在，可以一起展示；没有也不阻塞新 A/B。

同 seed 不保证策略更新后生成完全相同的答案。保证相同初始条件和数据曝光即可，不把在线 GRPO 改成固定离线 rollout 训练。

### 12.5 启动脚本要求

下列命令只展示新参数关系。Codex 应按真实 CLI 实现对应选项，读取本机已完成 SFT 的路径、配置和既有预算参数，生成能实际执行的脚本；不得臆造已有参数名或机器路径。

```bash
# SFT_INIT、OUTPUT_ROOT 和相同训练/预算参数从本地配置中落实。
python tasks/action-experiment/train_grpo.py \
  --init-adapter "$SFT_INIT" \
  --output "$OUTPUT_ROOT/A_r0_v2_mask_off" \
  --reward-version r0_v2 \
  --policy-update-mask off

python tasks/action-experiment/train_grpo.py \
  --init-adapter "$SFT_INIT" \
  --output "$OUTPUT_ROOT/B_r0_v2_field_mask" \
  --reward-version r0_v2 \
  --policy-update-mask field_v2
```

最终脚本必须包含明确的同一短程预算，不能因示意命令没写预算就默认执行长训练。检查输出目录冲突和 A 的退出状态；失败时不自动继续 B，不覆盖或静默 resume。

### 12.6 Dev、选模与结论

沿用原 dev `TC F1 -> TI F1` 选模；同分保留更早 checkpoint，step 0 的 SFT 起点可入选。保留初始 dev、固定间隔 dev 和末尾 dev。不把新奖励最高当成最优 checkpoint。

对同一个选中 checkpoint 报告原 TI/TC/AI/AC/EC 的 P/R/F1，保留 Recall、预测事件/论元数量等原统计。需要查看 dev 四项奖励时，可对已有 raw dev predictions 额外做 CPU 评分，不为此增加模型生成。

test 不用于调规则、改预算或选 checkpoint。短程首轮主要看 dev；既定最终 test 只在 dev 选定后独立执行。

A/B 直接检验的是 **mask 在新奖励下是否有帮助**；不能仅据此证明新奖励优于旧 R0。与原 SFT 结果比较可以判断这轮 RL 相对起点如何。一次短程提升不是稳定显著结论，best_step=0、无提升或下降都如实报告，不静默加机制追分。

## 13. 交付标准

交付以下内容：

1. 实际文件 diff 和简短修改说明，明确新训练解析/奖励与旧正式评测的隔离方式。
2. 测试命令与真实结果，关键 0.75/0.50/0.625 算例、匹配权重测试、真实 token mask 抽查和 GPU smoke 状态。
3. A/B 可执行脚本、真实初始化与输出路径、reward/mask 版本、预算、seed、训练配置和源码 hash。
4. 真实 A/B dev 对照：选中 step、同一 checkpoint 的原 TI/TC/AI/AC/EC P/R/F1、四项奖励诊断、保留率及回退率；没跑完的格子明确标记未完成。
5. 已完成与未完成项、失败原因、当前运行状态和下一条可执行命令。模型/数据/GPU 不可用时，不编造训练结果。

**完成实现不等于已证明有效。** 本轮以“定义正确落地、mask 确实进入 loss、A/B 条件一致、真实结果可比较”为完成标准。

## 14. 来源与适用范围

本任务书依据用户本轮最终确认的 R0-v2/Role 判分/四字段 mask 定义，以及旧 `codex_r0_structural_mask_v1.md` 中仍适用的训练、来源锁和评测边界编写。旧文件涉及严格 R0、两字段捆绑、schema 解析门控和相应测试的要求，以本文为准；本轮不要求重做其他阶段。

未核验服务器当前代码、模型路径、数据文件和运行进度。上述模块名、历史分支和旧超参数是定位线索；Codex 必须据本地真实状态落实，不能把文档计划当作已经完成。

---

**最终目标：用 Trigger/Argument 文本建立共享的一对一结构对应，四字段分别计分；Role 不经预测 Type 合法性门控；同一份命中结果生成字段 mask。只改训练奖励和策略更新位置，先准确实现，再跑新奖励下的 A/B。**
