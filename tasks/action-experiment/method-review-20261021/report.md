# 方法说明与四字段方案审阅

审阅对象：用户提供的 scientific_event_extraction_cot_sft_rl_overview.md 与 1015codex_r0_v2_four_field_mask.md。附件中的执行要求作为被审内容，不在本轮执行。检查范围为本地配置、来源、源码和已有统计；未使用 GPU、加载模型、修改附件或改变 B。

## Material Passport

- 类型：方法说明与实现一致性审阅。
- 状态：ANALYZED；没有重新运行完整训练，不作效果复现声明。
- 证据：同目录 evidence.json，两个独立只读审阅，以及下列项目文件。

## 结论

整体训练链条成立：Type-CoT SFT 的 final-adapter → Action SFT 的 best-adapter → GRPO。CoT 通过权重继承，不作为后续额外输入；未经过旧树 SFT 中转。Policy 和 Reference 初始化来自同一份 Action SFT 权重，Reference 冻结。

R0-v2、实例最优匹配、四字段标签和策略 mask 的核心实现与第二份文档一致。没有发现需立即变动 B 的定义落地错误；这不等于已经证明方法有效。

## 关键发现

### 1. 当前 A/B 在 800 步以后存在恢复方式混杂

第二份附件第 12.3 节要求唯一算法差异为字段 mask。实际 A 的 `continuation.json` 明确 source_step=800，optimizer_restored=false，rng_restored=false；其配置也标记 weights_only_warm_start。B 从 SFT 起点连续训练，没有这一重置。

因此，A 的 1200/1600 等后续权重包含 optimizer 和 RNG 重置的影响，不能把与 B 的差异完全归因于 mask。A 的 step800 是原训练真实权重，没有经过该后续重置；同 step800 对照可避开这一恢复混杂，但单 seed、在线采样和数值路径仍限制结论强度。B 继续运行，现有 A 也保留，不为审阅而中断或补跑。

证据：`runs/r0-v2-A800-recoverable-v5-42-20261019/continuation.json:2–7`；A/B configuration.json；`recovery_v5/train_grpo_recoverable.py:218–221,316–327`。

### 2. 总览的当前状态已经过时

第一份附件第 7.3、9、10 节仍将新奖励与细粒度 mask 写成待定方向。实际 A 使用 r0_v2 + off，B 使用 r0_v2 + field_v2；第二份附件才对应目前四字段方法。旧 R0、Structural Mask V1 是历史基线，不能用它们代替当前机制说明。

Action SFT 也已有自己的完整训练记录：Sorted1，实际 2807 step 早停、best1604。第一份总览关于“没有完整日志，不补写参数”的保留适合当时版本，当前可用真实记录更新。此处只指出差异，没有修改原文。

### 3. 四字段是分别计分，仍有结构依赖

Trigger 文本不匹配时，该事件的四类字段都不命中；Argument 文本不匹配时，Role 也不命中。新方法解除的是父 Type 对其他正确字段的直接门控、Role 对 Argument 的捆绑，不是四项完全独立。

单事件 Type 错、其他字段正确可得 0.75，符合定义；Trigger 错仍可能整事件得零。四字段 mask 复用这一命中结果，所以也继承其匹配依赖。这些限制已由第二份附件第 4.2 节写明，不是实现遗漏。

### 4. 奖励与最终指标没有相同的依赖关系

训练奖励在父 Type 错时允许正确 Argument/Role 得分；正式 presence 评测的 AI 签名包括 `(trigger,type,argument)`，AC 再包括 role。因此 Type 错仍使对应正式 AI/AC 不命中。训练奖励还进行一对一实例容量约束，正式 presence 匹配允许 counterpart 复用。

这是一种明确的代理训练目标设计，不代表程序算错；也意味着四项训练奖励上升不能直接推出 TI/TC/AI/AC/EC 提升。对外介绍宜说明 AI/AC 的父事件依赖，并注明 EC_I 或 EC_C，避免把奖励中的 Argument/Role F1 当成正式 AI/AC。

证据：`action_reward_v2.py:197–221`；`scripts/score_tree_presence.py:47–80`。

### 5. mask 的直接作用与有效性边界

`grpo_core.py:24–29` 确实仅在策略分子乘 field_mask，保留原有效 token 分母和 KL。实际 B 统计的回退率约 0.25%，正优势数据位置保留约 67%、负优势约 57%，说明不是全量回退导致 mask 失效。

它依然共享答案级优势，不产生字段级局部优势；混合边界 token 保留，共享参数与 KL 也会改变被屏蔽字段概率。保持原分母会减少策略梯度规模、提高 KL 的相对影响。第二份文档已明确承认这些近似，应保持这种表述；后续即使 B 改善，也不能仅凭 A/B 将原因唯一解释为信用分配更准确。

## 实验结论与后续边界

先让 B 按当前配置完成，不修改保护或增加模型。按既定 dev 规则选模，测试结果不反向用于调整奖励、预算或选择 checkpoint。A800/A1200 都已测试，若以后据它们改方案，则这些已见 test 结果应视为探索性证据，不再声称该轮为完全未接触的最终检验。

A/B 检验 mask 在 R0-v2 下的效果；新奖励相对旧 R0 的结论还需要可比的旧 R0 结果。现有单 seed 结果用于方向判断，不支持稳定提升或已解决信用分配的声明。A1775 数值一致性故障已单独诊断，不能据此认定奖励或 mask 公式错误。

## 验证复盘

核对附件原文、真实 A/B 配置和恢复记录、SFT 来源及采样审计、奖励/匹配/mask/loss/正式评测源码、B 实际 mask 汇总。两路独立审阅与主代理交叉核对一致。未启动训练或模型重放；本轮结论限于文档、实现和现有证据。
