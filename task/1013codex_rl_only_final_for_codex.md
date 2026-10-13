# Codex 执行任务：只更新 Action GRPO 奖励（保留现有 eval，不重训 SFT）

> 本文件是本轮**唯一执行入口**，代替旧 `codex_action_rl_token_only.md` 中关于“修改 `action_evaluation.py`”的要求，并吸收 `rl_only_review_addendum.md` 的必要修订。现有 Action SFT **已经完成**。本轮只接入 R0、验证 GRPO 正确性并运行 RL。不要重新设计任务或改动已验证的评测口径。

## 0. 不可变更的边界

- **不重跑 SFT / Type-CoT**；从已完成 Action SFT 的 dev 最佳 `best-adapter` 初始化 RL policy 和冻结 reference，二者初始均为相同完整 SFT LoRA。
- **v2 Prompt 原样复用**：始终读取完成 SFT 时保存的数据行 `row['prompt']` 和其 manifest / provenance；不得用后来上传的同名 prompt 覆盖。若 SFT 实际模板与仓库中 `prompts/action-v2-checked.txt` 不一致，以完成 SFT 的 provenance 为准。
- **评测代码原样保留**：不得修改 `scripts/score_tree_presence.py`、`tasks/action-experiment/action_evaluation.py` 的任何内容；保留 `parse_prediction()`、`to_scievents()`、`score_predictions()`、`evaluate_model()` 和当前 token 文本存在匹配 TI/TC/AI/AC/EC。继续按 dev **TC F1 优先、TI F1 次之**选模，同分保留较早 checkpoint，step 0 的 SFT 起点可入选。
- **不动来源锁**：不修改旧 SFT provenance、已有 manifest、Gold、已完成 checkpoint、`scripts/action_language.py`，也不关闭 `action_common.py` 的 SFT→RL 来源校验。
- 全部奖励与评测使用项目既有的**有序原文 token 文本**（Gold `tokens`、预测按当前 `to_scievents()` 方式切分），不是源 offset、字符位置或 Qwen BPE token ID。绝不在本轮引入 offset 对齐/位置回填。
- 保留当前五种文本 Action 和精简树，不增加动作、节点、Schema 子集过滤、CoT 输出或决策 token mask。

## 1. 奖励与评测分离：本轮最关键的改动

### 文件边界

| 路径 | 本轮操作 |
|---|---|
| `scripts/score_tree_presence.py` | 不修改：现有 token TI/TC/AI/AC/EC 评测 |
| `tasks/action-experiment/action_evaluation.py` | **不修改**：保留解析与 dev/test eval；旧 `reward_action()` 可以留在原文件中，但 RL 不再调用 |
| `tasks/action-experiment/action_rewards.py` | **新建**：实现 `reward_action_r0(raw_output, row, *, truncated=False, ...)` 以及实例匹配辅助函数 |
| `tasks/action-experiment/train_grpo.py` | 只修改必要接线、R0 日志和概率一致性诊断：`from action_evaluation import evaluate_model`；`from action_rewards import reward_action_r0 as reward_action` |
| `tasks/action-experiment/grpo_core.py` | 首轮保留现有 GRPO loss、clip、KL、advantage 计算与 EOS mask；不因换奖励而改优化目标 |
| `tasks/action-experiment/tests/test_action_rewards.py` | **新建** R0 单元与集成测试，不能通过修改旧 eval 测试为新奖励“消除失败” |

当前 `action_common.py` 在 SFT→RL handoff 时校验整个 `action_evaluation.py` 的 hash，**直接在该文件里替换旧奖励会触发 `SFT reference data/scoring inputs differ`**。使用独立奖励文件是为了保持历史来源锁正确，而不是绕过校验。

检查新建的 `action_rewards.py` 确实被 `source_files()` 收录到新的 RL run 的源码快照和 hash；不要伪造或改写原 SFT 的哈希。

## 2. 训练奖励 R0：完整 Event + Argument Link

模型仍生成 Action；奖励依据**严格解析后的事件结构**，不依据 Action 字符串相似度。

- Event：独立事件实例及 `(event_type, tuple(trigger_tokens))`。
- Argument Link：某一事件实例下的 `(role, tuple(argument_tokens))`。
- 允许重复事件、同一个 `(Type, Trigger)` 的多个实例、重复 Role、重复 Argument、跨事件共享同一源文本。**不做 set 去重，不用单值字典覆盖实例**。
- Event key **完全相同**才允许候选事件和 Gold 事件一对一匹配。
- 对同 key 的多个实例，在**最大事件配对数**前提下，求使内部严格 Link 匹配数最多的事件配对（矩形最大权指派；不能按列表序号贪心）。每对父事件内部的 Link 也一对一按多重集匹配，绝不能跨父事件借用论元。
- **事件列表/Role 输出顺序不构成实例 ID**。仅交换两个完整同 key 事件的顺序应不影响 R0。真正合并、拆分、错挂论元才应失分。
- 预测/Gold 未匹配的 Event 与 Link 都保留在分母；不额外堆叠 FP、FN、角色覆盖或长度惩罚。

```text
F_E = 2 * matched_event_instances / (pred_event_count + gold_event_count)
F_L = 2 * matched_argument_links / (pred_link_count + gold_link_count)
R0  = 0.5 * F_E + 0.5 * F_L
```

边界：
- 合法 `STOP` 且两侧均无事件：`R0=1`；仅一侧无事件：`R0=0`。
- 双方都无 Link 时 `F_L=F_E`；仅一侧有 Link 时 `F_L=0`。
- 解析不合法（缺 `END`/`STOP`、非法 Role/Type、STOP 后额外动作等）：`R0=0`，不自动修复、不能假装是合法空输出。
- **Action `STOP` 与模型 EOS 不同**：动作已经合法 STOP，但生成预算内未出 EOS，可单独标记 `has_eos=false`、`truncated=true`，仍按完整动作评分；无 STOP 到达长度上限视为非法并计 0。不要另加 EOS/STOP 奖励。

`reward_action_r0` 至少返回 `reward`, `event_exact_f1`, `link_exact_f1`, matched/pred/gold Event 与 Link 计数, `parse_valid`, `parse_error`, `predicted_events`, `truncated` 等，并以 `reward_version=event_link_token_exact_v1` 标注训练运行。新分项**不能冒名 TC/AC**。

**dev/test 继续使用原始 presence eval**；R0 是独立训练奖励，不能据此更改或重算现有 TI/TC/AI/AC/EC 的定义。

## 3. GRPO 保持当前算法，只增加必要验证与日志

首轮参数保持当前实现：`G=4`、`learning_rate=1e-5`、`clip=0.2`、`KL beta=0.01`、`temperature=1`、`top_p=1`、`top_k=0`、`max_new_tokens=768`，train seed 42 打乱走一次，保留 None，使用旧版组内样本标准差归一化；不要把 TRL 的 `scale_rewards=False` 作为一个不存在的当前代码参数。首轮不加入 H/R1、软匹配、token mask 或奖励模型。

- 保留原始采样 token、generation scores 和 old behavior logprob；不要重新格式化/重分词后计算 loss。
- Policy/reference 来自同一完整 Action SFT LoRA；reference 始终冻结；保留 LoRA dropout=0、首个 EOS 在 loss 中、后续 padding 被 mask。
- 记录 R0 总奖励及 Event/Link 分项的均值、组内方差、有效组比例；保留重复实例的规范化结构差异数量。**不新增“零奖励/同分组跳过训练”规则**。
- 原历史 GPU smoke 曾报告 `sampling_vs_teacher_max_difference≈11.338`，但不能直接视为本版本 bug 或本版本已验证。用当前真实模型、768 上限、R0 的短程 smoke 重测：
  1. cached 采样 vs no-cache eval；
  2. eval vs train/checkpointing；
  3. 实际**反向传播路径**的 `current_update_logprob` vs 对应采样时 `old_behavior_logprob`；
  4. 最初 policy vs reference、reference hash 不变。
- 记录有效 completion token 的 logprob 差异均值、P95、P99、最大值，以及实际 `ratio=exp(current-old)` 落在 `[0.8,1.2]` 外的比例。当前每组样本仅 backward 一次、参数在积累后才更新；相同参数和一致计算分布下，更新前 ratio 应接近 1。**若异常重复出现，定位 cache/mask/attention/精度/模式，不要只凭 loss 有限就放行正式 RL。**
- 不因本轮换奖励同时强制 batch generation、改 loss 归一化、改 advantage、改变采样分布或调 checkpoint 间隔；工程提速放到独立实验。

## 4. 必须通过的测试与回归

1. `Gold == prediction` 的复杂情况 R0=1：共享 Trigger 的不同 Type、同 `(Type,Trigger)` 的不同实例/论元、重复 Role、跨事件复用文本。
2. 删除全部论元、Role 错、增加错误论元、复制论元均会降低 Link 分；重复一个完整事件会因额外实例失分。
3. 真正合并/拆分/错挂父事件论元会失分；只交换完整事件的输出顺序**不应失分**。
4. 合法 `STOP`、非法 Action、缺 END/STOP、STOP 后额外内容、已有 STOP 但未出 EOS 分开验证。
5. 不变的 raw predictions 输入原 eval，修改前后的 token presence TI/TC/AI/AC/EC **完全一致**。
6. SFT→RL provenance / hash 验证**保持开启并通过**；RL 运行源码快照包含新奖励文件；policy/reference 初始化正确。
7. 当前运行环境 GPU 短程 G=4/R0 smoke：前向、反向、更新、梯度范数、old/current/reference 概率检查通过；真实奖励字段进入 rollout/group 日志。

## 5. 选模、运行与交付

- 沿用 dev `TC_f1 → TI_f1` 选模，step 0 的 SFT 参加对照；完全同分选择较早 checkpoint。**不要改成 R0 选模**。
- 沿用旧版 dev 频率，报告原 TI/TC/AI/AC/EC P/R/F1，同时将 R0/Event/Link 作为额外诊断字段；不修改 test 分数或旧评测器。
- test 只在 dev 选出最终 RL checkpoint 后运行。若 best_step=0，如实报告 RL 没有改善，不强行选一个更新后 checkpoint。
- 先完成代码修改、R0 单测、旧 eval 回归、provenance 验证和当前 GPU smoke。验证通过且资源允许时，使用新输出目录启动**正式 RL-only**；保留完整日志与运行配置。若真实概率路径复测异常，则定位并给出可复现证据，不要静默忽略。
- 交付：代码 diff、测试结果、GPU smoke 统计、正式运行命令、source hashes、run config、raw rollout、dev 历史和最终评测。明确标注**未改动**的 SFT、v2 Prompt、原评测代码。

```bash
python tasks/action-experiment/train_grpo.py \
  --init-adapter <已完成Action-SFT运行目录>/best-adapter \
  --output <新的RL输出目录>
```

**本轮目标：不触碰现有 eval 的情况下，让 GRPO 的训练奖励同时评价独立 Event 实例及其 Argument–Role 归属。**
