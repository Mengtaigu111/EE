# Seed 42：CoT → Action SFT → R0 GRPO 实验记录

记录日期：2026-10-15。本文依据实际运行配置、保存的监督数据、checkpoint 和完整评测 JSON 归档；所有结果表均为 **F1 × 100**，列顺序统一为 **TI、TC、AI、AC、EC**。

## 1. 本轮实验与当前结论

实际模型链路：

```text
Qwen3-8B + fresh LoRA
  → 类型 CoT 监督训练：seed42，1轮，802步，固定使用最终adapter
  → Action SFT：继承CoT adapter，首轮Sorted1、后续打乱
      实际2807步早停，dev选中step1604
  → R0 GRPO：继承Action SFT step1604，冻结同一SFT作为reference
      计划3207步，实际完成2195次更新后概率保护停机
      已完成dev：0 / 400 / 800 / 1200 / 1600 / 2000
      当前按dev选中step2000
```

SFT 已正常完成早停流程。RL 尚未完成计划的一轮，保持停止；没有 step2400 及之后的完整 dev。SFT、RL step2000、1600、800 的完整 test 均已完成。后两者是用户指定的补充测试，**不会据此覆盖原有 dev 选模结果**。

当前补测中，RL step800 的 test 五项 F1 为 **77.27 / 64.80 / 46.68 / 44.34 / 49.53**；按 dev 正式选出的 RL step2000 为 **75.88 / 64.17 / 45.28 / 42.96 / 48.34**。应同时报告，不能只把补测最好的 checkpoint 作为事先由 dev 选中的模型。

## 2. 数据、输出与评测口径

### 2.1 数据规模与输入

| split | 句级样本 | 事件 | 论元 | 无事件样本 |
|---|---:|---:|---:|---:|
| train | 12828 | 19488 | 45076 | 2124 |
| dev | 1593 | 2445 | 5618 | 271 |
| test | 1627 | 2448 | 5717 | 272 |

Action 使用修正后的 [GoldOrder / AppendixNoCopy 源数据](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/data/sentence-c-appendix-no-copy-gold-order-20261008)；冻结训练数据在 [action-v2-goldorder-20261012](/home/myluo/EE/EXCEEDS/tasks/action-experiment/data/action-v2-goldorder-20261012)。dev 有251篇文档，test有250篇文档。CoT 使用较早的 sentence-c 数据版本，不能声称各阶段事件结构毫无版本差异。

Action 模型输入是完整摘要、目标句及 schema；gold Action 只作 SFT 的 assistant target 或 RL 奖励依据。SFT 和当前 RL 复用同一份冻结 v2 prompt：[prompt_template.txt](/home/myluo/EE/EXCEEDS/tasks/action-experiment/data/action-v2-goldorder-20261012/prompt_template.txt)，与 [action-v2-checked.txt](/home/myluo/EE/EXCEEDS/tasks/action-experiment/prompts/action-v2-checked.txt) 一致。实际每条数据中的 `prompt` 是直接使用的输入，不能仅依据后来编辑的 task 草稿判断实际训练 prompt。

### 2.2 Action 与简化树

Action 数据由 gold 事件确定性转换，不是让 teacher 再生成一遍 CoT。语法为 `NEW(TYPE)`、`TRG("text")`、`ARG(Role, "text")`、`END`、`STOP`；无事件仅输出 `STOP`。解析后的简化树为：

```text
(ROOT (TYPE "触发词" (Role "论元")))
```

真实训练数据第1条，ID `febd573ada9568c635f6d8aeada27ec5::sentence-0`：

```text
program induction for answering complex questions over knowledge bases ( kbs ) aims to decompose a question into a multi - step program , whose execution against the kb produces the final answer .
```

对应监督 Action 和树：

```text
NEW(ITT)
TRG("decompose")
ARG(Target, "program induction")
END
STOP

(ROOT (ITT "decompose" (Target "program induction")))
```

构造代码：[build_action_dataset.py](/home/myluo/EE/EXCEEDS/scripts/build_action_dataset.py)、[action_language.py](/home/myluo/EE/EXCEEDS/scripts/action_language.py)。事件按触发位置稳定排序，角色按 prompt 的固定顺序，同角色论元按位置稳定排序；保留重复实例及片段内部 token 顺序。训练前已核查全量16048条 Gold → Action → Tree、一致性及 offset / 长度；审计在 [audit-v2-20261012](/home/myluo/EE/EXCEEDS/tasks/action-experiment/audit-v2-20261012) 和 [token-audit.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/data/action-v2-goldorder-20261012/token-audit.json)。gold Action 最长671 tokens，生成上限768覆盖 gold，但不能保证模型自由生成永不超长。

纯文本 Action 无法唯一恢复重复文本的 offset。因此保留位置检查为诊断，评测按文本进行；不据对齐诊断删除预测。

### 2.3 统一 dev / test 指标

仍使用原 [action_evaluation.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/action_evaluation.py) 调用 [score_tree_presence.py](/home/myluo/EE/EXCEEDS/scripts/score_tree_presence.py)，本轮 R0 修改没有改这两份评测代码。

- TI：触发文本；TC：触发文本＋事件类型。
- AI：父事件的触发文本及类型＋论元文本；AC：再要求论元角色相同。
- EC：AC 中论元文本也作为事件触发文本出现的链接子集；展示的是底层含角色字段 `EC_C_f1`，统一简称 EC。

这里的 token 是数据文本按空白分出的词序列，要求序列完全相同；不是 Qwen 的 BPE token，也不是官方有重复触发问题的 offset/token 实现。指标为全文档累计的文本存在匹配：保留实例计数，允许对侧匹配对象复用，预测侧与 gold 侧分别计命中后计算 P/R/F1。因此它与 R0 的一对一实例匹配不是同一套指标。解析错误按显式未匹配预测处理，不伪装成正确的无事件结果。

## 3. CoT：先学习事件类型解释

### 3.1 teacher 监督数据如何生成

离线生成 prompt 是 [task/prompt/3.10-1.md](/home/myluo/EE/EXCEEDS/task/prompt/3.10-1.md)。teacher 输入包含摘要、目标句、gold 类型计数、逐事件 gold 论元与角色、filled template 及官方 schema，生成事件族解释和逐事件语义/角色解释。分片 [manifest.json](/home/myluo/EE/EXCEEDS/task/prompt/runs/prompt-3.10-1-train-full-part1of2-20260925/manifest.json) 记录 teacher 为 `gpt-5.5`。

保存的解释合并在 [prompt-3.10-1-train-full-merged-20260926.jsonl](/home/myluo/EE/EXCEEDS/task/prompt/runs/prompt-3.10-1-train-full-merged-20260926.jsonl)。[summary](/home/myluo/EE/EXCEEDS/task/prompt/runs/prompt-3.10-1-train-full-merged-20260926-summary.json) 记录12828条、missing/unresolved/qc_bad均为0，同时有3723条 format warning。该检查不能代替全量人工语义审核。

学生训练输入使用另一份 [CoT prompt_template.txt](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/prompts/user-final-20261005/prompt_template.txt)：仅给摘要、目标句、10类定义及输出要求，不给 gold inventory 或 gold 论元。监督 target 是保存的 teacher `Reasoning` 加代码追加的 gold `Final Answer / Event Type Inventory`。类别为 PUR、ITT、RWS、RWF、PRP、WKS、MDS、FIN、CMP、FAC。

数据构造与 prompt 替换分别由 [build_dataset.py](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/scripts/build_dataset.py)、[rewrite_prompt_dataset.py](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/scripts/rewrite_prompt_dataset.py) 完成；最终数据见 [type-cot-recognition-user-final-20261005](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/data/type-cot-recognition-user-final-20261005)。

### 3.2 实际保存的 CoT 监督例子

下例来自最终 CoT `train.jsonl` 第5行，ID `febd573ada9568c635f6d8aeada27ec5::sentence-4`，是保存的训练解释文本：

```text
Sentence:
for program transfer , we design a novel two - stage parsing framework with an efficient ontology - guided pruning strategy .

Reasoning:

1. Event Family Reasoning:
   The sentence belongs to the Method family because it introduces the framework developed for the study. It presents the proposed methodological contribution and its research objective.

2. Event Reasoning:
   Event 1: “for program transfer, we design a novel two-stage parsing framework” involves we (Proposer), program transfer (Target), and two-stage parsing framework (Content). The authors introduce the framework as their proposed means for achieving program transfer, so these participants instantiate the Appendix F.2 Propose structure: a proposer puts forward content for a target; unlike MDS, the clause introduces the overall framework rather than describing an internal method step. Type: <PRP></type>

Final Answer:
Event Type Inventory:
PRP: 1
```

### 3.3 CoT 训练配置与完成情况

基座 `/opt/models/Qwen3-8B`，fresh LoRA，seed42。LoRA r16、alpha32、dropout0.05，目标模块为 q/k/v/o_proj、gate/up/down_proj。BF16、gradient checkpointing、关闭 Qwen 隐式 thinking；只在 assistant token 上算 CE。

**CoT 本身不是 Sorted1**：12828条按 seed42 全量随机打乱，一轮802个 optimizer steps，每条恰好一次。microbatch1、累积16、lr1e-4、cosine、warmup25、最大总长4096；训练最长2163 tokens，无截断。

已完成802步，train loss `0.6360274723`，耗时约2小时31分。固定导出最终 adapter，不做 CoT dev 选模；本次省略了独立 CoT dev 诊断，因此没有独立的 CoT dev/test 数字。该旧 run 中另一个 `tree/` 阶段的结果属于其他直接树生成实验，本文不混用。

证据：[cot/config.json](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot/config.json)、[cot/train_metrics.json](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot/train_metrics.json)、[cot-verification.json](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot-verification.json)。后续初始化权重为 [cot/final-adapter](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot/final-adapter)。

## 4. Action SFT：从类型解释转为完整事件 Action

### 4.1 训练与课程顺序

[train_sft.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/train_sft.py) 加载上述 CoT final adapter，沿用 LoRA 并新建 optimizer/scheduler。最大4轮，每轮约802个更新，最多3208步。首轮按既定 Sorted1 课程顺序，后续各轮按 seed 加 epoch 打乱。实际顺序记录在 [observed-order.jsonl](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/observed-order.jsonl)。

microbatch1、累积16、初始lr1e-4、warmup97，最大总长4096，dev batch4、最多生成768 tokens。assistant-only CE 包含 EOS，按实际累积组中 assistant token 总量归一化；尾组按实际大小处理。LoRA r16/alpha32/dropout0.05。

### 4.2 完整 dev、减半与早停规则

第401步即评完整1593条 dev，之后按预定 checkpoint 调度。选模先比 TC，再比 TI，完全相同保留更早 checkpoint。

另设 plateau anchor：TC 相对 anchor 提升至少 `0.001`（即0.1个百分点）才清零 bad_count；未达到则累计。bad_count 达3时学习率减半，达6时早停；**减半不清零 bad_count**。这与保存最佳模型的 TC/TI 排序是两个不同逻辑。

本轮 anchor 在1604达到最佳，1804/2005/2205连续3次未改善，于2205将lr从1e-4减到5e-5；2406/2606/2807仍未达到有效提升，2807累计6次早停。完成约3.50轮，用时约13小时39分，train loss `0.1088234191`，最终使用 step1604 的 best-adapter，而非2807的 last-adapter。

### 4.3 全部 Action SFT dev 结果

| optimizer step | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| 401 | 61.30 | 44.45 | 21.49 | 20.60 | 9.40 |
| 802 | 70.79 | 54.49 | 37.02 | 34.41 | 37.84 |
| 1203 | 75.10 | 61.36 | 43.45 | 40.84 | 45.82 |
| 1403 | 75.29 | 61.66 | 43.23 | 40.41 | 46.73 |
| 1604（dev 最佳） | 75.89 | 62.89 | 44.85 | 42.43 | 48.93 |
| 1804 | 75.61 | 62.62 | 44.23 | 41.53 | 47.66 |
| 2005 | 75.07 | 62.12 | 43.86 | 41.59 | 47.59 |
| 2205 | 73.42 | 60.83 | 43.05 | 40.81 | 45.35 |
| 2406 | 75.78 | 62.05 | 43.34 | 41.08 | 48.77 |
| 2606 | 74.92 | 61.89 | 43.38 | 40.82 | 47.22 |
| 2807 | 74.24 | 61.94 | 42.98 | 40.55 | 47.15 |

数值来自 [online-dev](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/online-dev) 下各 `dev-checkpoint-*/metrics.json`。选模证据：[best.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/best.json)；截止与反馈：[train_metrics.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/train_metrics.json)、[feedback.jsonl](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/feedback.jsonl)。3007/3208因早停未执行，不补造结果。

## 5. R0 RL：奖励与 GRPO 的具体实现

### 5.1 本次使用哪个版本

本文 RL 主结果来自 `action-grpo-r0-window-dev400-42-20261014`。此前的 `0.6TC+0.3TI+0.1Count` baseline、单独 `rl_v2` batch generation 实验，以及旧短程概率保护运行均不是这份主结果。

实现为自定义 PyTorch/PEFT GRPO：[train_grpo.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/train_grpo.py)、[grpo_core.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/grpo_core.py)。本轮只做 R0，没有加入 H、R1、奖励模型或合法 Action mask。

### 5.2 独立 R0 奖励

[action_rewards.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/action_rewards.py) 实现版本 `event_link_token_exact_v1`：

```text
R0 = 0.5 × Event_exact_F1 + 0.5 × Link_exact_F1
Event_exact_F1 = 2 × 匹配事件实例数 / (预测事件数 + gold事件数)
Link_exact_F1  = 2 × 匹配论元链接数 / (预测链接数 + gold链接数)
```

事件 key 是 `(类型, 有序触发词tokens)`。同 key 内保留重复事件实例，构造事件两两对应的论元链接重叠权重；用 SciPy `linear_sum_assignment(maximize=True)`，在最大可匹配事件数下最大化论元链接命中。一个实例最多匹配一次，不跨父事件借用论元。每个父事件内部的 link 为 `(角色, 有序论元tokens)` 多重集合，重复论元按实际次数匹配，未匹配父事件的论元仍计入分母。

严格解析失败奖励0；两边都无事件时奖励1；一边有事件另一边无事件时为0。两边都无论元时 Link F1 回退为 Event F1，只有一边有论元则 Link F1 为0。长度达上限不直接把奖励强制置零，仍按实际输出是否可解析及其结构计分，另记截断诊断。R0 明确含论元项，不能只输出类型、触发和数量就普遍拿满分。

### 5.3 采样、优势与优化目标

初始化 policy 和 reference 时，都加载 **Action SFT best step1604 的完整 LoRA**，共享冻结基座；reference LoRA 冻结，仅 policy LoRA 更新。reference 不是禁用 adapter 后的裸 Qwen 基座。RL dropout设为0。

每条 prompt 在同一参数版本下采样 G=4 个回答，**当前正式版本仍是串行生成4次**。temperature1、top_p1、top_k0，生成上限768、总长4096；保留实际生成 token IDs 和真实 sampling scores。old logprob 由实际行为分布计算，不重新生成或重新 tokenize 回答。

```text
A_i = (R_i - mean(R_group)) / (sample_std(R_group) + 1e-4)
ratio_t = exp(logπ_current,t - logπ_behavior,t)
δ_t = logπ_reference,t - logπ_current,t
KL_t = exp(δ_t) - δ_t - 1
loss_i = mean_valid_tokens[
    0.01 × KL_t - min(ratio_t × A_i, clip(ratio_t, 0.8, 1.2) × A_i)
]
```

组内4个回答等权平均，累积4组 prompt 再更新一次，即每次更新16个回答。有效 token mask 包含首个 EOS，排除其后 padding；current/reference 使用 no-cache teacher 前向，BF16、SDPA、gradient checkpointing。组内奖励全部相同时优势为0，仍保留 KL 路径，不因无奖励方差跳过整组。

优化参数：AdamW lr1e-5、betas(0.9,0.999)、eps1e-8、weight_decay0、max_grad_norm1、恒定学习率、num_iterations1。seed42 将12828条训练数据（含None）打乱一轮，共计划3207次 optimizer updates。实际完整 dev 参数是 `--dev-steps 400`，不是 CLI 默认的200；先评 step0，然后每400步评完整 dev，正常完成时再做终点评估。

### 5.4 概率检查、验证与实际停机

旧短回答中1/57个 token 超出优化 clip 区间即可触发1%阈值停机，之后改为跨回答汇总诊断；不改 R0、不改 prompt、不改评测器，也不同时引入 batch generation。

[grpo_diagnostics.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/grpo_diagnostics.py) 区分优化 clip `[0.8,1.2]` 与独立保护范围 `[0.5,2.0]`。至少16个回答且512个有效 tokens 组成一个不重叠窗口；P99绝对logprob差>0.15、优化clip外比例>1%、独立保护范围外比例>1%，任一项触发窗口告警，连续3个告警窗口硬停，正常窗口清零。非有限值或任一有效 token 的绝对logprob差>1.0立即硬停。

训练前已完成 R0 单元测试、原评测器回归和短程 GPU 概率检查。R0专项27项测试、80组暴力匹配对照，全量16048条 Gold 自比奖励均为1；后续主线119项CPU测试通过。64个回答/4次更新的 GPU smoke 包含3675有效tokens，current-old差P99约0.092955，clip外2/3675约0.05442%，reference保持不变且policy实际更新。诊断指出固定参数下 BF16 cached/no-cache 路径可能存在数值差异；FP32输出层仅用于诊断，正式训练仍保持BF16/SDPA，不能声称已完全消除概率差异。

验证报告：[verification-r0-20261013/report.md](/home/myluo/EE/EXCEEDS/tasks/action-experiment/verification-r0-20261013/report.md)、[verification-probability-20261014/report.md](/home/myluo/EE/EXCEEDS/tasks/action-experiment/verification-probability-20261014/report.md)、[GPU smoke result.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/probability-window-smoke-20261014/result.json)。

当前正式运行已完成 **2195次更新**，下一组采样诊断出现14有效tokens中最大logprob差 `1.0138816833496094`，触发严重偏差硬停；这不是SFT式patience早停，也不能记为完整一轮完成。实际异常和退出见 [rl.log](/home/myluo/EE/EXCEEDS/tasks/action-experiment/launch-probability-20261014/rl.log)。现有 adapter checkpoint 不含可精确恢复的完整 optimizer/RNG 状态，不能把重新载入权重说成精确断点续训。

### 5.5 全部当前 R0 RL dev 结果

| optimizer step | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| 0 | 75.89 | 62.89 | 44.85 | 42.43 | 48.93 |
| 400 | 75.91 | 62.93 | 44.94 | 42.58 | 49.40 |
| 800 | 76.51 | 63.32 | 45.48 | 43.35 | 51.42 |
| 1200 | 75.82 | 61.99 | 44.58 | 42.13 | 48.59 |
| 1600 | 76.52 | 63.66 | 45.20 | 43.11 | 50.12 |
| 2000（dev 最佳） | 75.96 | 64.29 | 45.56 | 43.38 | 49.95 |

记录：[dev-history.jsonl](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/dev-history.jsonl)。选模仍按 TC → TI → 更早 checkpoint；因此选 step2000，尽管 EC 单项最高在800。step2000相对初始化step0，dev TC提高约 **1.40个百分点**；五项指标不随步数单调增长。2195之后停机，未执行2400等后续dev。

## 6. 完整 test 结果

全部为完整1627条、250篇文档，使用同一冻结 v2 prompt、768生成上限和同一文本存在评测器。

| 模型 / checkpoint | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| Action SFT step1604（dev 选模） | 76.81 | 63.15 | 45.18 | 42.74 | 48.45 |
| R0 RL step2000（dev 选模） | 75.88 | 64.17 | 45.28 | 42.96 | 48.34 |
| R0 RL step1600（用户指定补测） | 76.88 | 64.54 | 46.30 | 44.07 | 49.14 |
| R0 RL step800（用户指定补测） | 77.27 | 64.80 | 46.68 | 44.34 | 49.53 |

RL800相对Action SFT的 test 提升（TI / TC / AI / AC / EC，百分点）：**+0.46 | +1.64 | +1.50 | +1.60 | +1.08**。但800是在查看多个 checkpoint test 后得到的补测结果；后续论文应明确说明这一选择过程，或另行预定选模标准进行独立实验。当前 dev-selected RL2000 的 test并非所有指标都优于SFT。

### 6.1 输出诊断

| 模型 | test 条数 | 截断 | 解析错误 | 论元文本顺序对齐失败 |
|---|---:|---:|---:|---:|
| Action SFT step1604（dev 选模） | 1627 | 1 | 1 | 54 |
| R0 RL step2000（dev 选模） | 1627 | 0 | 0 | 36 |
| R0 RL step1600（用户指定补测） | 1627 | 0 | 2 | 55 |
| R0 RL step800（用户指定补测） | 1627 | 0 | 1 | 33 |

“论元文本顺序对齐失败”表示预测论元不能按诊断器规定的顺序在原文中对齐，可能含重复文本或片段顺序问题；它是诊断记录，不据此删预测，也不等同于该数量的解析失败。正常退出码0只表示评测完成，不能据此断言每条输出都可解析。SFT的1条截断与1条解析错误是同一条异常输出，具体样本见 [generation-diagnostics.md](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/test-best-20261013/generation-diagnostics.md)。

### 6.2 与用户提供参考结果的对照

用户给出的参考中心值及波动为：TI `75.29±0.32`、TC `63.74±0.14`、AI `44.97±0.28`、AC `43.20±0.29`、EC `48.25±0.10`。RL800相对这些中心值分别为 **+1.98 / +1.06 / +1.71 / +1.14 / +1.28** 个百分点。

这些参考值的论文来源、±的定义及是否使用完全相同评测器未在本文核实；本轮只有seed42，不能据此报告多seed均值/标准差或统计显著性，也不能直接视作已确认同口径SOTA结论。

## 7. 关键结果、权重与源码位置

### 7.1 权重与运行依据

| 内容 | 位置 |
|---|---|
| CoT最终LoRA | [cot/final-adapter](/home/myluo/EE/EXCEEDS/tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot/final-adapter) |
| Action SFT最佳LoRA：step1604 | [SFT/best-adapter](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/best-adapter) |
| Action SFT配置与来源锁 | [config.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/config.json)、[provenance.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/provenance.json) |
| 当前R0运行目录 | [action-grpo-r0-window-dev400-42-20261014](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014) |
| R0 dev选中LoRA：step2000 | [RL/best-adapter](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/best-adapter) |
| R0补测LoRA：step1600 / step800 | [checkpoints/step-1600](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/checkpoints/step-1600)、[checkpoints/step-800](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/checkpoints/step-800) |
| R0实际配置与来源锁 | [configuration.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/configuration.json)、[provenance.json](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/provenance.json) |
| R0启动脚本 | [run.sh](/home/myluo/EE/EXCEEDS/tasks/action-experiment/launch-probability-20261014/run.sh) |

来源锁覆盖基座配置、5个权重分片、tokenizer、数据、prompt、初始化adapter及运行源码。训练过程真实代码以每个run的 `source_snapshot/` 与 `provenance.json` 为准；工作区后来有变更时，不以当前 HEAD 冒充历史运行版本。

### 7.2 test 原始文件

| 模型 | 完整结果目录 |
|---|---|
| SFT1604 | [test-best-20261013](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/test-best-20261013) |
| RL2000，dev-selected | [test-best-stopped-20261015](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/test-best-stopped-20261015) |
| RL1600，补测 | [test-step1600-stopped-20261015](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/test-step1600-stopped-20261015) |
| RL800，补测 | [test-step800-stopped-20261015](/home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/test-step800-stopped-20261015) |

这些目录中的 `metrics.json` 保存完整P/R/F1、计数和诊断；`predictions.jsonl` 保存原始模型输出，`selection-source.json` 记录选用checkpoint及中断训练后的阶段性评估来源。不要使用名称带 `interrupted-*` 的旧部分输出充当完整test。

### 7.3 实现入口

| 功能 | 文件 |
|---|---|
| Action数据与语法 | [build_action_dataset.py](/home/myluo/EE/EXCEEDS/scripts/build_action_dataset.py)、[action_language.py](/home/myluo/EE/EXCEEDS/scripts/action_language.py) |
| SFT | [train_sft.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/train_sft.py) |
| RL训练循环 | [train_grpo.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/train_grpo.py) |
| R0奖励 | [action_rewards.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/action_rewards.py) |
| GRPO优势、KL与loss | [grpo_core.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/grpo_core.py) |
| 逐token诊断与窗口保护 | [grpo_diagnostics.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/grpo_diagnostics.py) |
| 原评测器 | [action_evaluation.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/action_evaluation.py)、[score_tree_presence.py](/home/myluo/EE/EXCEEDS/scripts/score_tree_presence.py) |
| 中断RL按dev最佳做test | [evaluate_stopped_run.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/verification-stopped-test-20261015/evaluate_stopped_run.py) |
| 指定checkpoint补测 | [evaluate_checkpoint_when_free.py](/home/myluo/EE/EXCEEDS/tasks/action-experiment/verification-stopped-test-20261015/evaluate_checkpoint_when_free.py) |

运行环境使用 Conda `qwen3meld-run`，Python位于 `/home/myluo/miniconda3/envs/qwen3meld-run/bin/python`；单张RTX4090，BF16。长实验使用tmux；后续如需等待空卡应按用户要求空闲即启动，无连续180秒空闲等待。本次归档只读取既有训练/评测结果，没有重启训练或改变checkpoint、奖励、prompt、评测器和来源校验。
