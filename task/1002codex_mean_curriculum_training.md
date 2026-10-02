# Codex 执行任务：两阶段 Curriculum Training（含 ROOT/no-event 样本）

## 目标

在现有 Experiment C 训练框架上，实现并运行一版两阶段 cumulative curriculum learning。

当前课程难度已经固定，不再调整 difficulty formula。

本任务需要：

1. 基于既有 train difficulty assignment 构造 Stage 1 / Stage 2 训练数据；
2. 正确处理 ROOT / no-event 样本；
3. 保持课程训练总 optimizer steps 与 baseline 对齐；
4. 从 Stage 1 checkpoint 连续进入 Stage 2，不重新初始化；
5. 保存完整训练配置、数据划分、日志和最终 dev 评估结果；
6. 不修改 prompt、output format、event schema、loss 或模型结构。

---

# 1. 项目路径

项目根目录：

```text
/home/myluo/EE/EXCEEDS
```

Stage-2 / Experiment C 相关目录：

```text
tasks/stage2-event-tree/
```

已有 difficulty 分析目录：

```text
tasks/stage2-event-tree/difficulty-analysis/
```

优先复用已有文件：

```text
mean_percentile_difficulty_sentence_level.csv
mean_percentile_difficulty_assignments.csv
mean_percentile_difficulty_summary.json
1002codex_mean_percentile_curriculum_rationale.md
```

如果文件名略有差异，先在：

```text
tasks/stage2-event-tree/difficulty-analysis/
```

中查找对应 mean-percentile 结果。

不要重新设计 difficulty。

---

# 2. 当前固定的 Difficulty Definition

只对 positive sentence：

```text
N_e(s) >= 1
```

计算 difficulty。

事件级角色覆盖：

```text
C(e)
=
num_observed_legal_roles / num_legal_roles_for_event_type
```

事件级 type 内 percentile：

```text
P(e)
=
within-type tie-aware midrank percentile
```

句子级平均 percentile：

```text
P_mean(s)
=
mean(P(e) for e in sentence)
```

schema incompleteness：

```text
D_schema(s)
=
1 - P_mean(s)
```

normalized event multiplicity：

```text
D_event(s)
=
1 - 1 / N_e(s)
```

最终 difficulty：

```text
D(s)
=
0.5 * (D_event(s) + D_schema(s))
```

即：

```text
D(s)
=
0.5 *
[
(1 - 1/N_e(s))
+
(1 - P_mean(s))
]
```

lower = easier

higher = harder

本任务不要修改该公式。

---

# 3. 当前固定的 B1 / B2

使用 train positive difficulty 的中位数进行划分。

当前结果：

```text
positive train sentences = 10,704
ROOT/no-event sentences = 2,124
total train sentences = 12,828
```

mean-percentile difficulty median：

```text
tau = 0.4275338928340865
```

规则：

```text
B1:
D(s) <= tau

B2:
D(s) > tau
```

边界 ties 全部进入 B1。

当前：

```text
B1 = 5,427
B2 = 5,277
```

不要重新按 sample_id 拆 boundary ties。

不要重新计算新的 threshold。

优先直接复用已有 assignment CSV。

---

# 4. ROOT / No-event 样本

定义：

```text
ROOT:
gold_event_count == 0
```

ROOT 不计算 difficulty，不属于 B1/B2。

当前：

```text
ROOT total = 2,124
```

完整 train 中 ROOT 占比：

```text
2124 / 12828
≈ 16.56%
```

positive : ROOT 比例为：

```text
10704 : 2124
```

Stage 1 中需要保持与 full train 接近的 ROOT 比例。

---

# 5. Stage 1 ROOT 数量

Stage 1 positive：

```text
B1 = 5,427
```

按照 full train 的 positive:ROOT 比例，Stage 1 ROOT 数量计算：

```text
ROOT_stage1
=
round(
5427 * 2124 / 10704
)
```

请在脚本中实际计算，不要手工硬编码近似值。

预期约：

```text
~1077 ROOT samples
```

最终报告实际整数结果。

Stage 1 数据为：

```text
Stage1
=
B1
+
ROOT_stage1
```

ROOT_stage1 从全部 2,124 ROOT 中固定随机抽取。

固定 seed：

```text
42
```

要求：

```text
ROOT_stage1 ⊂ ROOT_all
```

保存具体 sample IDs。

所有 curriculum runs 必须复用完全相同的 Stage-1 ROOT subset。

---

# 6. Stage 2 数据

Stage 2 使用完整 train：

```text
Stage2
=
B1
+
B2
+
ROOT_all
```

即：

```text
12,828 sentences
```

Stage 2 不只训练 B2。

这是 cumulative curriculum：

```text
easy subset
→
full train
```

而不是：

```text
easy
→
hard only
```

---

# 7. 训练流程

训练必须连续：

```text
initial model
    ↓
Stage 1
    ↓
Stage-1 checkpoint
    ↓
Stage 2
    ↓
final model
```

Stage 2 必须从 Stage 1 最终 checkpoint 恢复。

不要：

```text
Stage 1 train
Stage 2 reinitialize from base model
```

---

# 8. 模型初始化

本次 curriculum run 应与 Original C baseline 使用相同初始化。

优先查找现有 Original C run/config，例如：

```text
tasks/stage2-event-tree/runs/C-42-20260918/
```

确认：

- base model / checkpoint；
- LoRA / adapter 配置；
- tokenizer；
- prompt；
- max length；
- optimizer；
- learning rate；
- scheduler；
- batch size；
- gradient accumulation；
- warmup；
- precision；
- seed；
- evaluation cadence；
- checkpoint cadence。

除 curriculum data schedule 外，尽量保持与 Original C baseline 一致。

不要引入新的 Stage-1 typed-mention adapter。

不要切换到 Appendix+Type-COT 版本。

本任务目标是：

```text
Original C
vs
Original C + curriculum
```

---

# 9. Prompt / Output / Task Definition 必须保持不变

保持 Original C 的任务：

```text
full abstract + target sentence
```

只抽取：

```text
trigger 位于 target sentence 的 events
```

arguments 可以来自 full abstract。

输出仍为 event forest：

```text
(ROOT
  (EVENT_TYPE
    (Trigger "...")
    (Role "...")))
```

无事件：

```text
(ROOT)
```

不要修改：

- prompt wording；
- output S-expression；
- event type definitions；
- role set；
- decoding format；
- training objective；
- loss function。

---

# 10. 总训练预算必须与 Baseline 对齐

这是本任务最重要的控制条件之一。

不要简单执行：

```text
Stage 1 = baseline full epochs
+
Stage 2 = baseline full epochs
```

这样会导致 curriculum 获得更多 optimizer updates。

必须先读取 Original C baseline 的实际训练预算：

```text
T_baseline
=
total optimizer steps
```

如果 baseline 是通过：

```text
num_train_epochs
```

控制，也需要从日志/config 中恢复实际：

```text
optimizer/update steps
```

最终 curriculum 满足：

```text
T_stage1 + T_stage2
=
T_baseline
```

---

# 11. 第一版固定 50% / 50% Step Schedule

本次先采用最简单的 schedule：

```text
T_stage1 ≈ 0.5 * T_baseline
T_stage2 = T_baseline - T_stage1
```

如果总 step 为奇数：

```text
T_stage1 = floor(T_baseline / 2)
T_stage2 = T_baseline - T_stage1
```

不要为两个阶段分别重新设置完整 epoch 数。

优先按：

```text
max_steps
```

或现有训练框架中等价的 update-step 控制方式实现。

---

# 12. Learning-rate Scheduler

优先保持 baseline 的 scheduler 逻辑。

但要避免 Stage 1 和 Stage 2 各自重新启动一个完整 LR schedule。

理想行为：

```text
global training step:
0 → T_baseline
```

LR schedule 也以总：

```text
T_baseline
```

为 horizon。

也就是说：

```text
Stage 1:
global steps 0 ... T_stage1

Stage 2:
resume global steps T_stage1 ... T_baseline
```

如果当前训练框架 resume 时能自然恢复 scheduler state，直接使用。

如果不能，必须明确记录实际行为，不要静默重启 scheduler。

---

# 13. Optimizer State

Stage 2 应同时恢复：

- model weights；
- optimizer state；
- LR scheduler state；
- global step；
- scaler state（如果有）；
- random state（如果框架支持）。

目标是：

```text
真正连续训练
```

而不是仅加载 Stage-1 model weights 后重新建 optimizer。

如果现有框架技术上无法完整恢复，必须在报告中明确写出。

---

# 14. Stage 1 Sampling

Stage 1 dataset：

```text
B1 + ROOT_stage1
```

正常 shuffle。

不要再根据 difficulty 进行 sample weighting。

不要在 B1 内排序喂入。

Curriculum 粒度仅为：

```text
Stage 1 subset
vs
Stage 2 full dataset
```

不是 sample-by-sample difficulty ordering。

---

# 15. Stage 2 Sampling

Stage 2：

```text
full train
```

正常 shuffle。

不要对 B1 或 B2 oversample。

不要让 B1 在 Stage 2 额外重复一次。

即 Stage 2 的数据分布应恢复成原始 train distribution。

---

# 16. Validation

Dev set 保持原始 Experiment C dev。

不要根据 B1/B2 改 dev。

每个训练阶段结束至少做一次完整 dev evaluation。

如果现有框架支持 baseline 相同的 periodic eval，则保持同样 cadence。

至少保存：

```text
Stage 1 end dev metrics
Stage 2 final dev metrics
best dev checkpoint metrics
```

评估指标至少包括 Original C 已有：

```text
TI
TC
AI
AC
EC
```

如果已有 type-only diagnostics，也可以额外保存，但不要替代官方指标。

---

# 17. Checkpoint Selection

尽量保持 Original C 的 best-checkpoint selection 规则。

例如如果 baseline 根据某一个 dev metric 选择 best checkpoint，则 curriculum 也使用相同规则。

不要因为 curriculum 改成针对某个更有利的指标选 checkpoint。

报告：

```text
best checkpoint
best step
selection metric
```

---

# 18. 第一版只跑一个 Curriculum 主实验

本任务首先完成：

```text
Original C + Mean-Percentile Curriculum
```

不要同时自动跑大量 ablation。

主 run 名称建议：

```text
C-CL-Mean-42-<date>
```

或遵循现有 runs 命名方式。

seed：

```text
42
```

保持与 Original C 对照一致。

---

# 19. 建议同时准备但不要自动训练的 Control 配置

只生成配置 / assignment，暂时不要执行训练：

## A. Event-count-only curriculum

定义：

```text
D_event(s)
=
1 - 1/N_e(s)
```

按相同规则构建 easy/hard 两组。

用途：

```text
control whether gains come mainly from event multiplicity
```

## B. Random-half control

从 positive train 中随机抽约 50% 作为 Stage 1 positive：

```text
same positive count as B1
```

同样使用完全相同的：

```text
ROOT_stage1
```

用途：

```text
control whether simply pretraining on half the dataset helps
```

当前任务只准备 assignment/config，不自动训练这两组。

---

# 20. 必须生成的数据文件

放到：

```text
tasks/stage2-event-tree/curriculum/
```

建议新增：

```text
build_mean_curriculum_data.py

mean_curriculum_stage1.jsonl
mean_curriculum_stage2.jsonl

mean_curriculum_stage1_ids.txt
mean_curriculum_stage2_ids.txt

mean_curriculum_root_stage1_ids.txt

mean_curriculum_manifest.json
```

其中：

```text
mean_curriculum_stage1.jsonl
=
B1 + sampled ROOT_stage1
```

```text
mean_curriculum_stage2.jsonl
=
full original train
```

不要修改原始：

```text
data/sentence-c/train.jsonl
```

---

# 21. Manifest 必须记录

`mean_curriculum_manifest.json` 至少包含：

```text
total_train
positive_train
root_total

B1_count
B2_count

difficulty_formula
difficulty_threshold
tie_policy

root_stage1_count
root_stage1_seed

stage1_total_count
stage2_total_count

baseline_total_optimizer_steps
stage1_optimizer_steps
stage2_optimizer_steps

base_model
training_seed
source_train_file
difficulty_assignment_file
```

---

# 22. 数据一致性验证

必须 assert：

```text
B1 ∩ B2 == empty
```

```text
B1 ∪ B2 == all positive train
```

```text
ROOT ∩ positive == empty
```

```text
ROOT_stage1 ⊂ ROOT
```

```text
Stage1
=
B1 ∪ ROOT_stage1
```

```text
Stage2
=
B1 ∪ B2 ∪ ROOT
=
full train
```

并检查：

```text
len(Stage2) == 12828
```

当前数据下。

---

# 23. ROOT 比例验证

报告：

```text
full train ROOT ratio
stage1 ROOT ratio
stage2 ROOT ratio
```

Stage 1 应接近：

```text
16.56%
```

允许因为整数 rounding 有极小误差。

如果偏差 > 0.2 percentage points，检查计算。

---

# 24. 训练启动脚本

建议新增：

```text
run_c_mean_curriculum.sh
```

或与项目现有 launcher 风格一致的脚本。

脚本应明确：

```text
Stage 1 dataset path
Stage 1 max steps
Stage 1 output dir
```

然后：

```text
Stage 2 dataset path
resume_from_checkpoint
total/global remaining steps
Stage 2 output dir
```

如果训练框架支持一个 process 内动态切换 dataset，也可以实现为单个 Python training script。

但优先选择：

```text
最小修改现有 C pipeline
```

---

# 25. 日志要求

保存：

```text
stage1_train.log
stage2_train.log
```

或统一日志。

至少能恢复：

```text
global step
train loss
learning rate
epoch / effective data passes
eval steps
dev metrics
checkpoint paths
```

特别报告：

```text
Stage 1 实际看了多少 optimizer steps
Stage 2 实际看了多少 optimizer steps
最终总 optimizer steps
```

确认与 baseline 相同。

---

# 26. 最终报告

输出：

```text
tasks/stage2-event-tree/curriculum/mean_curriculum_training_report.md
```

报告分为以下部分。

## Q1. Data construction

报告：

```text
B1
B2
ROOT total
ROOT_stage1
Stage1 total
Stage2 total
```

以及 ROOT ratio。

## Q2. Training budget

报告：

```text
baseline optimizer steps
stage1 steps
stage2 steps
curriculum total steps
```

确认：

```text
curriculum total == baseline total
```

## Q3. Resume correctness

确认 Stage 2 是否恢复：

```text
model
optimizer
scheduler
global step
```

若有任何状态未恢复，明确写出。

## Q4. Dev results

表格：

| Run / Stage | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| Original C baseline | | | | | |
| Curriculum Stage 1 end | | | | | |
| Curriculum final | | | | | |
| Curriculum best checkpoint | | | | | |

Original C baseline 可以从已有报告/日志读取，不重新训练。

## Q5. Delta vs Original C

计算：

```text
Curriculum best - Original C
```

分别报告：

```text
ΔTI
ΔTC
ΔAI
ΔAC
ΔEC
```

只报告结果，不做过度因果解释。

## Q6. Training behavior

简单报告：

- Stage 1 loss；
- Stage 2 loss；
- Stage 切换时是否出现明显 loss jump；
- 最优 checkpoint 位于 Stage 1 还是 Stage 2；
- 是否出现训练不稳定 / NaN / OOM。

---

# 27. Original C baseline reference

优先查找并核对已有 Original C dev 结果。

当前已有参考值可能为：

```text
TI = 74.09
TC = 62.55
AI = 43.03
AC = 40.75
EC = 45.37
```

但不要只依赖这里的手写数字。

必须从现有 run / report / checkpoint eval 文件中核验一次，并在最终报告注明来源文件。

不要重新训练 baseline。

---

# 28. 本任务不要做的事情

不要：

- 修改 difficulty formula；
- 从 mean 改回 min；
- 调 difficulty 权重；
- 重新调 median threshold；
- 强行让 B1/B2 精确 50:50；
- 把 ROOT 当最简单样本排序；
- Stage 1 使用全部 ROOT；
- Stage 2 只训练 B2；
- Stage 2 重新初始化；
- 给 curriculum 更多 optimizer steps；
- 修改 prompt；
- 修改 Appendix schema；
- 加 Type-COT；
- 加 Stage-1 typed-mention warm start；
- 加 complex argument curriculum；
- 加 schema confusion loss；
- 修改 event-tree loss；
- 修改 decoding；
- 自动跑多个 seed。

---

# 29. 当前主实验定义

最终主实验必须严格对应：

```text
Difficulty:
D(s)
=
0.5 *
[
(1 - 1/N_e(s))
+
(1 - mean_e P(e))
]

Positive split:
B1 = D <= train median
B2 = D > train median
boundary ties → B1

ROOT:
not scored

Stage 1:
B1 + sampled ROOT
ROOT ratio ≈ original train ROOT ratio

Stage 2:
full original train

Training:
continuous resume
same total optimizer steps as Original C
same model / prompt / objective / optimizer settings
seed = 42
```

---

# 30. 最终目的

本次实验只回答：

> 在不增加总训练预算、不改变 Experiment C 任务和模型的情况下，按照 event multiplicity + mean within-type schema incompleteness 构造的 two-stage cumulative curriculum，是否能改善 Original C 的 dev event extraction performance？

不要在代码或报告中提前假设答案。
