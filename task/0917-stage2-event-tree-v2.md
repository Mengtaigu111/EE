# Stage-2 Schema-Guided Event Tree Generation with Qwen3-8B

## 1. Goal

本任务实现 SciEvents 的 Stage-2 Event Tree Generation。

输入：

- 一篇完整 scientific abstract

输出：

- event type
- trigger
- argument role
- argument mention

所有事件统一表示为一个 Event Forest。

Stage 2 的核心目标是：

> 将 event type、trigger、argument role 和 argument mention 组织到统一的树结构中，并通过 canonical tree generation、autoregressive structural conditioning 和 event schema conditioning，使这些彼此相关的预测目标能够在同一结构中进行联合建模。

第一版 Stage 2 不要求把 Stage-1 predicted mentions 显式拼进 prompt。

Stage 1 主要作为：

```text
scientific argument mention / nugget type semantics
```

的预训练知识来源，通过 warm start 测试其是否可以迁移到 Stage 2。

显式 Stage-1 mention guidance、Role→NuggetType compatibility、
grammar-constrained decoding 均作为后续增强实验。

---

## 2. Core Method Overview

整体流程：

```text
SciEvents Gold Annotations
          ↓
Gold Event Forest Construction
          ↓
Canonical Ordering
          ↓
DFS / Bracket Linearization
          ↓
Gold Tree Sequence
          ↓
Qwen3-8B Autoregressive SFT
          ↓
Generated Tree Sequence
          ↓
Tree Parsing + Schema Validation
          ↓
Text-to-Offset Alignment
          ↓
SciEvents Event Format
          ↓
TI / TC / AI / AC / EC
```

核心方法由三部分组成：

```text
1. Tree Representation
2. Autoregressive Structural Conditioning
3. Event Schema Conditioning
```

三者分别解决：

```text
Tree Representation
→ 谁属于哪个 event、扮演什么 role

Autoregressive Structural Conditioning
→ 后面的结构决策依赖前面已经生成的 event components

Event Schema Conditioning
→ 不同 event type 只能使用 ontology 中合法的 roles
```

---

## 3. Event Tree Representation

一篇 abstract 对应一个 Event Forest：

```text
ROOT
├── CMP
│   ├── Trigger
│   │   └── "outperforms"
│   ├── Arg1
│   │   └── "our model"
│   ├── Dataset
│   │   └── "SciEvents"
│   └── Extent
│       └── "significantly"
│
├── PRP
│   ├── Trigger
│   │   └── "propose"
│   ├── Proposer
│   │   └── "we"
│   └── Content
│       └── "our framework"
│
└── ...
```

基本结构：

```text
ROOT
  └── EventType
        ├── Trigger → trigger mention
        ├── Role 1  → argument mention
        ├── Role 2  → argument mention
        └── ...
```

同一个 role 有多个 arguments 时，允许重复 role 节点：

```text
(CMP
  (Trigger "compare")
  (Arg1 "BERT")
  (Arg1 "RoBERTa")
  (Dataset "GLUE")
)
```

最终树不显式输出 nugget type。

---

## 4. Gold Event → Gold Event Forest

训练前，必须先将 SciEvents gold annotations 确定性地转换为 Gold Event Forest。

对于每一个 gold event：

```text
event type
trigger
arguments:
  role
  argument nugget
```

构造：

```text
EventType
├── Trigger → trigger text
├── Role → argument text
└── ...
```

例如 gold：

```text
EventType = CMP
Trigger   = "outperforms"

Arguments:
Arg1    = "our model"
Dataset = "SciEvents"
Extent  = "significantly"
```

转换为：

```text
CMP
├── Trigger → "outperforms"
├── Arg1 → "our model"
├── Dataset → "SciEvents"
└── Extent → "significantly"
```

整篇 abstract 中的所有 events 统一挂在虚拟 ROOT 下。

```text
ROOT
├── Event 1
├── Event 2
└── ...
```

Gold Event Forest 构建必须完全 deterministic，
不能依赖模型预测或启发式搜索。

---

## 5. Canonical Tree Linearization

模型实际训练的 target 不是二维树，
而是 Gold Event Forest 的唯一线性化序列。

因此同一个 gold structure 必须只有一种 target sequence。

### 5.1 Canonical Order

规则：

1. Events 按 trigger start offset 排序；
2. 每个 Event 中 Trigger 永远第一个；
3. Roles 按 SciEvents ontology 中预先固定的顺序；
4. 同一个 Role 的多个 arguments 按 argument start offset 排序；
5. 所有 span text 均从 abstract 原样复制。

例如：

```text
ROOT
├── PRP
│   ├── Trigger → "propose"
│   ├── Proposer → "we"
│   └── Content → "our framework"
└── CMP
    ├── Trigger → "outperforms"
    ├── Arg1 → "our model"
    ├── Dataset → "SciEvents"
    └── Extent → "significantly"
```

线性化为：

```text
(ROOT
  (PRP
    (Trigger "propose")
    (Proposer "we")
    (Content "our framework")
  )
  (CMP
    (Trigger "outperforms")
    (Arg1 "our model")
    (Dataset "SciEvents")
    (Extent "significantly")
  )
)
```

第一版使用 bracket-based DFS linearization。

训练、dev、test 必须使用完全相同的 linearization 规则。

---

## 6. Joint Structured Generation

Stage 2 不分别训练独立的：

```text
P(EventType | x)
P(Trigger   | x)
P(Role      | x)
P(Argument  | x)
```

而是直接生成完整 Event Tree：

```text
T = (
  EventType,
  Trigger,
  {(Role_i, Argument_i)}
)
```

例如：

```text
(CMP
  (Trigger "outperforms")
  (Arg1 "our model")
  (Dataset "SciEvents")
  (Extent "significantly")
)
```

树中的 parent-child relation 显式表示：

```text
CMP → Trigger → "outperforms"

CMP → Dataset → "SciEvents"

CMP → Extent → "significantly"
```

因此：

```text
"SciEvents"
```

不是先作为一个孤立 span 被抽取，
再在后处理中连接到 CMP。

它是在生成过程中直接作为：

```text
CMP 的 Dataset argument
```

被生成。

---

## 7. Autoregressive Structural Conditioning

底层模型仍然使用标准 autoregressive generation。

设 linearized Event Forest 为：

```text
y = (y1, y2, ..., yn)
```

则：

```math
P(T | x) = ∏_j P(y_j | y_<j, x)
```

其中：

```text
x = instruction + abstract + event schema
```

由于 target 的 tree order 被固定为：

```text
EventType
→ Trigger
→ Role
→ Argument
→ Role
→ Argument
→ ...
```

所以当模型生成某个 argument 时，
prefix 中已经包含当前：

```text
Event Type
Trigger
Argument Role
Previous Tree Structure
```

从结构层面可以理解为：

```math
P(T | x)
≈
P(e | x)
P(t | e, x)
∏_i
P(r_i | e, t, T_<i, x)
P(a_i | e, t, r_i, T_<i, x)
```

这里是对 autoregressive tree serialization 的结构性解释。

模型底层真正优化的仍然是 token-level likelihood，
不是显式实现四个单独的 probability heads。

例如已经生成：

```text
(CMP
  (Trigger "outperforms")
  (Dataset
```

此时生成 `"SciEvents"` 时，
模型实际 conditioned on：

```text
Abstract
EventType = CMP
Trigger   = outperforms
Role      = Dataset
Previous Tree Prefix
```

因此 argument prediction 显式依赖当前 event structure。

这就是 Stage 2 中 structural dependency 的核心来源。

---

## 8. Training Objective

Stage 2 不设计额外的 tree-specific loss。

训练流程：

```text
Gold SciEvents Annotations
        ↓
Gold Event Forest
        ↓
Canonical Tree Linearization
        ↓
Gold Tree Sequence
        ↓
Teacher Forcing
        ↓
Token-level Cross Entropy / NLL
```

对一个训练样本：

```text
Input:
instruction + abstract + event schema

Target:
linearized gold Event Forest
```

训练目标：

```math
L
=
- ∑_j log P(y_j | y_<j, x)
```

训练时使用 gold tree prefix。

即预测第 j 个 target token 时，
模型 condition 的 `y_<j` 来自 gold tree。

因此：

```text
tree structure supervision
```

不是通过额外：

```text
L_parent
L_child
L_edge
L_role
L_span
```

来实现。

而是通过：

```text
1. Event Tree parent-child organization
2. canonical structural ordering
3. bracket-based serialization
4. teacher-forced autoregressive likelihood
```

共同编码进 target sequence。

---

## 9. Why This Is Joint Structural Modeling

本方法中的“joint”不是指模型内部显式存在：

```text
event-type head
trigger head
role head
argument head
```

而是指：

```text
EventType
Trigger
Role
Argument
```

被组织在同一个结构化预测目标中。

Tree Representation 明确表示：

```text
which argument belongs to which event
```

以及：

```text
what role the argument plays
```

Canonical autoregressive generation 又使：

```text
later structural decisions
```

显式 conditioned on：

```text
previously generated event components
```

因此：

```text
EventType
     ↓
  Trigger
     ↓
    Role
     ↓
 Argument
```

不是四个互相独立的标签。

Stage 2 的联合结构建模来自：

```text
Tree Representation
+
Autoregressive Structural Conditioning
+
Event Schema Conditioning
```

---

## 10. Event Schema Conditioning

SciEvents ontology 定义：

```text
EventType → AllowedRoles
```

例如：

```text
CMP:
  Arg1
  Arg2
  Condition
  Dataset
  Result
  Metrics
  Extent
```

训练和推理时显式向模型提供完整 event schema。

作用：

1. 明确每个 EventType 可以出现哪些 roles；
2. EventType 为后续 Role prediction 提供结构条件；
3. 可以验证生成 tree 是否违反 ontology；
4. 后续可以进一步用于 constrained decoding。

第一版只把：

```text
EventType → AllowedRoles
```

作为 Stage-2 核心 schema。

暂时不把：

```text
Role → NuggetType
```

作为硬约束。

原因：

Stage-1 predicted nugget type 并不完全可靠，
因此不能让 Stage-1 error 决定 Stage-2 argument upper bound。

Role→NuggetType compatibility 留作后续增强实验。

---

## 11. Tree Grammar

第一版定义简单合法结构：

```text
ROOT
→ Event*

Event
→ EventType + Trigger + Argument*

Argument
→ Role + Mention
```

合法 tree 必须满足：

- Event 必须有 EventType；
- Event 必须有 Trigger；
- Argument 必须挂在 Role 下；
- Role 必须挂在 Event 下；
- Role 必须属于当前 EventType 的 schema。

Tree Grammar 的作用是定义：

```text
what constitutes a valid event tree
```

第一版不在训练阶段对 softmax 做 grammar mask。

---

## 12. Training vs Inference Constraints

训练阶段：

```text
gold tree
→ canonical serialization
→ teacher forcing
→ standard autoregressive CE
```

第一版不加入：

```text
grammar mask
schema mask
special tree loss
candidate pruning loss
```

推理阶段：

```text
abstract
→ autoregressive generation
→ tree parsing
→ schema validation
```

先统计：

```text
invalid tree rate
schema violation rate
```

如果两者已经很低，
不额外实现复杂 constrained decoding。

如果较高，再做后续增强：

```text
grammar / schema-constrained decoding
```

其中：

```text
structure tokens / EventType / Role
```

可以约束得较强。

但：

```text
Trigger text / Argument text
```

第一版仍允许模型自由从原 abstract copy。

---

## 13. Sub-event Recovery

第一版不把 sub-event 递归嵌套到树中。

所有 events 先独立挂在 ROOT 下：

```text
ROOT
├── Event A
├── Event B
└── Event C
```

生成结束并完成 text-to-offset alignment 后：

如果：

```text
Event A 的 argument offsets
==
Event B 的 trigger offsets
```

则恢复：

```text
Event A --Role--> Event B
```

最终用于 Event Correlation (EC) evaluation。

这样可以避免第一版 tree generation 同时承担：

```text
event extraction
+
recursive event nesting
```

两种复杂任务。

---

## 14. Stage-1 Knowledge Transfer

Stage 1 已学习：

```text
scientific argument mention
+
nugget type semantics
```

第一版不把 Stage-1 predicted mentions 显式加入 Stage-2 input。

测试两种初始化：

### Base

```text
Qwen3-8B
→ Stage-2 Tree SFT
```

### Warm Start

```text
Qwen3-8B
→ load Stage-1 trained adapter / weights
→ continue Stage-2 Tree SFT
```

目的：

> 测试 Stage-1 学到的 argument mention semantics 是否能迁移到 Stage-2 Event Tree Generation。

Stage-1 explicit guidance 不属于第一版主方法。

后续再测试：

```text
Abstract
+
Stage-1 predicted mentions
→ Event Tree
```

---

## 15. Dataset Construction

来源：

```text
data/SciEvents/{train,dev,test}.json
```

一个训练样本：

```text
Input:
完整 abstract

Output:
该 abstract 的 canonical linearized Event Forest
```

Gold target 只包含：

```text
event type
trigger surface text
argument role
argument surface text
```

不包含：

```text
nugget type
offset
recursive sub-event node
```

所有 trigger / argument text 必须从 abstract 原样复制。

---

## 16. Prompt

### 16.1 System Prompt

```text
You are an expert in scientific event extraction.

Given a scientific abstract, extract all scientific events
and represent them as one event forest.

Each event must contain:
1. its event type;
2. its trigger;
3. its argument roles and corresponding argument mentions.

Follow the event schema provided below.
Only use argument roles that are valid for the corresponding event type.

Copy every trigger and argument exactly as it appears in the abstract.

Use the required bracketed tree format.
Return only the event forest.

Event schema:
<full SciEvents event schema>
```

### 16.2 User Prompt

```text
Abstract:
<full abstract>
```

### 16.3 Assistant Target

```text
(ROOT
  ...
)
```

---

## 17. Experiments

第一轮只跑三个核心实验。

| ID | Initialization | Schema | Purpose |
| --- | --- | --- | --- |
| T0 | Base Qwen3-8B | No | Tree-generation baseline |
| T1 | Base Qwen3-8B | Yes | 测试 EventType→Role schema |
| T2 | Stage-1 warm start | Yes | 测试 Stage-1 knowledge transfer |

核心比较：

```text
T1 vs T0
```

回答：

> 显式 EventType → AllowedRoles 是否帮助 Event Tree Generation？

```text
T2 vs T1
```

回答：

> Stage-1 typed mention learning 是否能够帮助 Stage-2 argument extraction？

第一轮先使用 seed 42 完成筛选。

最终配置再补：

```text
seed 43
seed 44
```

并报告 mean ± std。

---

## 18. Training Configuration

模型：

```text
/opt/models/Qwen3-8B
```

优先沿用 Stage-1 已验证配置：

```text
precision        bf16

LoRA targets     q_proj k_proj v_proj o_proj
                 gate_proj up_proj down_proj

r                16
alpha            32
dropout          0.05

learning rate    1e-4
scheduler        cosine
warmup ratio     0.03

epochs           3
effective batch  16

gradient checkpointing    on
loss                       assistant tokens only
enable_thinking            False
```

Stage 2 在正式训练前必须统计：

```text
input token length
target tree token length
input + target token length
```

至少报告：

```text
mean
p95
p99
max
```

再确定：

```text
max_len
max_new_tokens
```

不得直接沿用 Stage-1 的长度配置。

OOM 时：

1. per-device batch 降到 1；
2. 再调整 max_len；
3. 所有修改写入 config 和 log。

---

## 19. Inference and Parsing

第一版：

```text
greedy decoding
enable_thinking=False
```

推理后依次：

1. parse ROOT；
2. parse Event；
3. parse EventType；
4. parse Trigger；
5. parse Role → Argument；
6. validate Role 是否属于当前 EventType；
7. 记录 malformed tree；
8. 记录 schema violations。

保存：

```text
raw_output
parsed_tree
invalid
schema_violations
```

---

## 20. Text-to-Offset Alignment

模型生成 surface text。

评测前将 trigger / argument text 映射回 document token offsets。

### Trigger

1. continuous exact token match；
2. 大小写敏感优先；
3. 多处相同文本时按 event tree 顺序确定性匹配；
4. 仍无法确定则记录 ambiguous / unaligned。

### Argument

1. continuous exact token match；
2. 大小写敏感优先；
3. 多处相同文本时优先当前 trigger 所在句；
4. 再优先距离 trigger 更近的位置；
5. 再按从左到右确定性匹配；
6. 必要时允许处理 SciEvents 中已有的 discontinuous nugget 对齐；
7. 无法对齐记 unaligned。

不使用 fuzzy replacement 修改模型输出。

---

## 21. Conversion and Evaluation

将 parsed tree 转换为 SciEvents official event format。

报告官方：

```text
TI
TC
AI
AC
EC
```

同时报告：

```text
invalid tree rate
schema violation rate
trigger unaligned rate
argument unaligned rate
```

错误分析至少包含：

```text
Event Type Error
Trigger Missing
Trigger Boundary
Argument Missing
Argument Boundary
Role Error
Wrong Event Attachment
```

---

## 22. Main Research Questions

### Q1

Tree-structured generation 能否直接完成 SciEvents end-to-end event extraction？

### Q2

canonical Event Tree 是否能够把：

```text
EventType
Trigger
Role
Argument
```

放入一个统一结构中进行 joint structured generation？

### Q3

显式加入：

```text
EventType → AllowedRoles
```

是否改善 event extraction？

### Q4

Stage-1 typed mention learning 是否能够迁移到 Stage 2？

---

## 23. Follow-up Experiments

第一轮不实现。

### F1. Explicit Stage-1 Mention Guidance

```text
Abstract
+
Stage-1 predicted mentions
+
Event schema
→ Event Tree
```

### F2. Nugget-Type Compatibility

```text
EventType
→ Role
→ NuggetType
```

只作为 soft guidance，
不把 Stage-1 predicted type 作为 hard candidate constraint。

### F3. Grammar-Constrained Decoding

如果：

```text
invalid tree rate
schema violation rate
```

明显较高，再实现。

### F4. Improved Stage-1 Recall

后续再测试：

```text
untyped span discovery → type classification
```

或：

```text
type-conditioned extraction
```

---

## 24. Deliverables

```text
tasks/stage2-event-tree/

README.md

scripts/
  build_tree_dataset.py
  build_schema.py
  linearize_tree.py
  train_lora.py
  predict.py
  parse_tree.py
  validate_tree.py
  align_tree.py
  recover_sub_events.py
  convert_to_scievents.py
  evaluate.py
  run_experiment.sh

tests/
  test_gold_tree.py
  test_tree_linearization.py
  test_tree_parser.py
  test_schema.py
  test_alignment.py
  test_sub_event_recovery.py

runs/
  <experiment>-<seed>-<timestamp>/

report.md
```

`report.md` 至少包含：

```text
1. T0 / T1 / T2 的 TI / TC / AI / AC / EC
2. invalid tree rate
3. schema violation rate
4. trigger / argument alignment failure rate
5. EventType / Trigger / Role / Argument error analysis
6. T1 vs T0：schema 是否有效
7. T2 vs T1：Stage-1 warm start 是否有效
8. 最终配置 3-seed mean ± std
```

---

## 25. Implementation Priorities

优先级：

```text
P0
Gold Event → Gold Tree
Canonical Linearization
Tree Parser
Schema Validator
Official Evaluation Conversion

P1
T0 Tree Generation Baseline
T1 Schema-aware Tree Generation

P2
T2 Stage-1 Warm Start

P3
Explicit Stage-1 Guidance
Role→NuggetType Guidance
Grammar-Constrained Decoding
```

第一轮最重要的是先验证：

```text
Gold Event
→ Canonical Tree
→ Linearized Tree
→ Autoregressive SFT
→ Parsed Event Tree
```

这一条完整链路。

只有这条链路稳定之后，
再逐步加入额外 guidance 或 decoding constraints。
