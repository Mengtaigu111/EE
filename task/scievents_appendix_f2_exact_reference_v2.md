# SciEvents Appendix F.2 — Type Warm-up 唯一参考 v2

来源：Lu et al., *EXCEEDS: Extracting Complex Events via Nugget-based Grid Modeling in Scientific Domain*, ACL 2026, Appendix F.2, Tables 9–18（论文页 6012–6014）。

本文件是模板正文与 Role 列表的唯一来源；输出格式见实现说明文件。

旧项目 `ontology.py` 中的 natural labels 不用于本实验输出。

## 1. Code -> FullName

```text
PUR -> Purpose
ITT -> IntroduceTarget
RWS -> RelatedWorkStep
RWF -> RelatedWorkFault
PRP -> Propose
WKS -> WorkStatement
MDS -> MethodStep
FIN -> Finding
CMP -> ExperimentCompare
FAC -> OutcomeFact
```

## 2. Appendix F.2 原始模板正文

以下保留论文模板正文中的措辞与 placeholder/arg 编号，包括论文中的原始拼写。

### PUR — Purpose

```text
In order to deal with <Aim:arg1> under <Condition:arg2> circumstance on <Dataset:arg3> datasets.
```

Roles:

```text
Aim
Condition
Dataset
```

### ITT — IntroduceTarget

```text
<Target:arg1> is the abstract research target under <Condition:arg2> circumstance on <Dataset:arg3> datasets in this paper.
```

Roles:

```text
Target
Condition
Dataset
```

### RWS — RelatedWorkStep

```text
Previously <Subject:arg1> on <Target:arg2> are mostly based on <BaseComponent:arg3> with <TriedComponent:arg4> under <Condition:arg5> circumstance on <Dataset:arg6> datasets.
```

Roles:

```text
Subject
Target
BaseComponent
TriedComponent
Condition
Dataset
```

### RWF — RelatedWorkFault

```text
Aiming to <Target:arg1>, to <Extent:arg5> degree, <Concern:arg2> has some <Fault:arg6> faults under <Condition:arg3> circumstance on <Dataset:arg4> datasets.
```

Roles:

```text
Target
Extent
Concern
Fault
Condition
Dataset
```

注意：这里 arg 编号顺序按论文原文保留，不要自行重排。

### PRP — Propose

```text
In this paper, <Proposer:arg1> propose <Content:arg2> for <Target:arg3>.
```

Roles:

```text
Proposer
Content
Target
```

### WKS — WorkStatement

```text
<Researcher:arg1> report <Content:arg2> under <Condition:arg3> circumstance on <Dataset:arg4> datasets for <Target:arg5>.
```

Roles:

```text
Researcher
Content
Condition
Dataset
Target
```

### MDS — MethodStep

```text
Our approach adopt <BaseComponent:arg1> with <TriedComponent:arg2> under <Condition:arg3> circumstance on <Dataset:arg4> datasets for <Target:arg5>.
```

Roles:

```text
BaseComponent
TriedComponent
Condition
Dataset
Target
```

### FIN — Finding

```text
In experiments, <Finder:arg1> find or demostrate findings that <Content:arg2>.
```

Roles:

```text
Finder
Content
```

注意：`demostrate` 按论文原文保留。

### CMP — ExperimentCompare

```text
Experimental results show that the <Metrics:arg6> of <Arg1:arg1> is <Extent:arg2> <Result:arg3> than <Arg2:arg4> under <Condition:arg5> circumstance on <Dataset:arg7> datasets.
```

Roles:

```text
Metrics
Arg1
Extent
Result
Arg2
Condition
Dataset
```

### FAC — OutcomeFact

```text
Experimental results show that <Subject:arg1> can <Extent:arg2> provide <Object:arg3> for <Target:arg4> under <Condition:arg5> circumstance on <Dataset:arg6> datasets because <Reason:arg7> reasons.
```

Roles:

```text
Subject
Extent
Object
Target
Condition
Dataset
Reason
```

---

## 3. 本文件的用途边界

本文件只提供：
- 每个 Event Type 的 `paper_template`（Appendix F.2 原始文本，含原始 `<Role:argN>` 编号与原始拼写）；
- 每个 Event Type 的 Role 列表。

本文件不定义模型 completion 的输出格式。TypeCode 前缀、placeholder 写法、重复 Role、`None.` 等一律以实现说明文件为准。

## 4. paper_template full rendering

`schema.py` 中的 `paper_template` 必须与本文件第 2 节的模板正文逐字符相等（空白 normalize 后比较），并加 snapshot test。`completion_template` 由 `paper_template` 仅去掉 `:argN` 得到。

---

## 5. Role constrained types

Argument Role 的合法性以 Appendix F.2 Tables 9–18 为准。当前 warm-up 只预测 Role 名及 multiplicity，不预测 nugget type，因此 constrained nugget types 只用于 schema 校验，不进入 completion。
