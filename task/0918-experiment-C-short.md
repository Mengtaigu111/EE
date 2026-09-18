# Experiment C — Document-Conditioned Sentence-Level Event Generation

## 1. Goal

验证一个问题：

> **是否必须一次生成整篇 abstract 的完整 Event Forest？**

当前 T0：

```text
Full Abstract
→ One Long Event Forest
```

实验 C：

```text
Full Abstract + Target Sentence
→ Only Events Triggered in Target Sentence
→ Merge All Sentence Predictions
→ Document-level Evaluation
```

核心思想：

> **保留 document-level context，但把 decoding granularity 降到 sentence。**

---

## 2. Prediction Unit

对一篇文档：

```text
D = [S1, S2, ..., Sn]
```

构造：

```text
(D, target=S1) → S1 中 trigger 对应的 events
(D, target=S2) → S2 中 trigger 对应的 events
...
```

Event 按 **trigger 所在 sentence** 分配。

重要：

- Trigger 必须位于 target sentence。
- Argument 可以位于整篇 abstract 的任意位置。
- 每个 gold event 只能分配一次。
- target sentence 没有 event 时输出：

```text
(ROOT)
```

---

## 3. Prompt

输入仍然包含完整 abstract：

```text
Read the full abstract for context.

Extract all scientific events whose TRIGGER occurs
in the TARGET SENTENCE.

Arguments may appear anywhere in the abstract.

Follow the SciEvents event schema.
Copy trigger and argument text exactly from the abstract.
Return only the local event forest.

Full Abstract:
[S1] ...
[S2] ...
[S3] ...

TARGET SENTENCE:
[S2] ...

Output:
(ROOT ...)
```

---

## 4. Target Tree

沿用当前 Stage-2 tree format：

```text
(ROOT
  (CMP
    (Trigger "outperforms")
    (Arg1 "our model")
    (Dataset "SciEvents"))
)
```

排序规则不变：

```text
events: trigger offset
Trigger: first
roles: ontology order
same-role arguments: offset order
```

---

## 5. Training

模型和 T0 保持一致：

```text
Qwen3-8B
LoRA
bf16
lr = 1e-4
epochs = 5
assistant-token loss only
enable_thinking = False
```

第一轮只跑：

```text
seed = 42
```

保存每个 epoch checkpoint，在 dev 上选择最佳 checkpoint。

训练前重新统计 sentence-level target token length，
不要直接沿用 T0 的 `max_new_tokens`。

---

## 6. Inference

对每篇 abstract 的每个 sentence 独立预测：

```text
Full Abstract + S1 → Events(S1)
Full Abstract + S2 → Events(S2)
...
```

各 sentence prediction 之间不传递前面生成的 tree。

最后：

```text
Document Events
=
Union(Events(S1), ..., Events(Sn))
```

只做 deterministic merge。

第一轮不做：

```text
document-level reconciliation
event-type verifier
Stage-1 hints
boundary refinement
constrained decoding
```

---

## 7. Alignment

Trigger：

```text
只允许在 target sentence 中对齐
```

Argument：

```text
允许在整篇 abstract 中对齐
```

这样避免同一个 trigger text 在多个句子出现时错配。

---

## 8. Evaluation

Merge 后使用与 T0 完全相同的 SciEvents evaluation：

```text
TI
TC
AI
AC
EC
AC-P
AC-R
```

同时重点比较：

```text
1. trigger correct + type wrong
2. trigger missing
3. false-positive events
4. argument missing
5. argument boundary error
6. generated target length
```

---

## 9. Main Comparison

只比较：

```text
T0:
Full Abstract → Full Forest
```

vs.

```text
C:
Full Abstract + Target Sentence
→ Local Events
→ Merge
```

核心问题：

> **完整 document context 保留不变，仅把一次性的 document-level decoding 改成 sentence-level decoding，是否能提高性能并减少长序列生成问题？**

---

## 10. How to Interpret

如果：

```text
C > T0
```

说明：

> full-document context 有用，但 full-document autoregressive decoding 可能不是最佳方式。

如果：

```text
生成明显更短，但 TC 仍然低
```

说明下一步再研究：

```text
Type / Trigger / Role / Argument co-refinement
```

如果 C 不提升，也先分析错误，不立即增加新模块。

---

## 11. Code

保留现有 T0 pipeline。

新增：

```text
build_sentence_dataset.py
merge_sentence_predictions.py
```

尽量复用：

```text
train_lora.py
predict.py
parse_tree.py
align_tree.py
evaluate.py
```

先做 gold roundtrip：

```text
gold sentence trees
→ merge
→ official evaluation
```

确认无损后，再开始训练。
