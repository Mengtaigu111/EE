# Codex Task: Stage-1 Typed Mention Extraction with Qwen3-8B

本文件是给 Codex 的实现规格。目标是用一组可控实验回答一个问题：

> 用 Qwen3-8B 直接从句子抽取带 nugget type 的 mention，能否在 dev 上同时做到召回不低于现有工具候选池、精度显著高于它；工具候选作为提示信息是否还有价值。

结果决定后续架构：Stage 1（typed mention 抽取）是否单独存在、是否需要过生成、句法工具是否保留在 Stage 1。本轮**不做** Stage 2（trigger、event type、role、树生成）。

---

## 0. 背景与已确认的事实（2026-09-15 诊断，dev split）

这些数字来自 `tasks/phase1-candidates/runs/full-v2` 冻结产物和 `data/SciEvents/dev.json`，实现时不必重算，但报告里对照基线必须在同一分母下重算一次。

| 事实 | 数值 |
| --- | ---: |
| dev 去重 nugget | 6800 |
| 其中只做 trigger / 只做 argument / 两者皆是 | 1655 / 4396 / 749 |
| 749 个“皆是”里属于 E-PUR / E-FAC / E-CMP（子事件论元，本身是 trigger） | 615 |
| Stage-1 目标 span（非 E-* 的 argument 位置集合） | 4565（train 36665） |
| 同一 span 带多个 nugget type | 21（0.46%），train 91（0.25%） |
| 四工具候选池在 argument-only 目标上的 P / R | 9.5% / 79.9% |
| 工具候选按类型召回最低的三类 | DEG 49.5%，WEA 67.1%，TAK 72.6% |
| 没有任何 gold 的句子 | 256 / 1593（16.1%），其中工具候选 5514 个 |
| 同文档内同字符串一处是 gold、另一处未标注 | dev 132，train 1104（占多 token gold 字符串 4.5%） |
| 跨句 nugget | 66（全部 nugget 口径） |

结论：nugget 是“参与了事件的 typed mention”，不是独立实体层；trigger 容易（89% 单 token，工具召回 91%），难点全在 argument 的语义类别和边界。因此 Stage 1 只抽 10 类 typed mention，trigger 和 E-* 交给 Stage 2。

---

## 1. 任务定义

### 1.1 输入输出

```text
输入：一个数据句（与工具缓存一致的 source-token 形式，空格分词），可选整篇 abstract 作上下文
输出：该句内所有 Stage-1 mention，每个带 nugget type
```

输出 JSON（assistant 内容为纯 JSON，无 markdown 围栏）：

```json
{"mentions":[{"text":"reliable training corpus","type":"DST"},{"text":"crux","type":"TAK"}]}
```

### 1.2 类型集合

10 个 nugget type，定义按论文 Appendix F.1（`Lu 等 - 2026 - EXCEEDS ...pdf` 第 15 到 16 页）：

```text
OG   Research Organization/Group   we; Li et al., 2013; they
APP  Approach                      完整方法/算法：... model; ... method; ... framework; baselines; state-of-the-art
MOD  Module                        方法内部组件：... encoder; ... decoder; ... module; beam search
FEA  Feature                       特征：... information; the first-order adjacency information
TAK  Task                          任务目标/研究关注点/指标：graph-to-sequence modeling; accuracy; F1 score; robustness
DST  Dataset                       TAC-KBP 2017 datasets; training data
LIM  Limit                         条件/环境限制，常带介词：for a small number of confusing type pairs
STR  Strength                      优点评价：state-of-the-art performance
WEA  Weakness                      缺点评价：biases and failure cases of beam search; cannot handle
DEG  Degree                        程度/数量：only; not fully; 1.5%; significantly
```

**排除**：`E-PUR`、`E-FAC`、`E-CMP` 三类（它们是子事件的 trigger）；所有 trigger span。若一个 span 既是 trigger 又带非 E-* 类型（dev 约 134 个，多为 STR/WEA），保留为 Stage-1 目标。

### 1.3 边界与身份口径（沿用项目既定规则）

- gold 身份 = `document_id + 文档内 token 位置集合`，去重；同字符串不同位置分别保留。
- 一个 span 多个类型：目标里每个类型各输出一条（同 text 不同 type）；评估时 typed 匹配要求预测 type ∈ gold 类型集合。
- 不连续 nugget：按位置顺序拼接选中 token，不填 gap（与 `tasks/llm-sft` 一致）。
- 跨句 nugget：句级样本不作目标；评估分母分两套报告（见 §7.3）。
- 空目标句保留，目标为 `{"mentions":[]}`。
- 输出顺序：按 start offset、end offset、annotation order。

---

## 2. 数据构建

### 2.1 来源（只读）

```text
data/SciEvents/{train,dev,test}.json              nugget_type、offsets、events
tasks/llm-sft/normalized/{split}.documents.jsonl   与工具缓存一致的 source tokens
tasks/llm-sft/normalized/{split}.gold.jsonl        已去重 gold（无类型），用于核对位置口径
tasks/phase1-candidates/runs/full-v2/candidates/candidates.jsonl     dev 候选（E2 hints 用）
tasks/train-three-tools/runs/v1/...                                  train 候选缓存（E2 hints 用）
tasks/test-candidates-recoverability/runs/v1/...                     test 候选缓存（E2 hints 用）
```

train/test 候选缓存的确切文件路径以各自 `verification.json` 记录为准，脚本里通过参数传入，不硬编码。

### 2.2 样本粒度与三种输入变体

一个样本 = 一个数据句。三种 user 输入：

```text
S     只有目标句
S+D   整篇 abstract（各句按原顺序拼接），目标句用 <<< >>> 标出，只输出目标句的 mention
S+D+H 在 S+D 基础上加一行 Hints（见 §4）
```

S+D 是主设置。理由：事件依赖和回指（our framework、this model）需要文档上下文；文档级整体输出目标过长且不便于逐句对照。

### 2.3 目标标签的两个版本

```text
G     gold 原样：该句内全部 Stage-1 gold span 及类型
G+P   过生成版本：G ∪ 同文档传播伪标签
```

同文档传播规则（确定性、不读模型）：

1. 取该文档所有连续、长度 ≥ 2 token 的 Stage-1 gold 字符串（小写比较）及其类型。
2. 在该文档全文中查找同字符串的其他出现位置（连续精确匹配）。
3. 若该位置集合不是 gold，也不是任何 gold 位置集合的子集，则作为伪标签加入，类型继承。
4. 单 token 字符串（we、it、model）不传播。
5. 伪标签在 processed sidecar 里标记 `label_source=propagated`，训练文件里不区分。

预期 train 新增约 1104 条。**不做**跨文档 train 词典传播（噪声大，作为后续可选项）。

### 2.4 输出目录

```text
tasks/stage1-typed-mentions/
  raw/            输入文件引用与 SHA256
  processed/      每样本 sidecar：document_id、sentence_id、offsets、types、label_source
  data/
    S_G/  SD_G/  SDH_G/  SD_GP/        各含 train.jsonl dev.jsonl test.jsonl（ChatML）
  prompts/        system prompt、few-shot 示例（固定文件，带版本号）
  scripts/
  runs/<attempt>/ 见 §8
  report.md
```

ChatML 格式与 `tasks/llm-sft` 一致：每行 `{"messages":[system,user,assistant]}`。

---

## 3. Prompt

### 3.1 System（G 版本）

```text
You are an expert in scientific information extraction.

Given one sentence from a scientific abstract (with the full abstract as context),
extract every mention in the target sentence that participates in a scientific event
described by the abstract (research purpose, proposed or adopted approaches, method
steps, related-work statements, experimental findings and comparisons).

Each mention must be labeled with exactly one type:
OG: a research team or organization, e.g. "we", "Li et al., 2013", "they".
APP: a complete method or algorithm with concrete inputs and outputs, e.g. "... model", "... method", "... framework", "baselines".
MOD: a component of a method or architecture, e.g. "... encoder", "... module", "beam search".
FEA: a feature used by an approach, e.g. "... information", "positional information".
TAK: a task, research focus, objective or evaluation metric, e.g. "graph-to-sequence modeling", "accuracy", "F1 score".
DST: a dataset or training data, e.g. "TAC-KBP 2017 datasets", "training data".
LIM: a conditional or environmental limitation, often a prepositional phrase, e.g. "for a small number of confusing type pairs".
STR: an evaluative phrase describing an advantage, e.g. "state-of-the-art performance".
WEA: an evaluative phrase describing a shortcoming, e.g. "biases and failure cases of beam search".
DEG: an expression of degree or quantity, e.g. "only", "not fully", "1.5%", "significantly".

Do not extract event predicates (verbs such as "propose", "show", "evaluate").
Copy the mention text exactly as it appears in the sentence. Return only:
{"mentions":[{"text":"...","type":"..."}]}
```

G+P 版本把第一段改为 “extract every mention of the following types in the target sentence, whether or not it participates in an event”。

### 3.2 User

```text
Abstract:
<s1> <s2> <<< target sentence >>> <s4> ...

Target sentence:
<target sentence>
```

S 变体只保留 Target sentence 段。S+D+H 在末尾追加：

```text
Hints (noisy candidate phrases from a syntactic parser; some are wrong, some mentions are missing):
our method | training data | the highly compressed word embeddings | ...
```

### 3.3 Few-shot（仅 E0）

从 train 固定选 4 个句子，覆盖 OG/APP/TAK/DST/LIM/DEG/WEA，写入 `prompts/fewshot_v1.json`，包含 doc_id 与 sentence_id 以便复核。SFT 实验不用 few-shot。

---

## 4. 句法工具在本轮的使用范围

只在 **E2** 中作为 Hints 使用；其余实验不接触工具输出。AMR 本轮不使用。

Hints 生成规则（从缓存 `candidates.jsonl` 的 evidence label 读取，不读 gold）：

```text
保留：目标句内的候选，且满足任一
  C:NP 或 C:NP/derived，长度 ≤ 6 token
  C:PP，长度 ≤ 8 token            （LIM 的主要来源：dev PP 命中 306 个 gold 里 298 个是 LIM）
  C:ADJP，长度 ≤ 4 token
  R:nominal_coordination_member/derived
丢弃：C:VP、C:SBAR、所有 S:* 论元跨度、所有 D:* 谓词、C:NP 长度 > 6
去重：按位置集合；按 start 排序；每句上限 25 条，超出按长度升序截断
```

依据：dev 全池 P 12.9%，其中 VP 1.3%、SBAR 3.5%、长度 ≥ 12 的候选 0.5%；NP 与 PP 是仅有的稠密有效来源。Hints 只给文本，不给类型（工具没有类型信息），不给 ID。

可选 **E2-full**：Hints 改为该句全部候选（不过滤），只为回答“原始候选池是否误导模型”。优先级最低，预算不足可跳过。

---

## 5. 实验矩阵

| 编号 | 输入 | 目标 | 训练 | 目的 | 优先级 |
| --- | --- | --- | --- | --- | --- |
| E0 | S+D，4-shot | 无 | 不训练 | 零训练基线，检验 prompt 与解析器 | 必做，先做 |
| E1 | S+D | G | LoRA | **主实验**：Stage 1 能否单独成立 | 必做 |
| E1-S | S | G | LoRA | 文档上下文是否必要 | 必做 |
| E2 | S+D+H | G | LoRA | 精简工具 Hints 是否有增益 | 必做 |
| E3 | S+D | G+P | LoRA | 过生成能否提高 gold 召回 | 必做 |
| E2-full | S+D+全候选 | G | LoRA | 原始候选池是否有害 | 可选 |
| Tool | 缓存候选 | 无 | 无 | 对照：§4 过滤后候选与全池，同分母 P/R | 必做，纯统计 |

执行顺序：Tool 与 E0 先做（半天内可完成），E1 → E1-S → E2 → E3。

Seed 策略：E0 不涉及；E1、E1-S、E2、E3 先各跑 seed 42 做筛选；在 dev 上选定的最终配置补 seed 43、44，报告均值 ± 标准差。test 只对最终配置的 3 个 seed 各推理一次，结果直接进报告，不再改任何设置。已在候选分析中看过 test，报告里如实注明。

---

## 6. 训练与推理配置

### 6.1 环境约束

- **不得**向 `qwen3meld-run` 安装任何包（历史教训：装包炸过其他训练）。该环境已有 torch 2.4.1+cu121、transformers 4.52.3、peft 0.19.1、accelerate 1.13.0、datasets 2.12.0，足够用 HF `Trainer` + PEFT 实现 LoRA SFT 与 HF `generate` 推理。
- 如需 vLLM 或 LLaMA-Factory，新建独立 conda env（磁盘剩余约 831G），记录 `environment.yml` 与 pip freeze。
- 模型：`/opt/models/Qwen3-8B`，bf16。单卡 RTX 4090 24GB，运行前用 `nvidia-smi` 确认空闲，不沿用历史状态。
- 运行放 tmux，每次 attempt 独立目录、`.running`/`.exit` 文件、源码快照。

### 6.2 LoRA 超参（统一，不为单个实验调参）

```text
target_modules   q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
r / alpha / dropout   16 / 32 / 0.05
lr               1e-4，cosine，warmup ratio 0.03
epochs           3
effective batch  16（per_device 1 或 2，其余用 gradient accumulation 补齐）
max_len          S: 1024；S+D / S+D+H: 1536（超长样本记录数量，不截断 assistant）
precision        bf16；gradient checkpointing 开
loss             只在 assistant token 上计算
chat template    tokenizer.apply_chat_template(..., enable_thinking=False)
seed             42（筛选），43、44（最终配置）
checkpoint       每 epoch 保存 adapter；筛选阶段用最后一个 epoch 推理
```

显存预算：权重约 16.4GB + LoRA 状态 + 激活，len 1536、per_device 1 可在 24GB 内；OOM 时先降 per_device 至 1、再降 max_len 至 1024，并在日志记录。预计每个 S+D 运行 5 到 7 小时，S 运行更短；预算总计约 6 个 LoRA 运行（含补 seed）。

### 6.3 推理

```text
greedy decoding，max_new_tokens 400，batch 8，left padding
enable_thinking=False
输出解析：严格 json.loads；失败则提取首个 {...} 块重试；仍失败记 invalid，样本预测计空
记录：invalid 数量、每样本原始输出字符串、解析后 mentions
```

---

## 7. 评估

### 7.1 预测到位置的对齐（确定性）

对每条预测 `text`：

1. 按空格切分为 token 序列。
2. 在目标句 token 序列中找**连续**出现：先大小写敏感，再不敏感。多个出现时取最左且未被同文本前序预测占用的位置；都被占用则取最左并记 `duplicate`。
3. 无连续匹配时，在目标句内做最左**子序列**匹配（不连续 nugget），记 `discontinuous_alignment`。
4. 仍无匹配：记 `unaligned`，计为 FP，单独统计比例。
5. 句内位置 + 句首全局 offset → 文档位置集合。

对齐器必须有单元测试：连续、同句重复字符串、不连续、无法对齐、大小写差异五类。

### 7.2 指标

- **Untyped identification**：位置集合精确匹配 Stage-1 gold。P / R / F1。
- **Typed identification**：位置精确匹配且 type ∈ gold 类型集合。P / R / F1。
- **Per-type recall**：按 gold 类型分 10 行，untyped 与 typed 各一列。这是最重要的诊断表。
- **Trigger contamination**：预测命中 trigger-only gold span 的数量（Stage 1 定义下算 FP，单列报告）。
- **Invalid / unaligned / duplicate** 比例。

### 7.3 两套分母

```text
D-sent   Stage-1 gold 中完整落在单句内的 span（句级可达上限）
D-all    全部 Stage-1 gold，含跨句 span（跨句必然漏检，如实计入）
```

两套都报；正文以 D-all 为主，D-sent 用于解释差距。

### 7.4 错误分类（每条 FN / FP 一个主类别，写入 errors.jsonl）

```text
FN
  BOUNDARY        存在预测与该 gold 同句、Jaccard ≥ 0.5 且不精确
  TYPE_ONLY       位置精确、类型错（untyped 视作 TP）
  MISSING         无任何位置重叠的预测
FP
  BOUNDARY        与某 gold 位置重叠
  TYPED_KNOWN     无重叠，但文本等于同文档其他位置的 gold 字符串，或出现在 train Stage-1 gold 词典中
  OTHER           其余
```

`TYPED_KNOWN` 的占比直接检验“精度损失主要来自事件依赖”这一假设。

### 7.5 对照基线（同一评估脚本、同一分母）

```text
Tool-full      该句全部缓存候选作为预测（无类型），untyped P / R
Tool-filtered  §4 过滤后的 Hints 作为预测，untyped P / R
```

预期 Tool-full 在 D-all 上约 P 9% / R 78%。E1 至少要在 R 不低于 Tool-full 的前提下把 P 提到 50% 以上，Stage 1 才算成立。

---

## 8. 交付物与目录规范

```text
tasks/stage1-typed-mentions/
  README.md                  复现命令、环境、口径
  scripts/
    build_dataset.py         §2，支持 --variant {S,SD,SDH} --labels {G,GP} --splits
    build_hints.py           §4，只读候选缓存，不读 gold；输出每句 hints sidecar
    train_lora.py            §6.2
    predict.py               §6.3
    evaluate.py              §7，输入 predictions.jsonl + 原始 SciEvents json，不依赖训练中间件
    tool_baseline.py         §7.5
    verify_run.py            独立重算 metrics.json 并比对；核对输入 SHA256
    run_experiment.sh        单实验入口：训练 → 推理 → 评估 → 验证，写 attempt 日志与 exit 码
  tests/
    test_aligner.py  test_dataset_rules.py  test_hints_no_gold.py
  runs/<experiment>-<seed>-<timestamp>/
    config.json  source-snapshot/  train.log  adapter/  predictions.dev.jsonl
    metrics.json  per_type.csv  errors.jsonl  verification.json  .exit
  report.md
```

`report.md` 必含表：

1. 主表：Tool-full、Tool-filtered、E0、E1、E1-S、E2、E3 在 dev D-all 上的 untyped / typed P R F1，invalid 率。
2. Per-type recall 表：E1 与 E3 各 10 行，附 Tool-full 对照列。
3. 错误分类表：E1 的 FN / FP 六类计数与比例。
4. 最终配置 3 seed 的 dev 与 test 结果（均值 ± 标准差）。
5. 训练/推理耗时、显存峰值、超长样本数。

每张表附带对应文件路径，逐例证据在 `errors.jsonl`。报告中的例子必须给完整原句和 gold 类型，不用省略号节选。

---

## 9. 验收标准

- `build_hints.py` 与 `build_dataset.py` 的候选/提示路径不读取任何 gold 字段，用测试证明（构造无 gold 输入仍能产出相同 hints）。
- `evaluate.py` 只依赖 predictions 与原始 `data/SciEvents/*.json`；`verify_run.py` 独立重算并逐字段比对。
- 对齐器五类单测通过；数据规则测试覆盖多类型 span、不连续 span、跨句排除、空句保留、传播伪标签不覆盖 gold。
- 每个 run 有 `.exit`（0 为成功）、源码快照、输入 SHA256；失败尝试保留不删。
- 原始数据、工具缓存、`tasks/llm-sft` 产物全部只读。
- test 只在最终配置上推理，报告注明 test 曾在候选分析中被查看。

---

## 10. 结果到决策的映射

| 观察 | 结论 | 下一步 |
| --- | --- | --- |
| E1 untyped R ≥ Tool-full R 且 P ≥ 50% | Stage 1 单独成立 | 进入 Stage 2 规划，Stage 1 不用工具 |
| E2 − E1 ≥ 1.0 F1（3 seed 均值） | 精简 Hints 有效 | Stage 1 保留 NP/PP hints，其他工具输出退役 |
| E2 ≤ E1 | 工具在 Stage 1 无价值 | 工具只保留给 Stage 2 做边界后处理或不用 |
| E1 FP 中 TYPED_KNOWN ≥ 40% | 精度损失主要是事件依赖 | 采用 E3 口径作 Stage-1 目标，参与判断交给 Stage 2 |
| E3 gold R − E1 gold R ≥ 3 点，且 E3 新增预测中 TYPED_KNOWN + BOUNDARY ≥ 90% | 过生成可控 | Stage 1 改为过生成 |
| E1 FN 以 MISSING 为主且集中在 DEG/WEA/STR | 语义类别是瓶颈 | 为这三类补 prompt 示例、上采样；考虑 DEG 词性规则后处理 |
| E1 FN 以 BOUNDARY 为主 | 边界约定是瓶颈 | 检查各类型边界规范（冠词、介词），考虑 NP 边界后处理 |
| E1-S 与 E1 差距 < 1 F1 | 文档上下文非必要 | 后续用 S 变体省算力 |

---

## 11. 本轮明确不做

- 不训练或设计 Stage 2（trigger、event type、role、sub-event、树生成）。
- 不使用 AMR；不修复 AMR 对齐 bug。
- 不把 `tasks/llm-sft` 的 A/B 四版数据直接拿来训练（其目标是无类型 nugget，含 trigger 与全量候选证据，与本轮任务定义不同）。
- 不做跨文档词典传播、不做 LLM 补标（留作 E3 之后的可选项）。
- 不在 test 上做任何选择。
- 不改 `tasks/todo.md` 以外的状态文件；执行时按项目惯例新建 `goal-11/`，并在 `tasks/todo.md` 登记为当前任务。
