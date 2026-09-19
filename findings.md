# EXCEEDS：稳定事实与结果索引

当前工作状态仅维护在 [tasks/todo.md](tasks/todo.md)；本文件保留数据口径、方法边界与权威报告入口。

## 项目与数据

EXCEEDS 面向科学摘要的复杂事件抽取，主模型使用 RoBERTa、BiLSTM 和词对网格；代码分工与原始任务见 [README](README.md)。当前 nugget 研究关注 Document → 去重边界集合，与事件类型、角色、归属和层级分开。

SciEvents 的原始句子 token 拼接与 document 一致；nugget 身份为 `(doc_id, sorted unique offsets)`，重复标注不重复计边界。固定数据版本为 `664aa3a61e35dbe1262aa0fbb026d5cecfbf0c66`，来源核验见 [数据/预测审计](tasks/verification.json)。

| split | 文档 | 句子 | unique gold nuggets |
| --- | ---: | ---: | ---: |
| train | 2007 | 12828 | 54489 |
| dev | 251 | 1593 | 6800 |
| test | 250 | 1627 | 6874 |

## 方法与解释边界

- 三工具证据为 Stanza Constituency/Dependency 与 AllenNLP SRL；正式全量 AMR 使用 BART-large，历史 BART-base 探针独立保留。AMR A/B/C、R 来源排除和四种累积视图以 [冻结协议](tasks/amr-all-splits/protocol.md) 为准。
- 精确命中比较实际 token 集合；纯子集 union 是用 gold 选择组件的 oracle，候选 P/R/F1、图可达性和对齐覆盖都不能替代最终 LLM 性能。v1 未运行 LLM、未删除 SRL。
- 原始标签只定义边界及事件身份；句法类别、构式分类、人工样本推断必须标明来源，不能称为全量语义 gold。连续 span/BIO 的表示上限与模型实测分开。
- 原始解码和评分器有共享起点分支、同跨度多标签覆盖及复杂子集过滤等已记录限制。作者单次预测重算、重新训练和全文复现不是同一验收目标；使用历史结果前核对对应版本与口径。
- conda 位于 `/home/myluo/miniconda3`；环境与模型准备依据见 [环境记录](tasks/environment/README.md) 和 [模型清单](tasks/local-model-inventory.md)。运行时重新检查资源，不沿用历史空闲状态。

## 权威结果入口

| 内容 | 报告或验收记录 |
| --- | --- |
| 四工具内部规则与问题分层 | [完整报告与建议](tasks/tool-internals/report.md)、[336组CSV](tasks/tool-internals/groups.csv)、[交叉验收](tasks/tool-internals/verification.json)、[AMR模糊匹配抢占精确匹配复现](tasks/tool-internals/amr/alignment-reproduction.json)；仅缓存诊断，训练数据未改 |
| 删除 Constituency 前的候选消融 | [三组比较与 C-only 报告](tasks/constituency-ablation/results/v1/report.md)、[CSV](tasks/constituency-ablation/results/v1/metrics.csv)、[逐条完整原句](tasks/constituency-ablation/results/v1/constituency_only.jsonl)、[独立验收](goal-9/independent-verification.json)；只读缓存，训练数据未改 |
| 四工具实现与 Constituency 规则噪声 | [工具/规则说明及类别分解](goal-9/extraction-audit/explanation.md)、[三 split NP 与 noun/verb/NP 对照](goal-9/extraction-audit/summary.json)；整路消融不代表全部 C 候选应保留 |
| Semantic Nugget Discovery SFT 四版本数据 | [交付报告](tasks/llm-sft/statistics/report.md)、[脚本与复现](tasks/llm-sft/README.md)、[全量校验](tasks/llm-sft/statistics/validation.json)；仅缓存预处理，未训练 |
| AMR 全量候选对比 v1 | [实验报告](tasks/amr-all-splits/results/v1/report.md)、[原始 AMR 验收](tasks/amr-all-splits/raw-all-splits-verification.json)、[代码与流程复审](tasks/amr-all-splits/review-20260914/review.md) |
| Phase 1 dev 候选 | [报告与复现入口](tasks/phase1-candidates/report.md) |
| Phase 2 dev 漏检可恢复性 | [报告](tasks/phase2-recoverability/recoverability_report.md) |
| test 冻结规则评估 | [dev/test 对照报告](tasks/test-candidates-recoverability/report.md) |
| train 三工具缓存 | [全量验收](tasks/train-three-tools/runs/v1/verification.json)、[禁用推理的缓存复用验收](tasks/train-three-tools/runs/v1/cache-reuse-verification.json) |
| 原始工具展示与 gold 实例 | [三句话原始输出](tasks/three-sentence-raw-tools/raw-outputs.md)、[五例完整原句与 gold](tasks/nugget-discovery/full-sentences-gold.md) |
| Nugget 边界困难 | [几何、句法与构式报告](tasks/nugget-discovery/report.md) |
| 历史重叠与身份统计 | [重叠口径](tasks/nugget-overlap/report.md)、[exact-only 子集](tasks/nugget-overlap/exact-only/report.md) |
| 句级/文档级连接口径 | [句界与跨句统计](tasks/sentence-scope-report.md) |
| dev Dependency/AMR 结构 | [结构覆盖报告](tasks/structure-parsing/analysis_summary.md) |
| 漏检与 AMR 可行性探针 | [TI/AI 双语案例](tasks/dev-ti-ai-false-negatives-zh-en.md)、[AMR 探针](tasks/amr-probe/README.md)、[原始输出中英对照](tasks/amr-probe/comparison-zh-en.md) |
| 模型运行与消融 | [三 seed 结果](tasks/three-seed-ablation-results.md)、[此前两 seed 队列](tasks/cpu-seeds-ablation-results.md)、[padding 修正](tasks/padding-mask-results.md)、[原版 seed52](tasks/full-seed52-results.md)、[运行性 smoke](tasks/smoke-results.md) |
| 原论文复现审计 | [公开资料与限制](tasks/reproduction-evidence.md)、[评分/源码核验](tasks/verification.json) |
| Stage-2 Event Tree 生成 | [README/协议](tasks/stage2-event-tree/README.md)、[完整结果报告](tasks/stage2-event-tree/report.md)；16项单元测试通过，T2 oracle 与普通 hints 的差距是当前最重要诊断，但不能等同于纯 Stage-1 因果增益 |

各报告保留自己的逐例证据、冻结版本、日志与限制，指标不在本文件重复维护。更早的调研过程与当时状态见 [五份文档原件归档](tasks/archive/20260914-active-docs/manifest.json)。
# Experiment C implementation findings (2026-09-18)

- The approved design keeps the existing T0 document-level pipeline unchanged and adds a separate sentence-level path.
- Existing `align_forest` searches triggers over the whole document; C requires per-event trigger scopes restricted to the target sentence.
- Existing argument alignment can search the document but prefers the trigger sentence; C needs an explicit document-wide mode so cross-sentence arguments are represented deterministically.
- Existing `predict.py` hard-codes `MAX_NEW_TOKENS = 768`; C must pass a budget derived from sentence-level token audit.
- Existing `evaluate.py` realigns parsed events using document-level defaults; C merge output therefore needs an aligned-prediction path or it would lose target-sentence trigger restrictions.
- Formal C training is intentionally not started during this implementation pass; the first acceptance gate is gold sentence-tree roundtrip through the official scorer.

## Stage-3 v1 implementation readiness (2026-09-18)

- No existing `tasks/stage3-tree-refinement/` implementation was found; the new work is isolated from Stage-2.
- The provided v1 specification is the approved design. Its required helpers map cleanly to the existing Stage-2 parser, aligner, ontology, model utilities, and official scorer.
- Current C isolation is still active: its queue launches Stage-2 subprocesses from the shared filesystem, so Stage-2 scripts must remain unchanged until C completes.
- Confidence check: duplicate implementation absent; architecture compliant; Transformers generation/model-output APIs and PyTorch loss APIs verified against official documentation; existing repository model-loading checks provide the local reference; root cause and acceptance boundaries are explicit. Confidence: 1.00.
- Stage-3 initially resolved the generic module name `run_queue` to Stage-2 because the compatibility path was inserted first; appending the Stage-2 path preserves the isolated Stage-3 import namespace while keeping explicit Stage-2 helper imports available. The full Stage-3 suite then passed 38 tests.

## Stage-3 v1 explanation findings (2026-09-19)

- Stage-3 v1 is a wrapper around the frozen Stage-2 T0 generator: it expands alternative event candidates, delays/competes event types, and learns a whole-tree candidate scorer. It does not alter Stage-2 scripts and explicitly excludes the generative R2 tree refiner.
- D1 dev completed with 251 documents. Greedy trigger recall was 73.96%; the support≥1 pool reached 85.69% (+11.73 points), with 5.97 unique forests per document on average. The raw pool's AC was only 19.69%, so recall expansion alone is not a result; R0 filtering is necessary. The candidate oracle reached AC 62.13%.
- Experiment C is now completed according to `tasks/stage2-event-tree/runs/C-42-20260918/run.json`, despite the older results summary saying it was still running. Its recorded test metrics are TI 74.09 / TC 62.55 / AI 43.03 / AC 40.75 / EC 45.37.
- D2 all-mode dev is still running in tmux. An earlier attempt hit CUDA OOM during teacher-forced scoring; the current rerun had completed 69/251 documents when checked. No D2/R1/R0 conclusion is valid yet.
- The fixed qwen3meld-run environment passes all 78 Stage-3 tests. The current implementation specifically audits assistant-side forced prefixes, disallows silent truncation, records provenance/attempt state, and keeps reward training's fp32 master weights under bf16 autocast.
