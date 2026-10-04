# 独立评测修复审查

审查范围：本目录 plan.md、instance_scorer.py、run_evaluation_audit.py 及测试；仅 CPU，不修改官方 utils.py、gold、历史 run。独立审查新增 verify_repair_results.py 与本报告。

## 已通过的证据

- 18 个现有 unit tests 全部通过（qwen3meld-run Python，CUDA_VISIBLE_DEVICES=''）。
- 独立穷举 oracle：固定 seed 4281，500 个随机 gold/pred 文档对，各以 offset/token 验证，总 1000 次。oracle 枚举每个 (type, trigger) 组的所有一对一配对并比较 (AC, AI, EC_C, EC_I) 总量，不调用 scorer 的 prepare/assignment/pair helpers。结果完全一致。
- 同批验证事件/论元排列不变性、匹配事件不重复复用、六项 TP/FP/FN 非负及全部 18 项 P/R/F1 由 counts 正确导出。
- dev/test × offset/token 原 gold 自评，全部 18 项均为 1。
- 静态审查确认：TI/TC multiset 保留实例；论元 typed/untyped 分别 multiset，EC 要求论元表达在对应侧 trigger 集合中，TP 两侧都符合；组内全局最大权配对优先 AC→AI→EC_C→EC_I。输入原始 offsets 转为 ordered tuple，未排序/去重。排序签名后 scipy 处理平局，完全相同实例的 index 配对可互换，不改变指标。
- 新评分保留官方定义中“父事件 type/trigger 必须匹配”的论元条件，且分数标注独立诊断口径，未冒称官方 benchmark。

## 需明确的边界

1. runner 的 residual_text_matches 在 exact-offset 配对后再配剩余文本，等式 token_TP−offset_TP=residual_count 需要“相同 offsets 必须有相同 token expression”。真实 aligned 输出若由文档 tokens 构造可满足，但当前函数未显式校验该前提。现 test 的 report/other 同 offset fixture 本身不满足：offset TP=3，token TP=2，residual=1；test 仅验证 residual 列表，没有验证净差值。已通知主代理并修复：runner 对原 gold/aligned 输出的全部 mentions 校验 tokens==document[offsets]，offsets 非空、整数且合法；真实全量数据通过此闸门。test_grounding_mismatch_is_rejected 覆盖前提失败。诊断等式现在仅在校验通过后使用。
2. canonical_events 的 offset 排序/去重丢失 reverse/nonmonotonic 序列是另一个表示缺陷，评分器修复不能恢复丢失信息。gold roundtrip 报告确认 95 处 argument 序列表示损失；任何表示修复必须独立保留原序列，而不能暗改 matching 为 sorted span。
3. EC 保留表达成员定义，同文异位 token 模式不能证明事件边/实例身份；AC 优先的同 TC 组单次配对可能牺牲 AI 最优值，这是明示的稳定联合评估规则。
4. official token 模式优先 text 字符串；新模式优先精确 tokens tuple、缺失时 text.split()。真实数据字段一致时无差异，不保证任意不一致 text/tokens fixture 官方完全一致。

5. 原 runner 用 `{**historical, **locks}` 合并校验可覆盖历史 expected hash，削弱“不修改历史文件”闸门。已修复：runner 入口及出口独立 verify_hash_sets，两份分别逐项校验；test_historical_hash_cannot_be_overwritten 覆盖覆盖攻击。独立脚本亦分别验证两份且不合并覆盖。

## 全量结果复核

evaluation-results.json 产出后，独立 verify_repair_results.py 实际运行 exit 0：

- 19 组 dev/test checkpoint 从原 sentence predictions 重新 parse/merge 后评分。Baseline/Sorted1 各 5 dev+1 test，Easy2 6 dev+1 test（另有 step814）。全部 official_offset、official_aligned_token、official_raw_token 18 项以及 instance 三组 18 项/counts 均精确一致。
- 6 份 test raw/aligned 文档 artifacts 覆盖与排序正确，独立重算旧官方与实例分数一致；全部 TI 残差记录满足 offset 不同、残差总数等于固定预测集合 tokenTP−offsetTP。
- 73 个历史/当前源 hash 逐项验证通过，历史 expected 不被当前值覆盖。
- 各组 dev AC 最优 step 按声明的 tie policy 复算一致；未从 test 选模。
- 同次继续复验 1000 次独立穷举 oracle、dev/test 两模式 gold-self 全部18项100%。验证输出中 all_18_records_original_sentence_predictions_rebuilt 是早期字段名，实际 records=19，全部19条均重构核验。

## Lossless positioned forest 独立审查

lossless_forest.py 与其11项 tests 静态/CPU运行通过，无阻断问题。明确保留 event/argument 原顺序、重复 event/role、offset 原顺序与重复indices；JSON字符串escaping正确；At使用document-global indices并在source token上严格校验，无text-only猜测fallback。对dev/test全部原gold encode→decode，offsets/tokens/type/roles与实例数量逐项一致，18项实例评测两种mode均100%。注释清楚限定nugget metadata不属于该协议；未修改历史schema或实际模型输出，不能将此诊断自评当成旧模型位置信息恢复。

## Ambiguity resolver 独立审查

ambiguity_resolver.py 最终审查无阻断问题。所有严格递增 exact-token subsequences 同等枚举，不因 contiguous 命中忽略 discontinuous；explicit offsets 保留逆序与重复。scope 为 global half-open bounds，latest-suffix pruning 保持完整候选集合；迭代栈不受 Python 递归深度限制。候选超过上限时需要实际观察第 limit+1 条，status=truncated、offsets=None；ambiguous同样不猜位置。

独立随机 seed7801，3000 次 scoped/limited 小实例与 itertools.combinations 穷举候选完全一致；10000 个重复 source tokens、5000-token query、limit1 在0.018秒返回 truncated 且无猜测。现有长期query1200及dead-suffix fixture通过。审查发现空 explicit=[] 可产生成功status，已TDD修复为 ValueError，普通空text仍unaligned；最终回归 test_empty_explicit_span_is_invalid_but_empty_text_is_unaligned 通过。

最后全目录 CPU unittest discover 实际执行41项、exit0。gold代理报告40/40旧greedy遗漏候选可被新完整枚举覆盖，但这些case全部ambiguous；这里只证明不再遗漏可行candidate，不宣称恢复旧预测位置或提高实际模型分数。

最终审查结论：无阻断问题。官方历史评分口径、gold、历史run文件未改，当前独立评分/新协议/候选诊断均明确限定各自含义。

复现命令：

```bash
CUDA_VISIBLE_DEVICES='' /home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m unittest discover -s tasks/stage2-event-tree/scorer-repair-20261004 -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' /home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/stage2-event-tree/scorer-repair-20261004/verify_repair_results.py
```
