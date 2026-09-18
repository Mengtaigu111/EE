# 当前任务：Stage-1 Typed Mention Extraction

唯一状态源。[规格](../goal-11/spec.md) · [计划](../goal-11/plan.md) · [日志](../goal-11/experiment_log.md)

- [x] I1 数据/环境/计划核验
- [x] I2 S_G / SD_G 数据与验证
- [x] I3 对齐/解析/评测与测试
- [x] I4 LoRA/推理/队列与真实smoke：训练、adapter保存、推理、评测均已实跑；29项测试通过。
- [x] I5 tmux E0→E1→E1-S→E1 seeds43/44（E1、E1-S及三 seed均完成）
- [x] I6 独立验收与完整report：四个正式 run 的 dev/test 指标、三 seed汇总和Q1-Q3已写入报告。

## 结果复盘
全部split长度无超限，未截断gold。E0、smoke、E1、E1-S及E1三seed均已完成；队列状态为 completed，最终报告已生成。[历史状态](../goal-11/previous-task-todo.md)。

---

# 当前会话：Stage-2 Event Tree 实验理解与结果核对

目标：弄清 Stage-2 的建模对象、T0–T5 的实际代码差异、评测链路，以及当前结果能支持和不能支持的结论。

## 计划与检查项

- [x] 阅读项目教训、Stage-2 README、实验规格和权威报告
- [x] 核对数据构造、prompt、线性化、对齐、官方评分及队列状态
- [x] 对照 Stage-1 报告，确认普通 hints 与 oracle 的含义
- [x] 运行 Stage-2 单元测试并做报告/产物一致性核对：16 项通过；主结果 JSON 与报告一致
- [x] 将理解结果和当前异常/边界归档

## 结果复盘

Stage-2 是文档级 event-forest 生成：Qwen3-8B LoRA 先生成 S-expression 树，再由确定性文本对齐器回到 SciEvents token offsets，最后调用 EXCEEDS 官方 `utils.calculate_F1`。16 项单元测试和 dev gold roundtrip（TI 99.16 / TC 98.58 / AI 94.80 / AC 94.77 / EC 97.23）通过。

- T0 是“有 event schema、无 hints”的真正基线；test AC 39.88。
- T2 普通 hints 使用 Stage-1 E1-42 的真实预测，训练阶段使用按 Stage-1 dev 误差画像生成的伪噪声 hints；test AC 39.21，比 T0 低 0.66。
- T2 oracle 只把推理输入替换成 gold typed mentions，adapter 不变；test AC 56.46，比普通 T2 高 17.25，比 T0 高 16.58。因此它是“理想 mention guidance + 当前 Stage-2”的上界诊断，不是可部署结果，也不能把全部差距严格归因于 Stage-1 单一因素。
- T3 是无 hints 的自然语言事件标签消融（代码把事件码替换为 Proposal/Comparison 等），不是在 T2 上追加自然语言标签；test AC 39.27。
- T4 先用 2 epoch 平铺 trigger/argument 子结构，再用 5 epoch full tree 继续训练；test AC 37.53。
- T5 是无 schema、也无 hints 的对照；test AC 38.78。故 schema 相对 T0 的收益约为 1.10 AC 点，但在 T2 中被低质量 hints 抵消。
- 所有正式模型 invalid=0、触顶截断仅 T5 test 1 篇，未对齐数量也很小；主要问题不是树语法或 parser 崩溃，而是事件/trigger/argument 的联合识别、类型和归属。
- `T1` 正式 run 在 train 阶段以 exit -15 失败，没有有效完整指标；不能纳入结论。
- `tasks/stage2-event-tree/report.md` 对应 2026-09-17 两个队列，后一个队列 completed。工作区另有 2026-09-18 的 `V2-T2warm-*` 手工 warm-start 目录，但只有训练前三步、没有 adapter/prediction/metrics/completed 标记，不能并入 stage2b 报告。

代码归档：已生成合并包 [stage1-stage2-code-20260918.zip](../stage1-stage2-code-20260918.zip)，包含 Stage-1/Stage-2 scripts、tests、README、两份实验规格、ontology 和官方评测依赖源码，共 40 个文件；未包含数据集、模型、checkpoint、训练日志或缓存。zip 完整性检查通过，SHA256=`2d00eaf3d79c036ea507657c20c8f3defc42a3ae9485252fd11c07748fa971eb`。此前的 Stage-2 单独包仍保留。

本次对照 `task/0917-stage2-event-tree-v2.md` 后确认：stage2b 不是按该 v2 规格原样执行，而是执行了一个后续扩展协议。原规格的 T0/T1/T2 语义与实际 run 不一致：原规格 T0=Base+无 schema、T1=Base+schema、T2=Stage-1 warm start+schema；实际 T0=Base+schema、T1=Base+flat predicted hints（且 train 以 -15 失败）、T2=Base+typed predicted hints/schema（不是 warm start）。实际新增了 T3/T4/T5，并使用 5 epochs，而原规格要求首轮 3 epochs、先筛选 T0/T1/T2、再补 seed 43/44。故现有报告回答的是“schema / 显式 hints / oracle / natural labels / curriculum / no-schema”的扩展问题，不能当作原规格 Q3（schema）和 Q4（Stage-1 warm-start）的严格实验结论；尤其没有已完成的 Base vs Stage-1 warm-start 对照。

Experiment C 评估：方向值得做，能直接测试“文档级输入、句子级 decoding”是否缓解长 forest 的结构竞争；但正式运行前必须控制训练预算和 checkpoint 选择。C 的 train rows 会从 2,007 篇文档扩大到 12,828 个句子，固定 5 epochs 不再等价于 T0；并且 T0 当前没有触顶截断（test hit_max_new_tokens=0），所以 C 若提升，结论应是减少 autoregressive 结构竞争/错误传播，而不是解决长度截断。数据上约 15.1% 的 test events 有跨句 argument，约 16.7% 的 test sentences 无 event，必须在 gold roundtrip、merge 去重、局部 trigger/全局 argument 对齐和空句/无事件输出中逐项验收。C 与 T0 还要统一 final checkpoint vs dev-best 规则，并报告 1,627 次 sentence inference 相对 250 次 document inference 的总 token、耗时和显存成本。

验证记录：一次 `jq` 核对因环境未安装 jq 失败，已改用当前 conda 环境的只读 JSON 校验完成；没有改写实验产物。

---

# 当前会话：Experiment C 句子级 Event Generation 实现

目标：按 [task/0918-experiment-C-short.md](../task/0918-experiment-C-short.md) 实现 Full Abstract + Target Sentence 的训练、句级对齐、文档合并和官方评测入口；保持 T0 行为不变。

## 计划与检查项

- [ ] 写入并核对实现计划
- [ ] 句级数据构造：每句一行、target-only events、全文 context、无事件 `(ROOT)`
- [ ] 句级 trigger scope + 全文 argument alignment
- [ ] deterministic merge 与预对齐评测入口
- [ ] C 队列：长度审计、显式 step budget、dev-best checkpoint、test 评测
- [ ] gold sentence trees → merge → official scorer
- [ ] 最小真实数据 build/audit smoke；本轮不启动正式 8B 训练

## 设计决策

- C 使用独立脚本，不改变 T0 的文档级数据和默认对齐行为。
- merge 后输出 document-level aligned predictions，再由官方 `utils.calculate_F1` 评测。
- target sentence 只约束 trigger；argument 在全文范围内确定性对齐。
