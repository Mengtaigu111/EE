# EXCEEDS 当前任务

更新：2026-10-23。只保留当前主干与下一步；历史过程见[本次归档原文](archive/20261023-todo-cleanup/todo.before.md)。

## 当前主线：B1950 完整恢复

- **tmux**：`ee-stable-B1950-resume60-20261023`。
- **状态**：整理时已核验会话、控制器与 trainer 存活；已完成 CPU 权重准备，仍在等 GPU，尚未实际恢复训练。实时以[资源状态](action-experiment/runs/r0-v2-B-stable-v6-from1490-42-20261021/resource-state.json)和[B 日志](action-experiment/launch-stable-B1490-v6-20261021/B.log)为准，队列 `running` 不等于已上卡。
- **恢复边界**：同一 v6 B run，从完整 periodic1950 恢复 optimizer/RNG、next_group7800、best1200 与早停状态；保留 step1952 磁盘写满失败记录。
- **入口与证据**：[恢复说明](action-experiment/restart-B1950-disk-recovery-20261023/README.md) · [恢复请求](action-experiment/restart-B1950-disk-recovery-20261023/request.json) · [启动核验](action-experiment/restart-B1950-disk-recovery-20261023/launch-command.json) · [队列状态](action-experiment/launch-stable-B1490-v6-20261021/pipeline-state.json)。

## 下一步

- [x] B1950 恢复入口、来源锁及完整状态核验；tmux 已挂载，仅执行这一次显式恢复。
- [ ] 等连续60秒满足 GPU 准入条件，核验恢复日志与 step1950 之后的真实更新。
- [ ] B 结束或停止后，验收退出原因、完整 dev、best/早停状态、checkpoint 与来源；汇总结果，test 不参与选模。
- [ ] 维持既有每2小时只读巡检：仅新异常、阶段完成或需用户操作时通知；失败留证，不自动重启。见[巡检状态](action-experiment/launch-stable-B1490-v6-20261021/monitoring/state.json)。

## 执行约束

- GPU 加载前连续60秒无其他计算进程且 free ≥21000MiB；占用、显存不足或查询失败重置计时；等待不预留显存。
- 单模型、单巡检；A 保持停止，不启动旧 B 或第二个模型，不自动重试后续故障。
- 保持训练 G4×accum4、FP32 主体算术/MATH attention、原概率保护；dev 有效 batch 上限2，仅 CUDA OOM 降至1，生成预算768，使用 expandable allocator。
- RL 最多3207次更新，完整 dev 每400步；TC 为早停主指标，相对 anchor 至少 +0.001 才重置，连续3次无有效提升即停；TI 仅作 TC 平分时的选模排序。
- 同一 run 恢复继承 optimizer/RNG、数据游标与早停；旧源码、来源锁、checkpoint 和失败证据保留。adapter-only 接续不能称为完整恢复。

## 方法与已确认结论

- 主线：Qwen3-8B，Type-CoT → Action SFT → GRPO；A/B 共同原始初始化为 `action-sft-v2-42-20261012/best-adapter`（step1604）。A 使用四字段 `r0_v2` 奖励、回答 token 全部参与策略 loss；B 增加 `field_v2` 选择性更新。基座/reference 冻结，policy LoRA 可训练。
- 数据：train/dev/test = 12828/1593/1627，test 为250篇摘要。汇报 TI/TC/AI/AC/EC（EC_C）；官方 offset、树文本 presence 等口径分别标明，不混用结果。
- A 在 step1775 概率保护退出，完整快照保留；不得写为完成训练或 dev2000。历史 A800 接续曾重置 optimizer/RNG，A/B 比较存在混杂，不能直接归因字段 mask。
- 已用原 A/B 权重受控重放定位共享 BF16 路径的形状相关数值差异；v6 统一 FP32 算术修复，概率保护未放宽。见[根因与验证边界](action-experiment/diagnosis-AB-20261021/root-cause-report.md)。
- 历史 dev1600 OOM 未完全复现，根因仍未完全证实；最新授权为降低 dev 显存后继续，诊断不再阻塞恢复。见[OOM 调查](action-experiment/diagnosis-dev1600-oom-20261022/report.md)与[真实性审查](action-experiment/review-stability-20261022/report.md)。

## 结果与历史入口

| 内容 | 入口 / 边界 |
| --- | --- |
| CoT、Action SFT、旧 R0 全过程与 test | [seed42 实验记录](../task/result/20261015_CoT_ActionSFT_R0GRPO_seed42_实验记录.md) |
| 四字段奖励 / 字段更新规格 | [R0-v2 规格](../task/1015codex_r0_v2_four_field_mask.md) · [方法审阅](action-experiment/method-review-20261021/report.md) |
| A800 / A1200 补测及后续 B 队列 | 两组均完成，各1627条预测；[历史队列](action-experiment/launch-test800-test1200-B-after-A1775-20261021/README.md)。旧队列的后续 B 失败不代表补测失败 |
| 较早树模型 / RE / 数据词序修复 | [既有结果索引](../findings.md) · [词序修复](stage2-event-tree/argument-order-repair-20261008/README.md)；详细验收与授权变更见归档 |
| 历史待核验：RE 78失败补跑 | [补跑目录](scievents-re-20261006/runs/failed-rerun-20261008/)：shard2 exit2，最终合并未证实；不自动重跑 |
| 暂停支线 | Stage-3 refinement、Nugget E2、AMR P2 等不自动排队；恢复前核对原规格与产物 |
| 操作规则与目录 | [lessons](lessons.md) · [tasks 导航](README.md) |

## 本次精简计划与复盘

- [x] 查阅教训与原文，逐字归档1000行原文并保存 SHA-256。
- [x] 核对最新 tmux、进程、日志与资源状态；合并已完成和已覆盖计划，保留主干、限制及结果入口。
- [x] 核对所有本地链接、关键约束、归档哈希和文档差异。

维护规则：只更新上方状态和待办；详细过程写入对应实验目录，完成后保留结果链接。

复盘：原文1000行压为55行，完整原文 SHA-256 一致，20个本地链接有效；主线与遗留事项独立复核，文档空白检查通过。仅整理 todo 与对应教训，实验进程和产物未改动。

## Git 同步（2026-10-23）

- [x] 核定当前主线源码、测试、规格与文档范围，排除模型、缓存及实时运行产物。
- [x] 运行适用 CPU 回归：405 tests、23 subtests 通过；主线依赖闭包及内容独立复核通过。
- [ ] 提交到当前分支并 push，核对远端提交一致。
