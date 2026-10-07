# 课程学习实验交接索引

截至 2026-10-04：本对话四组主要实验已完成；最近 Sorted1 于 12:17 完成，独立校验通过。这里是简短交接，不代替原始日志。

1. [实验与结果](01-experiments.md)：四组设置、offset/token 分数、原始记录。
2. [数据与评测](02-data-and-evaluation.md)：句子级流程、难度排序、对齐和评分限制。
3. [下一步与约定](03-next-and-constraints.md)：当前结论、尚未执行的诊断、保护旧产物的要求。

4. [修复前评分核查](04-score-consistency-audit.md)：gold自评和实际案例。
5. [评测修复交接](05-evaluation-repair.md)：全部指标的新实例评分、位置歧义及代码入口。

6. [在线dev训练完整方案](06-online-dev-training-plan.md)：最近Sorted1保持prompt/课程，3208步、在线验证/降率/早停。
7. [新窗口执行说明](07-online-dev-execution-prompt.md)：可直接复制的实施指令。

8. [方案独立审查](08-online-dev-plan-review.md)：Approved；来源与步数核验见online-dev-plan-verification.json。

2026-10-05状态：新在线dev实验仅方案已写，代码未实现、训练未启动。新窗口实施先读06，再按07执行。

2026-10-05 实施更新：[09 在线 dev 实现与启动](09-online-dev-implementation-and-launch.md)。06 的初稿状态“仅方案”已由新代码与冒烟验证取代；正式实验运行中，尚无最终结果。

9. [运行中检查（2026-10-05）](09-online-dev-runtime-check-20261005.md)：OnlineDev已由另一窗口实现并启动；约965步正常，dev401/802已完成。本文此前“仅方案”状态现为历史记录，最新运行状态以09及run.json为准。

2026-10-05 最终结果：在线dev实验已完成、exit0、独立复算通过，best1604、test官方offset AC41.80%。[完整结果报告](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128/report.md)。
