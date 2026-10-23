# `tasks/` 导航

先看 [`todo.md`](todo.md)：它是唯一状态入口；再看 [`../findings.md`](../findings.md) 获取稳定事实和权威结果索引。

| 内容 | 位置 |
| --- | --- |
| 操作规则 | [`lessons.md`](lessons.md) |
| Stage-1/2/3 实验 | `stage1-typed-mentions/`、`stage2-event-tree/`、`stage3-tree-refinement/` |
| 恢复性诊断 | `type-recovery/`、`argument-recovery/`、`joint-type-role-recovery/` |
| Nugget / 工具分析 | `nugget-discovery/`、`nugget-overlap/`、`tool-internals/`、`amr-*`、`structure-parsing/` |
| Trigger / Type 诊断 | 顶层 `trigger-*`、`type-*` 报告及对应生成脚本 |
| 环境与下载记录 | `environment/` 及顶层 `*installation*`、`*verification*` 文件 |
| 历史入口归档 | `archive/` |

顶层 HTML、JSON、日志和脚本是可复现的派生产物；不要把它们重新复制到 `todo.md`。需要新实验时，在对应主题目录创建独立 run/报告，并只把入口链接写入任务清单。
