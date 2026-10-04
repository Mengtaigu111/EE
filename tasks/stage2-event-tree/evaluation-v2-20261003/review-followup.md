**跟进审查结论：**缓存重评现在会核对 raw 输入证据、merged 文件哈希、文档数据哈希、脚本代码哈希和指标文件哈希；test 缓存还要求复杂场景评估退出码为 0，且七个指定分数键齐全。[缓存检查](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/run_reassessment.py:95) 已覆盖 `review.md` 中的主要遗漏。当前 13 份结果的相关哈希均匹配，版本清单也与当前文件匹配。

**解析诊断已传到指标。** 合并步骤汇总每句的 `parse_diagnostics`，评分步骤写入 `alignment.parse_*`。[合并代码](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/scripts/merge_sentence_predictions.py:76) 与 [评分代码](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/scripts/evaluate.py:78) 的路径一致；逐份比对现有结果未发现计数不符。此前举出的 curriculum dev 2406 样本现显示 `parse_dropped_args=1`、`parse_unknown_roles=1`。

仍有两处具体缺陷：

1. **完成状态可能误报。** [write_report.py](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/write_report.py:41) 仅凭复杂分数数量为 7 判断通过，并仅凭 `verification.log` 存在判断已验证；[状态写入](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/write_report.py:73) 还将 `baseline_test` 固定为 `complete`。缺失、过期或失败的验证记录，以及名称错误的七项分数，都可能产生错误的完成报告。
2. **逐文档诊断仍丢失原文。** merged 结果保存了 `sentence_outputs`，但 [逐文档指标记录](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/scripts/evaluate.py:90) 只读取不存在的顶层 `raw_output`；上述实际样本的 `.metrics.docs.jsonl` 因而仍为 `raw_output: null`。

本次没有编辑文件、使用 GPU 或依据 test 选择 checkpoint。尝试运行独立验证脚本时，当前只读环境因没有可用临时目录而在导入 `torch` 阶段停止；结论依据代码、现有验证记录及独立的只读哈希和计数检查。