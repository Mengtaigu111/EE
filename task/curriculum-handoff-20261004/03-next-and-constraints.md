# 下一步与约定

## 当前状态与结论

Sorted1完整训练、dev/test、offset/raw-token和独立verifier均完成，main.exit=0。当前无需启动或恢复这组实验。结果基本追平基线，不能宣称课程稳定有效或过拟合原因已确定。

Text2Event是“type-trigger / role-argument子结构→完整树”；我们是句子排序/筛选课程。作者代码对子结构平铺、复杂句也参与；默认额外5轮子结构+30轮完整树，不能直接套到Qwen。全量AC收益仅0.7pp，25%数据反而下降。[对比记录](/home/myluo/EE/EXCEEDS/tasks/text2event-curriculum-comparison-20261003.md)。

## 当前最值得验证的问题（尚未执行）

用户怀疑表示/后处理/评分低估实际正确预测。该怀疑合理，但未证明“真实结果一定更好”。建议：

1. 原gold直接评gold，检查评分器自身重复实例限制。
2. gold→文本树→解析/回填→评分，量化模型之外的表示与对齐损失。
3. 真实预测按模型错误、位置误对齐、官方配对错误分类计数。不能把gold往返损失直接加到模型F1。

若继续改课程/模型，先用dev选择方案，test只做确定方案的最终评估；新实验需明确选模口径和预算，单seed结论保持谨慎。

## 用户约定

- 保持句子级任务、完整摘要上下文；不要误改成一次生成整摘要。
- 原prompt/数据/gold和旧代码、旧run保留；新代码、新prompt、新结果用独立路径。
- prompt不要Region或Semantic Decision节点；未授权不额外增加Extraction Guidelines。
- 用conda环境 `/home/myluo/miniconda3/envs/qwen3meld-run/bin/python`，GPU实验优先tmux，先测试/审查/冒烟再长跑。
- 项目规则/待办在 [tasks/lessons.md](/home/myluo/EE/EXCEEDS/tasks/lessons.md)、[tasks/todo.md](/home/myluo/EE/EXCEEDS/tasks/todo.md)。

先前Stage3审查问题尚未在本课程任务中修复；未进入当前训练流程，不要误记为已解决。

2026-10-04 更新：评分实例覆盖修复、全指标复算、gold往返与TI位置分解已经完成，见[05评测修复交接](05-evaluation-repair.md)。旧文本的多位置歧义仍无法唯一恢复；新位置协议仅为已验证的可选新实现，尚未重训。
