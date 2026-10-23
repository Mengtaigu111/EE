# B1600 dev OOM：根因调查进行中

用户要求先找原因再修复。本轮未部署adaptive batch候选，未恢复正式B。

## 已证实事实

- 完整checkpoint1600：next_group6400、pending0、optimizer/RNG齐全，11文件manifest SHA核验通过。旧v6数值源码保持冻结。
- OOM在dev第6批（行20:24）的Qwen3 lm_head `_linear_forward`执行weight.float()时。151936×4096×4=2489319424 bytes=2.318359375GiB，精确对应报错2.32GiB，和batch大小无直接比例关系。
- 出错时allocated17.52GiB、reserved未分配3.99GiB、device free1.53GiB。allocated加本次申请约19.84GiB，低于独立验收成功peak20.6375GiB。多块未使用空间不能简单相加当连续可用块；这些账目支持碎片假说，不能单凭它们证明。
- 原失败批prompt长度1340/1214/1222/1223；之前各batch padded长度1297/1299/1289/1165/1209；验收是同一最长prompt1406重复组成batch且每组测试前empty_cache。失败输入更短，验收缺少混合长度连续分配历史场景。
- 旧训练外层保留少量sample/current/loss等引用，但多数逐token张量很小，backward释放saved tensors；尚无数GiB graph leak证据。
- 不支持其他大型GPU任务导致原故障：原错误的容量账目只剩约0.16GiB未解释，而当前其他用户GPU任务只是阻塞新诊断准入。

## 候选原因及可证伪验证

1. native allocator跨不同长度训练/评估分配形成inactive split块。原样重放抓pre-head、OOM现场及异常栈退出后的snapshot；检查inactive_split和最大inactive块是否不足2.318GiB，保持batch4和768预算。
2. 训练进入评估的存活引用/allocator历史与新进程不同。若新进程128条都成功，不声称复现，应继续保存末尾token的无optimizer更新预热对照。
3. stream延迟释放/workspace/环境差异。记录native backend、allocator配置及全部分配统计。

PyTorch官方文档说明inactive split与动态大小分配可能产生无法直接回收的碎片；empty_cache不能释放活跃张量，不能据此保证修复：
https://docs.pytorch.org/docs/main/notes/cuda.html#memory-management
本机Torch2.4行为以本地源码与真实受控重放为准，最新官方文档仅机制背景。

## 已启动的受控验证

tmux ee-B1600-dev-memory-diagnosis-20261022；runner.py串行执行native和expandable两个独立进程。精确1600权重/optimizer/RNG，不optimizer.step，不写生产run。原顺序dev首24条后继续至128，原20条预测逐token核对。每次模型加载须GPU无foreign且free>=21000MiB连续60秒，不预留。

CPU helper3tests通过；入口/脚本语法已验，完整checkpoint与原部署来源核验通过。GPU实际重放尚待资源，不能声称诊断已完成或修复已验证。矩阵结束后会向本chat已有授权队列投递继续分析请求，不启动额外训练。

## 首轮矩阵实测结果（2026-10-22）
Native与expandable新进程均通过前128条原dev，以及清缓存逐批的前24条；前20条原预测逐token一致，策略/reference摘要前后一致，无optimizer更新。Native reserved约24.02GB、inactive split约187.98MB；expandable reserved约21.72GB。两组均未复现OOM，不能据此宣称碎片原因已证明或allocator切换修复已验证。下一步独立诊断重放末16条训练样本的分配/反向，比较保留/删除最后训练引用；仍不更新参数、不写生产run。详见comparison.json。
