# 当前训练实现与修复真实性审查（2026-10-22）

审查范围：冻结recovery_v5对照当前recovery_v6、GRPO/reward/mask、正式控制器、60秒资源包装器、未部署的adaptive dev候选和当前只读OOM诊断。Git HEAD f720b99935a21681c4dff1bcf58c6149bdce18bb；这些实验文件包含未跟踪快照，使用已冻结文件内容为基准，不将全工作区无关变更归入本次审查。按实现标准及用户规格两轴分别审查。未修改生产源码/阈值，未重启模型，既有诊断继续后台等GPU。

## 实现/标准轴

### [P1] dev显存运行可靠性尚未修复
位置：recovery_v6/stable_arithmetic.py:57-62，action_evaluation.py:125。
完整head权重151936×4096每次通过weight.float生成2.318359GiB临时FP32张量。token_chunk16只切completion位置，dev走generate，未使用chunked scoring路径。B1600真实OOM堆栈准确落在此cast；因此现有代码不具备已经验证过的dev稳定性。是否由inactive split碎片主导仍需当前GPU受控重放，不能把分配失败现象冒充已证实的allocator根因。此项优先完成原权重/原顺序/原batch4重放的分配快照及单变量对照，再据结果修复，不预先认定降batch或empty_cache就是根治。

### [P2] 显存验收未覆盖真实连续评估历史
位置：recovery_v6/verify_gpu.py:163、172、195-205。
验收先清缓存，重复同一个最长prompt构造4条，再强制768；真实训练在不同长度训练序列之后，连续评估不同padded长度。其通过证明该孤立负载可运行，不能证明完整训练→dev切换可运行。真实dev前20条成功、第6批输入最大1340反而短于验收1406，却OOM，已经说明该覆盖缺口有实际后果。需要真实混合长度连续dev，以及训练后切评估的完整场景，而非单独更长输入。

未发现可复现的奖励、mask、GRPO损失或reference冻结错误；该结论限本次范围，非形式化正确性证明。

## 需求/恢复规格轴

### [P2] 控制器中断可留下训练子进程
位置：launch-stable-B1490-v6-20261021/run_controller.py:257-264；diagnosis runner的wait/except也有同类生命周期风险。
process.wait被KeyboardInterrupt打断时只写failed并退出，不转发可保存的退出信号/等待子进程，也没有模型自己持有的全局运行锁。用户之后显式resume可能再启动一个加载器。CPU用真实sleep子进程代替模型已复现：queue_status=failed，而child_still_alive=true；测试finally已回收该sleep，不留下进程。证据controller-interrupt-repro.json。建议控制器退出先通知子进程保存并等待退出，再释放运行锁；另以模型持有的项目锁约束多个入口。此问题不等于此前两次数值故障原因。

### [P2，未部署候选] 防重复检测遗漏诊断模型
位置：restart-B1600-safe-dev-20261022/resume.py:95-99。
assert_no_models只识别train_grpo_recoverable/verify_gpu/--trainer，漏掉目前实际存在的诊断probe.py。候选若此时部署会重复创建本项目CPU模型加载器，并可能产生准入竞争。此候选尚未配置request/source锁、尚未运行，不是当前生产故障。应使用统一项目模型锁/注册，而不是脚本名枚举。

完整状态恢复的主要规格成立：1600恢复点step1600/cursor6400/pending0、无梯度，optimizer/RNG齐全，11个文件哈希通过；best1200及早停历史继承。resume先截断未提交日志、隔离残缺dev，再做待完成dev1600。60秒门禁CPU测试覆盖foreign/显存不足/查询异常/加载争抢重置，不预留GPU。

## 原概率错误的验证结论

- 旧A/B原故障log-prob偏差约1.215576/1.113964。真实旧权重6条响应重放修复后最大0.000104904；不是仅toy测试。
- 真实16条样本累积、AdamW1490→1491，梯度有限、policy改变/reference不变，完整恢复policy/RNG一致。
- 本轮从旧1490日志字节边界读取到1600提交边界，排除继承BF16日志：共1760条响应，cached vs eval和current vs old最大0.000104904，eval vs train最大0；guard配置与旧B相同。证明新算术已在110次实际optimizer更新中显著控制原差异；不证明未来任意输入无限稳定，不回溯修正旧BF16轨迹。
- 显存OOM是另一问题，尚未完成GPU根因验证/修复。当前受控native探针仍waiting_gpu；正式B保持failed/1600。

## 方法实现与解释边界

- GRPO clip、KL估计、组内sample std+1e-4、4回答×4组累积除16符合当前设定；只policy LoRA训练，base/reference冻结。
- 四字段奖励与一对一实例/argument匹配实现吻合既定F1定义；代码正确不等于该奖励一定优于旧R0，需要真实同协议对照。
- B是field-masked GRPO变体：mask只筛选surrogate奖励驱动项，KL仍覆盖所有有效token；结构/EOS保留。不能把它描述成未选字段的预测完全不受更新影响。
- B1490后数值协议由BF16算术迁移为FP32主体；旧dev1200的TC0.634804是继承BF16记录。比较方法收益需要统一精度、数据、评估与checkpoint选择规则，并明确该迁移。

## 本轮验证

- v6+正式launch：130 passed、13 subtests；日志cpu-v6.log（2项弃用提醒）。
- pause/60秒包装器：28 passed、8 subtests；日志cpu-pause.log。
- 未部署safe-dev候选+入口：11 passed；日志cpu-safe-dev.log。
- 诊断内存摘要helper：3 passed；日志cpu-probe.log。
- 独立数值审查另跑稳定算术/reward/mask范围66 passed（与上方有重叠，不累加）。
- 首次将不同目录用importlib模式合跑出现bare import路径错误，已分别按各目录依赖路径重跑通过；原始collection日志保留cpu-extensions.log，不将其伪装成生产代码失败或已通过。
- 本轮未使用GPU跑新前向，不将CPU通过冒充OOM已修复；真实运行证据为existing verification与1600提交日志。本轮源文件23份SHA保持不变；证据verified-evidence.json。

两轴各2项发现：实现轴关键是dev运行可靠性/验收覆盖；规格轴是子进程生命周期及未部署候选的模型识别。
