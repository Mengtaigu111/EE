# B1490 stable-v6 单阶段部署草案

本目录目前仅准备 controller、监控脚本和 CPU 测试；未冻结来源锁，未启动任何模型、调度器或投递消息。

`run_controller.py` 只启动一个 B。初次执行要求新 run 不存在，从旧 B1490 的显式完整恢复点分叉；
没有 A 阶段，没有自动恢复或重试。训练失败会保留失败状态，重新调用默认命令会拒绝。
后续恢复必须显式传入 `--resume-checkpoint`，且 checkpoint 必须位于新 run 自己的 recovery-checkpoints 中。
controller 对完整 checkpoint 的 manifest、文件清单和哈希进行验证，trainer 再验证配置/数值协议和状态边界。
完成状态同时要求最终 metrics、完整 complete checkpoint 和最佳 adapter 哈希有效。

初次命令为 conda 环境 Python 执行本目录 `run_controller.py`，须位于 tmux；标准输出/错误可追加到
本目录 pipeline.log，结构化运行状态也会写入该日志。阶段输出单独追加到 B.log。

部署由主代理依次完成：

1. 完成 GPU 验收并确定 v6 的最终评估 batch 配置；再次运行 CPU 回归。
2. 冻结 `source-sha256.json`，至少包含 controller、recovery_v6 全部 Python 文件，以及验收目录的
   result.json、gpu.exit、runtime-source-sha256.json，建议同时覆盖本目录 monitoring 脚本。
   本草案不生成该锁，controller 遇到缺锁、缺必要文件锁或哈希变化会拒绝启动。
3. 确认没有旧 A/B 或验收模型残留，确认新 output 不存在，在 tmux 启动 controller。
4. 只对已确认的旧 scheduler 单 PID 发终止信号，并确认该 PID 退出；不终止训练 tmux 或模型。
5. 读取旧 monitoring/state.json 最终状态，调用新 scheduler 的 `migrate_deadline(old_state,new_state)` 移植 next_check_unix。
   该函数只写新监控状态，不停/启进程、不投递消息、不修改旧状态。
6. 确认旧 scheduler 已退出后，在单独 tmux 启动新 monitoring/scheduler.py。新目录文件锁防止重复新实例。
   迁移后的 next_check_unix 如已到期，则立即投递一次；否则等待原截止时间，每次成功投递后再间隔两小时。

新监控消息核查新 B、旧 A1775/B1490 完整恢复点与旧 test800/test1200。生产文件只读，只保存本次巡检摘要。
无新增异常、阶段/队列完成或需用户操作时保持安静。正常训练进度变化不发送状态。
队列 completed 后，调度器在成功投递最终完整性检查后退出；投递失败按原规则每 60 秒重试。

旧 A/B 的 failure.json 与 checkpoint 指针保持原样，不能把历史故障当新异常。新 run 显式恢复时也保留其历史 failure.json；
巡检应结合 pipeline-state 的 explicit_resume、started_unix、last-resume.json 和文件时间判断新旧故障。

controller 通过 `verify_acceptance()` 核验 GPU 验收退出码0、result completed、最长 dev 完整768 token
生成通过且同 batch 的 benchmark 为 passed，并核对数值运行源码 SHA。只有该函数选出的 batch（1/2/4）
会传给 trainer 的 `--evaluation-batch-size`；不能从另一个未验收 batch 覆盖。该 batch 写入训练配置和精度转换记录。

CPU 验证（不会启动模型）：

```
CUDA_VISIBLE_DEVICES='' /home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m pytest -q tasks/action-experiment/launch-stable-B1490-v6-20261021
```

## 授权后台等待与串行接力

用户明确要求tmux等待资源后再跑。当前验收模型只在CPU准备，沿用无foreign计算PID且free>=21000MiB立即上卡，无连续空闲窗口、无预留。

`after_acceptance.py` 绑定既有验证器PID及进程启动ticks，只CPU等待，不启动第二个验收模型。验收正常完成、所有显式stage及runtime SHA通过、CPU测试source收据一致后，才生成source-sha256.json并exec单阶段B controller。失败停止并记录pipeline-state，不自动重启；尚未完成验收不能声称修复完成。最终batch取实际通过完整768token显存测试的批量，CLI不能覆盖为未验收值。

运行中的验证器日志：../verification-stable-v6-20261021/gpu.log，resource-state.json表示waiting/loading/ready。接力日志handoff.log；正式B启动后B.log。新run r0-v2-B-stable-v6-from1490-42-20261021；旧A1775/B1490始终只读，初始fork继承优化器/RNG/已提交日志及历史best1200，新数值协议与旧dev指标明确区分。

两小时monitor迁移保留原deadline，stage=verification时新B目录不存在属正常；stage=B才检查新训练。它不自动恢复任何故障，不重复通知旧A/B已知异常。pipeline completed时投递最终完整性检查后退出。

## 后续启动策略（2026-10-22用户更新）
当前B继续运行，不中断。之后所有GPU启动/恢复都必须连续60秒无其他计算进程且空闲显存≥21000MiB；任何占用/不足/查询失败都重置计时，等待不预留。next-start-policy.json为最新要求。冻结v6的gate目前仍为即时加载，后续启动前必须接入并验证60秒gate；本轮不修改正在运行的冻结代码，也没有安排自动重启。
