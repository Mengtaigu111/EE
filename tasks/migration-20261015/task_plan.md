# 跨服务器迁移计划

用户最新规格：GitHub搬代码；新机已有Conda，按固定版本重建；不搬任何CoT/SFT/RL adapter/checkpoint，重新训练。搬数据、原始CoT生成结果、prompt、schema、课程顺序、配置和来源记录。新机已有Qwen8B，默认使用project-only清单、不搬基座，保留基座文件校验，不修改原评测器/Prompt/R0，不启动新训练。

- [x] 并行盘点递归数据/源码/来源锁及环境；剔除缓存。
- [x] 核对方案：GitHub源码 + rsync精确文件清单；不复制环境、不复制训练权重。
- [x] 生成无checkpoint传输清单与SHA256，写新机重建和验证说明。
- [x] CPU回归、清单完整性/环境验证、检查暂存内容。
- [x] 本地提交核心源码及迁移工具；push受网络超时阻塞，不声称远端更新。
- [x] 按新要求生成project-only ZIP，逐文件SHA及ZIP全量CRC检查通过。
- [ ] 收到目标SSH后执行传输及新机验证（目标未提供）。

## 限制

历史manifest有绝对路径，应保留真实目录或bind mount，软链不足。历史代码锁的两个版本冲突保留原记录，不能将当前代码冒充历史源码。旧RL没有optimizer完整快照，不能精确resume。无adapter迁移包只能重建训练，不能直接test现有CP或继续RL。
