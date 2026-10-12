# Action v2 正式训练（2026-10-12）

会话：`ee-action-v2-20261012`。串行SFT→完整GRPO预检→GRPO，报错或用户中断立即停止，不执行后续阶段。脚本：[run.sh](run.sh)。

输入为`../data/action-v2-goldorder-20261012/`；模板[兼容短v2](../prompts/action-v2-checked.txt)，只修改prompt，原Gold/Action/简化Tree/Sorted1顺序逐条不变。所有生成上限768、上下文4096；已验证完整最长Gold671。SFT从已保存原CoT LoRA启动fresh optimizer，不续训此前中断run。

SFT输出`../runs/action-sft-v2-42-20261012/`，RL输出`../runs/action-grpo-v2-42-20261012/`。`sft.log`、`grpo.log`、`status.log`与`exit.code`是运行中生成的本地产物，未提交模型权重和大数据。

```bash
tmux attach -t ee-action-v2-20261012
tail -f /home/myluo/EE/EXCEEDS/tasks/action-experiment/launch-v2-20261012/sft.log
cat /home/myluo/EE/EXCEEDS/tasks/action-experiment/launch-v2-20261012/status.log
```

正式训练过程中不要修改provenance锁定的源码、数据、基座或初始化adapter。SFT首轮Sorted1，后续seed+epoch打乱，最多3208步；401步起测完整dev，TC有效改善阈值0.001，连续3次未改善学习率减半、6次早停。RL使用同一prompt与数据，以SFT dev-best LoRA为policy与冻结reference初始化，保留None，G4，全train一次。test不控制训练。

启动前证据：[新数据审计](../audit-v2-20261012/README.md)。Git只存代码/prompt/小型来源与验证证据；数据生成与运行产物由hash锁定，文件存在不代表正式训练完成。
