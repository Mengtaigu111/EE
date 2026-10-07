# Sorted1 在线 dev：实现与启动

2026-10-05，用户授权在当前聊天实施06冻结规格。旧训练代码、prompt、gold、adapter与历史run均保留。

## 实现

新增 `curriculum/online_dev_config.py`、`dev_feedback.py`、`online_dev_evaluation.py`、`online_dev_trainer.py`、`train_sorted_first_online_dev.py`、`run_sorted_first_online_dev.py`、`verify_sorted_first_online_dev.py` 及4个测试文件。相同原始Qwen3新LoRA、AppendixNoCopy和Sorted1冻结顺序，后3轮随机；3208步上限，每401步驻留模型完整dev；97步真实warmup后平台LR，bad3减半、bad6早停，官方offset AC控制，raw-best含精确平局后step。test仅在训练结束并验收后运行。

每次验证保留全部18项P/R/F1、原始预测、解析/对齐诊断、完整安全checkpoint和反馈状态。best/last独立；正式runner会在完成后对dev-best进行一次test推理，保存官方offset/raw-token及新实例offset/raw-token四种口径，独立复算并写run内report.md。

## 验证

- 全部curriculum CPU测试60项通过；修复评估/新实例评分回归27项通过。
- 独立规格审查与代码质量审查通过。曾发现并补齐完整dev绑定、官方控制score一致性、必需hash清单、checkpoint真实LR、best/last验收闸门；错误案例先红后绿。
- 历史Sorted1完整dev18项官方指标及对齐diagnostics精确重算；历史test四种评分与既有修复结果完全一致（仅验证评分代码，未参与新训练控制）。
- 真实GPU冒烟退出0：4更新，step2/4两次真实dev，前2篇完整摘要14句，训练继续、checkpoint独立重载原始文本逐字一致；峰值allocated18.7701GiB，训练+dev81.15秒。最长32个真实训练样本保持原课程相对顺序。

配置/测试日志/审查入口：[新配置目录](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum/sorted-first-online-dev-20261005)。
冒烟run：[run.json](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-071815-smoke/run.json)。

## 正式启动

新tmux：`ee-sorted1-online-dev-20261005-072128`。
正式run：[run.json](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128/run.json)。
训练日志：[training.log](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128/training.log)。
脚本与退出码入口：[main-launch.json](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum/sorted-first-online-dev-20261005/main-launch.json)。

初始预计11–13小时，参照近期Sorted1每更新8.81秒，最多8次全dev（历史一次约1422秒），以及一次test（约1455秒）；训练减少20%不代表总耗时减少20%。实际吞吐/第一轮完整dev完成后再修正；触发早停会缩短。当前正式实验运行中，最终分数与完成验收尚待运行结束。

启动实证（2026-10-05T07:23:58.332222+08:00）：tmux存在，runner状态train_online_dev，≥10次真实参数更新、166个已消费microbatch；首轮顺序及全部锁hash通过。首次全dev将在401更新，test尚未运行。启动证据：[main-startup-verification.json](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum/sorted-first-online-dev-20261005/main-startup-verification.json)。
