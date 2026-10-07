# Seed52正式启动核验

Run：/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-52-20261007-000051。Tmux：ee-cot1-seed52-20261007-000051。北京时间2026-10-07T00:01:01.960746+08:00启动，实际CoT训练已完成至少8更新/802，141条真实模型输入匹配seed52随机前缀；GPU训练进程存在。

CPU7tests及独立review通过；GPU最长32 CoT2更新→树4更新两轮/2dev通过；完整checkpoint/采样/feedback/18指标独立复算通过；标准adapter重载14条原始生成完全一致。正式源hash、初始化/optimizer/scheduler、12828全train无截断token审计及seed52实际顺序通过。

原native micro1/acc16/bf16/SDPA，没有采用Liger/batch2；原prompt/data/course/13完整dev/officialoffset AC反馈和testonlydevbest保持，唯一非seed差异是省略不控制训练的CoT dev诊断。预算CoT802+树最多3208，训练完成后自动复核全部dev并评估devbest test。预计跑满15–17小时，早停可能提前；新日志进度会更新估计。

当前只确认正式训练启动，训练/完整dev/test尚未完成。证据startup-verification.json、main-launch.json、review.md、cpu-tests.log、smoke-verification.json。
