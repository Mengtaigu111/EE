# 新窗口执行说明

直接把下面这段发给新窗口：

> 请先阅读 `/home/myluo/EE/EXCEEDS/task/curriculum-handoff-20261004/06-online-dev-training-plan.md`，按方案新写代码、测试并实施。以最近的 Sorted1 为基础，使用完全相同的原 AppendixNoCopy prompt、数据、首轮从简单到难/后三轮随机的课程和原始 Qwen3-8B 新LoRA初始化；只接入训练中的 dev 验证、学习率调整和早停，最大3208步、每401步完整dev验证。不要覆盖旧代码、prompt、gold或历史run，不要拿旧adapter续训。先通过CPU测试、独立review与GPU冒烟，再开新tmux跑正式实验，报告真实启动状态和预计耗时。训练用修复后官方offset dev AC控制，全部其他指标并列记录，test不能参与控制。当前仅有方案，新在线dev代码尚未实现。

背景按需阅读：01实验区别、02数据/评分、05评分修复。方案第1节是冻结规格，第3节定义精确反馈规则，第5节是可核对执行清单。
