# OnlineDev运行中检查（2026-10-05）

当前run：C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128，tmux同名ee-sorted1-online-dev-20261005-072128。只读检查，不修改source、不干预GPU训练。

- 当前约950/3208更新，已进入第二轮。GPU利用率96%，约21.1GB显存；最近loss0.12–0.15、无非有限loss/OOM/报错。
- 两次完整dev：401步AC12.4885%，802步AC32.1848%，各1593句/251篇摘要。原始预测独立CPU复算全部18项，与feedback/metrics精确一致。
- 39个冻结来源hash与8个新源码/快照hash核对通过；原prompt hash、首轮12828样本完整顺序、后续seed43随机前缀一致。
- scheduler在checkpoint401/802的last_epoch分别401/802；warmup97已结束，LR1e-4、scale1。两次dev有有效改善，所以bad_count=0，不降率/早停是正确行为。最佳当前802。
- 模型驻留评估后恢复train/cache/padding/RNG，dev没有反传，test只在训练结束后最佳模型评估；未发现当前run的阻断问题。

最近完整dev（802步）F1：TI68.6957、TC51.0342、AI34.5833、AC32.1848、EC_I35.1830、EC_C35.0649%（精确原值在feedback/metrics）。反馈及真实tiny Trainer CPU共20项tests通过，checkpoint401/802全部continuity hash通过。

## 已看到的限制

802步11条dev输出达到319 token上限（0.69%）；部分出现重复事件及未闭合尾部，容错解析有dropped_args/events，invalid=0不代表完整树。原Sorted1同阶段也有10条hit，dev AC31.9986%；当前32.1848%处于接近水平。这个上限与旧实验一致，因此不是新增在线dev机制的配置偏差，也不能凭11条hit断定全是缺少长度。为保持实验一致，本次不临时改生成预算或重新评估控制历史。

在线降低学习率和早停尚未在正式run触发；当前只证明两次评估与训练恢复正常，不能提前证明完整训练结果或最终泛化收益。恢复接口限制另见独立反馈审查；不要在当前run改代码或直接运行要求训练完成的final verifier。

独立报告：[反馈审查](online-dev-runtime-feedback-audit-20261005.md)、[评估审查](online-dev-runtime-evaluation-audit-20261005.md)。
