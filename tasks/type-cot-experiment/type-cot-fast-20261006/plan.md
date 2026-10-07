# CoT1→树同seed训练提速实施计划

目标：先真实GPU比较训练microbatch1/2，采用通过显存、loss、吞吐验收的配置，在新tmux以seed42重跑；不覆盖历史代码、数据、prompt或run。

冻结规格：原始Qwen3-8B/newLoRA，CoT一轮802步（cosine/warmup25），树最多3208步（warmup97/plateau）；相同prompt、gold、Sorted1首轮顺序/后三轮随机；有效batch16。树dev完整1593句，原13个检查点及official-offset AC反馈规则保持，生成batch4/greedy319，test不控制。移除独立CoT dev诊断。phase-specific训练microbatch只能1或2，累积16/microbatch；数值/dropout轨迹变化是新实验，非逐位复现。

架构：新config/run/verify沿用原版冻结锁及验证流程；训练类仅扩展实际多样本microbatch日志/审计。benchmark每种phase/batch独立fresh模型进程，2warmup+8measured更新，固定每16样本组，最长32压力与随机代表样本；选择batch2需至少5%时间收益、有限loss/gradient和显存余量。未通过保留batch1，不安装或升级依赖。

文件：config.py配置/manifest；sampling.py实际采样审计；train_cot.py/train_tree.py/dense_trainer.py训练；run.py串行pipeline；verify.py独立采样/feedback/评分复算；benchmark.py短测及decision；tests/CPU回归；launch*.sh仅tmux入口。

- [x] CPU测试先失败后通过：phasebatch参数、实际完整/截断epoch采样、尾组、不同答案长度下tokenweighted梯度。
- [x] 独立review冻结规格及采样/loss边界。
- [ ] 新tmux GPU四组benchmark，记录吞吐/显存/finite/完整组。
- [ ] 选择后manifest冻结，CPU校验与新tmux两阶段冒烟，标准adapter重载逐条比较生成。
- [ ] 正式新run/tmux启动，核验真实训练进度、采样/配置/显存，报告按实测估计的耗时。
- [ ] 归档启动证据、tests/review/benchmark/smoke，正式训练结束由独立verifier复算所有dev及选中test。

## 追加方案：融合loss短测

原生benchmark：CoT batch2 OOM；树batch2平均每更新慢11.62%，因此不采用。按原已授权③追加Liger0.6.2融合loss，独立HERE/vendor无deps安装，原conda不改。仅patch fused_linear_cross_entropy，RMSNorm/RoPE/SwiGLU不改。先真实GPU tiny模型对比loss/梯度与生成logits，再同16samplegroups完整压力benchmark；native与liger方案有明确source locks及kernelmetadata，新smoke标准adapterreload比较；无>=5%真实时间收益则native回退，不能强行宣称提速。

首次tinykernel进程因wheel METADATA Name=liger_kernel与包装器精确匹配liger-kernel不一致，在kernel计算前拒绝；已通过包名下划线/连字符标准归一化修正并真实vendor导入测试。旧exit1/log保留，新v2入口重新全部验证，未推进正式训练。

最终状态：用户改为seed52复现实验，原生配置；本加速研究仅归档，不启动同seed正式run。原生benchmark完整完成/回退1+1；Liger tiny GPU loss/梯度/推理通过，Liger完整benchmark中途按用户新方向停止，结果不能称完整加速验收。
