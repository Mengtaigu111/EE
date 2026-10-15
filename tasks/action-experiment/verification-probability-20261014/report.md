# 概率保护修复与验证（2026-10-14）

只修改诊断保护和训练接线；R0、v2 Prompt、原评测器、来源校验、GRPO core、G4串行、learning rate1e-5、beta.01、loss clip.2不变。完整dev仍每400 optimizer steps。

## 原失败与根因范围

原57-token响应仅token29（0-based）越出优化clip：ID12526，文本`(Object`，非EOS。cached/full logprob=-3.288908/-3.049423，ratio1.270594；1/57=1.754%触发旧逐回答1%硬门槛。60个回答3841tokens只有这个越界，不能据此声称GRPO严重错误。

旧run未保存step3完整policy/optimizer。用原48样本重建前三次更新，首次更新前teacher精确，更新后不精确；不得拿重建state冒充精确resume。详情见token-replay-report.md。

新live GPU短测在第3次更新后固定同一parameter digest重放原57tokens：前后digest相同，排除对照期间参数变化。此live状态仍不完全等于原停止状态，不能将其宣称为原异常精确重放。SDPA BF16 full/cache该token差约.11984；仅将LM输出层投影改FP32作控制，差约.00310。支持数值路径/输出量化敏感性，不足以断言唯一根因；MATH attention对照也保存。生产精度、attention和采样分布不改。

## 新保护

- 每token原IDs、mask、old/current/no-cache概率先保存，再检查，保留失败证据。
- 真实优化clip诊断仍[.8,1.2]；独立诊断保护range[.5,2]，不拿它替换loss clipping。
- 非有限/单token绝对logprob差>1立即停。
- 跨至少16回答且512有效tokens的非重叠窗口汇总；P99>.15或优化clip越界率>1%或保护range越界率>1%，连续3坏窗才停；正常窗reset。
- 短回答单个离群点报警，不直接停；真实loss和优势没有过滤/缩放，零奖励组仍正常backward。

## 验证

主线119项CPU passed；加上旧独立batch回归总144 passed。包含24项新guard测试、旧R0/eval回归和源文件hash。6个不可改文件hash仍一致。

64次真实回答、16prompt groups、4次AdamW（每4groups一次）短GPU测试passed/exit0。step0 policy/reference完全一致，reference结束hash不变，policy参数实际改变；eval vs train/checkpoint差始终为0。实际current-old汇总：

|均值|P95|P99|最大值|ratio范围|优化clip越界率|
|---:|---:|---:|---:|---|---:|
|0.003275|0.013535|0.092955|0.241560|0.785402–1.226100|0.05442%|

有效tokens=3675，窗口检查=4，最大连续坏窗=0；梯度范数最后更新=0.376035。逐组KL与原clip_fraction、真实R0、每次梯度norm保存在gpu-groups/gpu-updates/gpu-rollouts。仅短程证据，正式运行继续持续监测。

## 运行

已直接启动tmux `ee-action-r0-window-20261014`，无180秒空闲等待。旧step3无可精确续训的checkpoint，故新run从已完成SFT best-adapter初始化fresh optimizer；这是验证后重新启动，不是假称恢复原optimizer。

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u tasks/action-experiment/train_grpo.py \
  --init-adapter /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/best-adapter \
  --output /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014 \
  --dev-steps 400
```

日志：launch-probability-20261014/rl.log。先step0完整dev，然后训练；最终test未运行。正式运行仍TC→TI选模且允许SFT step0胜出；不声称训练已完成。

外部核对：[TRL说明优化clip统计与训练/生成数值差异](https://huggingface.co/docs/trl/grpo_trainer#dealing-with-the-training-inference-mismatch)。本文没有引入其vLLM修正算法。
