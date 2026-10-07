# CoT→树训练提速审计

只读调查，独立子代理核对一致；没有启动新训练、安装依赖或修改历史代码/run。

| 部分 | 秒 | 小时 | 全流程占比 |
|---|---:|---:|---:|
| CoT训练 | 9045.22 | 2.5126 | 14.66% |
| 树非dev（含加载/保存等） | 28313.57 | 7.8649 | 45.87% |
| 树13次dev | 17610.89 | 4.8919 | 28.53% |
| CoT dev诊断 | 5258.23 | 1.4606 | 8.52% |
| 其余阶段 | 1492.93 | 0.4147 | 2.42% |

总计 61720.83 秒。树非dev 8.8259 秒/更新。

## 已核实配置

- RTX4090 24GiB，当前GPU空闲；microbatch1/accum16，bf16，SDPA，非重入gradient checkpointing。
- allocated峰值CoT20.23GiB、树18.77GiB。实际长度CoT mean1329/max2163、树mean1063/max1430；动态padding，microbatch1无批内padding浪费。
- 当前torch2.4.1/transformers4.52.3/peft0.19.1；无flash-attn/liger-kernel，不能直接套最新文档而忽略兼容性。
- Qwen3默认forward计算全序列lm_head logits；prompt标签-100只屏蔽loss，不省去prompt上下文计算，也不自动省去输出投影。未做profiler，不能认定其为最大瓶颈。

## 推荐顺序与验证规格（尚未实施）

1. 将不控制训练的CoT dev诊断移出主训练串行路径。若不再执行，可省5258秒；若仅后移，总GPU耗时不会减少。
2. 新隔离benchmark比较训练microbatch1×accum16与2×8：同一固定16样本更新分组/相同初始化与样本顺序、实际答案token权重、warmup后计时、最长样本显存压力、有限loss与梯度；CoT和树分别测，不依据推理batch4/8的速度推断训练速度。
3. 如显存或吞吐收益不够，隔离环境测试Qwen3支持的Liger融合损失/算子；先核验现有版本兼容性、ignore_index与shift/归一化、loss/gradient误差，再核对训练和生成模式切换以及adapter标准重载。不得原地升级历史conda环境。
4. 使用CPU验证、独立review和GPU冒烟通过的加速配置，新run从原始Qwen3-8B/newLoRA重跑seed42；原数据/prompt/课程/3208树步预算/13次完整dev/控制规则保持；推理仍batch4/greedy319。microbatch与融合算子改变数值/dropout轨迹，不能称逐位复现。只依据dev选模型，test不参与加速配置选择。

不建议为了提速削减dev或输出319预算，也不建议按长度改课程。没有实测新吞吐，不承诺加速百分比或结束时间。

参考：[HF gradient accumulation](https://github.com/huggingface/transformers/blob/main/docs/source/en/grad_accumulation.md)、[Liger官方Qwen3支持与融合损失](https://github.com/linkedin/Liger-Kernel)。

耗时来源：上述run的cot/train_metrics.json、tree/train_metrics.json、cot-dev/diagnostic.json、run.json。
