# A/B 概率一致性故障：共同机制已定位

## 结论

A1775与B1490的共同故障机制是：共享的BF16模型前向对计算形状敏感，逐token cached生成和整段full评分在基座矩阵乘、attention等计算环节产生数值差异，经过Q/K归一化及后续多层传播被放大，最终导致同一token的两路log-prob相差超过1。已在原始权重上精确复现并用提高主体精度的反事实实验验证，不再只是“怀疑BF16”。

这是执行路径层面的根因定位，不等于证明某个CUDA kernel实现有缺陷。未发现token、位置、cache语义错位；现有数值证据无需假设cache损坏即可解释。FP32干预也改变缓存dtype，不能形式上排除所有未测试的BF16特定问题。

## 证据链

1. 原始重放：分别导入A1775/B1490真实完整异常快照中的policy权重，固定原prompt及completion token，不重建optimizer，不采样。A121/B127有效token的full评分与原teacher全部逐值一致，cached评分与原采样记录全部逐值一致，最大误差0。
2. 首层定位：B首层input norm的整个合法前缀完全一致，基座Q/K/V投影已出现差异；相同输入、仅改变full矩阵与单行计算形状，基座Q投影重现最大0.00048828125差异。因此存在cache存取之外的直接误差来源。K/V整个前缀的BF16基座投影最大差约0.00390625/0.001953125，Q/K归一化后偏差可放大；第一层k_norm权重绝对最大34。
3. 同Q/K/V attention控制：相同旋转后Q/K/V、相同数学可见集合，full causal目标行与单query/合法前缀SDPA结果仍有小差异，换MATH明显缩小。差异来自不止一个数值环节，不能全部归为最终输出head或某一个attention算子。
4. 后续层：失败预测位置在第一个decoder层输出已有小差异，后续层差异总体增大。hook不改变两路原始概率。该trace不是所有token逐算子的完整因果链。
5. 关键反事实：保持原权重值、同token，统一提高linear算术、norm/RoPE/残差/中间激活与对应缓存等主体精度为FP32，TF32关闭；分别测默认SDPA选择和固定MATH。固定MATH排除“仅换attention backend”的混杂，原大偏差下降三个数量级以上。

## 控制实验

| 条件（全部有效token最大绝对log-prob差） | A | B |
| --- | ---: | ---: |
| 原BF16默认SDPA |1.215576|1.113964|
| 仅FP32输出head，主体BF16 |1.113197|1.167236|
| BF16主体＋MATH attention |0.485207|0.242245|
| 仅首个decoder层FP32、出口回BF16 |0.985367|0.367432|
| 全主体FP32，默认SDPA选择 |0.00006485|0.00015640|
| 全主体FP32，固定MATH attention |0.00008011|0.00015450|

主体FP32控制没有把存储基座权重改成另一组数值；BF16存储权重在linear调用时临时转为FP32，不做更新。LoRA权重全部为FP32且有显式断言。与“整个模型持久以FP32存储”不同：此处只为反事实实验降低常驻显存。FP32_default可自动选择不同attention backend，因此核心精度比较以固定MATH为准。首层局部控制保留上游BF16 RoPE、出口回BF16，其不足不能否定第一层的贡献。

## 概率偏差如何组成

`full_logp − cached_logp = Δ目标logit − Δ词表logsumexp`。

- A：目标logit48.75 vs49.5，差−0.75；归一化项53.014164 vs52.548588，差+0.465576；合计−1.215576。
- B：目标logit31.375 vs31.75，差−0.375；归一化项40.014481 vs39.275517，差+0.738964；合计−1.113964。

因此不是简单“一个选中token的输出值少算了一点”：目标和其他词表logits的共同变化会通过softmax分母放大概率比。大差只在少数敏感token越过阈值，解释为何能训练很久才触发单token保护，而不是每一步都失败。

## 额外响应验证

选择与两次失败权重同一optimizer step的额外响应，确保无需重建旧权重。四个控制样本基线仍与原采样/teacher逐值一致；分别为较小差异、同step其余样本中的较大差异，不把它们称为整个训练史最高差样本。

| 样本 | 原最大差 | FP32 MATH最大差 |
| --- | ---: | ---: |
| A group7101 sample0 |0.002434|0|
| A group7100 sample0 |0.244904|0.00001907|
| B group5961 sample0 |0.003986|0.00000381|
| B group5960 sample0 |0.199482|0.00004578|

共6条独立响应、457个有效token，FP32 MATH的最大差0.000154495，未出现token超0.15/1或ratio超clip区间。CPU验证另将这6组已记录数值重复8次，只为覆盖ProbabilityGuard的3个完整窗口状态；这不是48条独立模型响应，也不构成新训练数据或长程稳定性证明。

## 已排查与修复边界

- reward、字段mask、backward或本次optimizer更新不是复现该差异所必需的操作。B字段mask在guard之后才应用；不能将新奖励或B mask直接归为本次故障原因。此前训练轨迹对数值敏感性的间接影响仍可能存在。
- 只改输出head、只改第一层、换MATH attention、禁用BF16 reduced precision reduction均不是已验证的通用修复。
- 原BF16采样概率不能直接与新的FP32 teacher混用；若未来切换训练计算路径，应从完整optimizer边界重新生成rollout，并记录计算配置变化。不能替换old概率或提高guard阈值来掩盖差异。
- 全主体高精度已作为诊断充分干预，但未进行训练backward/显存/吞吐验证。前向能在24GB GPU运行不代表完整FP32训练可直接运行，不能据此自动恢复B。
- 本任务为定位故障，未部署训练代码修改、未启动新训练，A1775/B1490恢复点和原数据/阈值保持不变。

## 复现与验证

工作记录：`/home/myluo/EE/EXCEEDS/goal-28/`。

- 首层：tmux运行 `bash tasks/action-experiment/diagnosis-AB-20261021/run-first-layer.sh`；证据first-layer-results.json、first-layer.log/exit。
- 主体精度：tmux运行 `bash tasks/action-experiment/diagnosis-AB-20261021/run-precision.sh`；证据precision-results.json、precision.log/exit、probe_precision-original-snapshot.py。
- 额外样本：tmux运行 `bash tasks/action-experiment/diagnosis-AB-20261021/run-validation.sh`；证据validation-fixtures.json、precision-validation-results.json、对应log/exit。
- CPU终验：`CUDA_VISIBLE_DEVICES='' /home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/action-experiment/diagnosis-AB-20261021/verify_root_evidence.py`；证据root-verification.json/log。

三轮GPU实验exit0。15个冻结生产文件来源hash仍一致，A/B恢复点各11文件hash仍一致。所有诊断模型均退出，纯CPU资源等待器亦无残留。独立审查认可系统级共同机制及上述归因边界。

一般浮点背景参考PyTorch官方说明：https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html 。它不替代本次原权重实验，也不作为某个具体kernel有bug的证据；安装版本2.4.1的接口和实际行为以本地源码与运行结果为准。
