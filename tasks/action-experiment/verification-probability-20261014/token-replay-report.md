# 原异常 token 重放诊断

2026-10-14，通过 tmux `ee-prob-token-replay-20261014` 执行，退出码 0，耗时 96.30 秒。完整来源/SFT handoff 校验通过，冻结 reference 的 SHA256 保持不变。

## 原记录

停止发生在 group 14、sample 3、optimizer step 3。响应共 57 个有效 token，其中一个越出优化 clip 区间，占 1.754%。异常为 **从 0 开始的位置 29，token ID 12526，文本 `(Object`，不是 EOS**。

原 cached log probability 为 -3.2889080048，原 full teacher 为 -3.0494232178，ratio 为 1.2705943584。

## 重建限制

使用原 group 0–11 的 48 条完整 rollout、原 tokens/mask/old/reference/advantage，重放三次原 AdamW 更新，没有重新生成或计算奖励。

第一个 optimizer update 前，16 条 sample 的 teacher、四组平均 loss/KL 与原记录完全一致。然而，第一次 optimizer update 后 teacher 出现偏差，全部 replay 的最大偏差为 0.1327228546。目标响应 teacher 的最大偏差为 0.2394313812，超过 1e-5。

因此 **exact_reconstruction=false**。这份重建 policy/optimizer **不能作为原 run 的精确恢复 checkpoint**；现有证据也不能将原异常准确归因于某一个内核。原 run 没有保存第 3 步 adapter/optimizer 的完整状态。

## 在重建参数下保持参数不变的控制实验

下列值均为同一个原 token 的 log probability：

| 路径 | log probability |
|---|---:|
| 原记录 cached generation | -3.2889080048 |
| 原记录 full teacher | -3.0494232178 |
| 重建 SDPA full | -3.2888545990 |
| 重建 SDPA forced cached | -3.1688671112 |
| 重建 SDPA 每步完整 prefix | -3.1688804626 |
| 重建 MATH attention full | -3.1689586639 |
| 重建 MATH attention forced cached | -3.1688709259 |
| 重建 FP32 LM head full | -3.2014236450 |
| 重建 FP32 LM head forced cached | -3.1643905640 |

SDPA prefix 与 cached 在这个 token 上相差 0.00001335；MATH full 与 cached 相差 0.00008774；只将输出层改成 FP32 后，full 与 cached 的差缩小至 0.03703308。Transformer 本体仍为 BF16，这不是全模型 FP32 实验。

这些控制结果表明重建模型存在对计算路径/输出精度的数值敏感性。但因原模型参数未精确重建，不能把这种敏感性直接当作原异常的确定性归因，更不能宣称已证明所有 GRPO 路径都一致。

全部 token/path 数值及分布见 `runs/probability-token-replay-20261014/tokens.json` 与 `report.json`；原 tokens、probability 文件和诊断脚本 SHA256 均纳入该 run 的 `provenance.json`。
