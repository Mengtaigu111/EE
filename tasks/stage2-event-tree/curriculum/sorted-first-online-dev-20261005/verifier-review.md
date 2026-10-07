# 在线 dev 独立验收复审

日期：2026-10-05（Asia/Shanghai）。审查范围：`dev_feedback.py`、`online_dev_config.py`、`verify_sorted_first_online_dev.py`；对照交接方案 `06-online-dev-training-plan.md` 第3/4/5节。审查者只读源码和运行CPU验证；本次仅新增本文档。

**结论：此前4项验收遗漏均已修复，当前审查范围内未发现阻止GPU冒烟的缺陷。此结论不代替GPU冒烟、实际run完整checkpoint验收或正式实验成绩。**

## 复审结果

1. 控制分数与官方评分已绑定：`replay_feedback`核对日志`score`与`official.AC_f1`；`verify_dev_scores`从同批原预测重新解析、合并、官方评分，并比较全部18项。再次注入`score=.405、official.AC_f1=.9`，现在正确报错退出。
2. 验证集已绑定冻结来源：`validate_dev_inputs`验证配置SHA、实际内容及顺序。正式实验要求原dev句子/文档路径和1593句；冒烟由原dev前两篇摘要重新推导全部句子和gold，不接受任意缩小子集。该验证在训练验收和dev重算入口均调用。
3. 必需来源锁已覆盖：manifest验证逐项要求历史来源/产物hash和额外dev/test来源hash；新源码清单必须与`source_files()`严格一致，并逐项核对本地源码和snapshot。再次将`frozen_hashes`清空，现在正确报错退出。
4. 完整checkpoint语义已补强：除完整状态hash、更新步数、warmup97/冒烟1外，独立核对scheduler平台缩放、base LR、各组最后LR和optimizer LR与反馈结果；控制器状态逐字段对照日志。best/last导出均逐文件比较对应checkpoint，Trainer最佳路径须与重放选模一致。

反馈控制器的raw best/后step平局、独立anchor、0.001有效改善、bad3单次降率、bad6停止优先、warmup与平台缩放保持，均符合方案。采样审计独立重建首轮顺序与后续seed+epoch排列，检查完整轮和末尾部分轮、microbatch/ID/optimizer边界，包含epoch最后不足累积数量的flush；未发现公式错误。

## CPU证据及限度

- `test_dev_feedback.py`：9项通过（首次审查已运行，日志`/tmp/review-dev-feedback.log`）。
- 修复后`test_sorted_first_online_dev.py`：8项通过，4.392秒；包括错误LR、score/official不一致、替换正式dev子集、错误/缺失冻结hash、部分epoch错序，以及验收失败不能运行test；日志`/tmp/review-online-verifier.log`。
- 原失败案例重复执行：错误控制分数、空冻结hash集合均拒绝。
- checkpoint新增LR字段和导出比对已逐行复审；本次没有启动GPU或伪造正式run，实际产物仍由主任务GPU冒烟及完整独立verifier证明。

## 审查源码SHA256

| 文件 | SHA256 |
|---|---|
| dev_feedback.py | `78be21ff2a380d5e321054284e390742bcd7f4037bf9586ceae04bf90bb539ea` |
| online_dev_config.py | `cb0e0ac7a844acc057f0d82805c8ba13cdb1ed363af2465bac291039d256f64d` |
| verify_sorted_first_online_dev.py | `32e5507ab52b4fccee949d0310efa80a64054cdd6f529a71ef6a87d05039aa55` |
