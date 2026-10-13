# R0 RL-only交付与验证

旧RL已停止；Action SFT、v2冻结数据prompt、原action_evaluation.py、score_tree_presence.py、action_language.py、grpo_core.py、action_common.py均未修改。原SFT provenance与checkpoint保持原样。

## 实现

`action_rewards.py`：严格Action解析，事件实例按type+有序trigger token配对，同key最大配对数优先、矩形最大权指派最大化父事件内严格Role/Argument多重集交集。未匹配实例与Link保留分母。`R0=.5 FE+.5 FL`；合法STOP无EOS照常评分、非法Action零分；不加H/R1/offset/软匹配。

`train_grpo.py`：只接线奖励与必要统计/概率检查，原串行G4、sample-std优势、clip.2、beta.01、LR1e-5、温度1/top_p1/k0、768上限、None、单遍seed42、dev间隔按用户追加要求设为400 optimizer steps，TC→TI选模保持原实现。初始SFT参与选模，同分保留早checkpoint。原loss没有修改。

## 验证

- 主线102项CPU测试通过（`python -m pytest tasks/action-experiment/tests -q`），包含27项R0边界/重复/最优配对测试、8项概率诊断测试、固定raw原eval回归与旧算法tests。加上独立旧batch实验目录的25项回归，本地全工作区127项通过；该独立实验不属于本轮R0提交。
- 16048条Gold completion全部严格合法且R0=1，80个随机矩形配对与独立穷举一致。
- 固定1627条SFT test raw prediction，原presence全部metric/count与改动前完全一致；受保护文件逐字节hash一致。
- 完整SFT→RL handoff与模型/数据来源预检通过，新reward/diagnostics默认进入source_files及快照。

## 当前真实Qwen GPU smoke

4组G4，16次真实采样，758个有效completion token，真实R0累积一次更新。上限768但无强制最短长度；这是短程一致性验证，不是所有长响应的数值证明。

|路径|差异均值|P95|P99|最大值|ratio越界比例|
|---|---:|---:|---:|---:|---:|
|cached sampling / no-cache eval|0.003440|0.016081|0.098537|0.121199|0.000000|
|eval / train checkpointing|0.000000|0.000000|0.000000|0.000000|0.000000|
|实际backward current / 采样old|0.003440|0.016081|0.098537|0.121199|0.000000|
|初始policy / reference|0.000000|0.000000|0.000000|0.000000|0.000000|

梯度范数0.290468；policy参数数值digest变化，reference数值digest不变，所有锁验证通过，smoke exit0。

门槛记录在run配置/源码中：P99≤.15、最大差≤.5、actual ratio超[.8,1.2]比例≤1%。正式每条样本也检查并在失败前保留probability-checks.jsonl；不因loss有限自动放行，不静默提高阈值。

## 正式运行

tmux：`ee-action-r0-dev400-20261013`；输出：`runs/action-grpo-r0-dev400-42-20261013`；日志：`launch-r0-20261013/rl-dev400.log`。

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -u /home/myluo/EE/EXCEEDS/tasks/action-experiment/train_grpo.py \
  --init-adapter /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-v2-42-20261012/best-adapter \
  --dev-steps 400 \
  --output /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-r0-dev400-42-20261013
```

正式先step0完整dev，完成后开始更新。rollouts/groups/probability-checks记录原token、真实行为概率、R0及Event/Link分项/方差、有效组比例、保留重复实例的结构差异。dev-history保留原presence指标。正式训练与最终test尚未完成，最终test只在dev选出最终checkpoint后运行；best_step0需如实报告未改善。

代码差异：`code.diff`；源码hash：`source-hashes.json`；不可变文件hash：`immutable-before.json`；主线CPU：`core-cpu-tests.txt`；全工作区CPU：`cpu-tests.txt`；全量Gold：`full-gold-r0.json`；来源：`preflight.json`；真实GPU结果：`gpu-smoke-result.json`，原始短程采样/组记录：`gpu-smoke-rollouts.jsonl` / `gpu-smoke-groups.jsonl`；正式配置/来源：`run-config.json` / `run-provenance.json`。
