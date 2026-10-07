# 当前 OnlineDev run：在线评估与数据隔离只读审查

审查日期：2026-10-05。对象：`tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128`，对照 `06-online-dev-training-plan.md`。范围为评估、配置冻结、runner/verifier、当前已完成 dev 产物与采样证据。只读 CPU 审查；没有改源码、干预训练、加载 GPU 8B、运行要求训练结束的 `verify()`/final verifier。

## 结论

在此范围内**未发现应阻止当前训练的严重实际问题**。401、802 两次已完成验证均为完整 1,593 句 dev，原始预测重新解析、合并、官方 offset 评分后的全部 18 项与保存 metrics 及 feedback 一致。当前 run 仍为 `running/train_online_dev`；未完成后续验证、终态验收和 test 属于正常运行阶段，不能作为缺陷。

## 审查计划及执行结果

- [x] 对照 06 规格读取配置、评估、runner、verifier 和实际 run。
- [x] 核验冻结 prompt/数据/旧来源、新源码及快照 hash。
- [x] CPU 从已完成 401/802 原预测重算全部官方指标及反馈轨迹。
- [x] 核验实际首轮完整消费和第二轮随机消费前缀。
- [x] 核查 train/dev/test 隔离、生成参数、上下文恢复及 test 闸门。
- [x] 记录当前阶段的证据与验证限制，不冒充最终验收。

## 规格与源码证据

所有行号指当前工作区源码，源码 hash 已与 run 快照核对一致。

| 要求 | 实际实现和证据 |
|---|---|
| 原 Sorted1 prompt、数据、排序 | `online_dev_config.py:9–12,66–84,94–120` 绑定历史 manifest/run；逐字节对比排序 train 与 source train 每行，检查 12,828 唯一 IDs 及 ordered IDs，不重新写样本。prompt SHA256 为 `2bcc66773a71904973ae4f85b865e60eddfacf4230515618576d3217e6776b41`。|
| 原 Qwen、新 LoRA | `online_dev_config.py:15,19,22,92` 固定原模型且 adapter/resume 为 None；实际 `train_sorted_first_online_dev.py:87–94` 从 base 加载并新建 LoRA，未加载旧 adapter。|
| 首轮排序、随后随机 | `sorted_first_sampler.py:18–36,85–98` 首轮顺序，随后独立 generator 的 `randperm(seed+epoch)`；观察日志由实际 `training_step` 写入，排除仅 collate/prefetch 的伪消费。|
| 完整 dev、greedy、batch4/max319、thinking关闭 | `train_sorted_first_online_dev.py:48–56,65–80,110–120` 使用原 dev 与独立 eval tokenizer；`online_dev_evaluation.py:125–145` 固定批量和 greedy 生成；`scripts/model_utils.py:68–71` 关闭 thinking 且不截断。|
| 修复版解析与官方 offset 全18项 | `online_dev_evaluation.py:53–73` 原预测重解析后合并评分；`verify_sorted_first_run.py:16–27` 显式加载固定文件并核对模块来源；`evaluation-v2-20261003/scripts/evaluate.py:22–34` 调用 `utils.calculate_F1(..., 'offset')`。|
| 仅 AC 控制 | `online_dev_trainer.py:63–77` 检查全部18项有限0–1，然后仅将 `official['AC_f1']` 传给控制器、更新 best。独立 `verify_sorted_first_online_dev.py:17–50` 直接重放阈值/等待/降率/best规则。|
| dev 不进训练 loss | `train_sorted_first_online_dev.py:48–51,93–94,117–120` train IndexedDataset 单独来自冻结 train；dev 仅注册 eval dataset 与闭包，override evaluate 经驻留模型生成。`scripts/model_utils.py:74–87` assistant-only loss。|
| test 不调参 | `run_sorted_first_online_dev.py:69–95,168–176` 子进程训练正常结束、独立训练/dev verifier 成功后才运行最佳 checkpoint 的 test；test 四种口径仅诊断，`16–41`。冻结 manifest 会读取 test 文件计算 hash/数量，这属于完整性核验，并非读取 test 成绩来控制训练。|
| mode/cache/RNG/padding 恢复 | `online_dev_evaluation.py:112–124,149–158` 保存 Python/NumPy/CPU/CUDA RNG、顶层 training 状态、cache 和 tokenizer padding/truncation side，在 finally 恢复；训练 tokenizer 独立且保持 right padding。|

## 新鲜 CPU 验证证据

执行解释器为 `/home/myluo/miniconda3/envs/qwen3meld-run/bin/python`，环境设 `CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`。调用 `validate_manifest()`、来源/快照 hash 检查、`verify_dev_scores()`、`replay_feedback()`、已完成前缀 `audit_sampling_prefix()`，全部退出码 0。没有调用 `verify_training()` 或 `verify()`；没有生成新评估产物。

- 39 项 frozen input/source locks 校验通过；run 所记 8 项新源码及 source_snapshot hash 校验通过；manifest SHA256 与 run 所记一致。
- dev 句子文件 SHA256：`2acc33d6ef60abcbc821818d03b89cbca1cb8a76dc7042f733feced70c0aab30`。
- dev 文档 gold SHA256：`d8e0dd35980600bcba5d3f5b5130c1ee95a008bbc9c607b9f0b0cb1ddfd369c8`。
- 重算检查覆盖 complete/count/hash、原 ID 顺序、merged 内容、全部18项、full scope 与 SciEvents 文档产物，实际对应 `verify_sorted_first_online_dev.py:191–218`。

| step | 原预测句数 | 重算官方 AC_f1 | 18项与 saved/feedback 一致 | invalid | hit319 | LR前/后 | best/anchor | bad |
|---:|---:|---:|---|---:|---:|---|---|---:|
|401|1593|0.12488497436571577|是|0|0|0.0001 / 0.0001|401 / 同步改善|0|
|802|1593|0.3218475073313783|是|0|11|0.0001 / 0.0001|802 / 同步改善|0|

对应反馈证据：当前 run 的 `training/feedback.jsonl:1–2`。`hit319=11` 已保存，未丢弃这些句子或缩小 dev。

首轮实际 12,828 行完整消费通过 `audit_sampling_prefix(..., global_step=802)`，IDs/index/microbatch/optimizer 边界与冻结排序一致。第二次快照当前共 15,285 个实际 training_step 行，epoch1 前 2,457 行逐项符合 `torch.randperm(12828, generator=manual_seed(43))`；最后一行 `optimizer_step_before=955`。这是动态运行快照，不代表 step955 已完成。checkpoint401/802 的 `trainer_state.json` 与 `feedback_state.json` 中 global/best/eval count 均与对应反馈一致。

当前 run 无 `checkpoint-eval/test*` 产物；runner 状态仍为训练中。这与 test 在独立验收后才进入的设计一致。

## 验证限制与后续正常验收

本审查用原预测验证评分及当前来源一致性，没有重跑 GPU 生成，也无法仅从保存的文本证明每次真实生成的精度、RNG恢复过程；这些由源码及先前 smoke/测试证据支持，不应将 CPU 复算称为 GPU 重新复现。读取 checkpoint JSON 不等同于完整 checkpoint integrity/optimizer/RNG 验收。

尚未到达平台降率/早停分支；已观察两次均为有效改善，不能从当前轨迹宣称降率/早停在本次实际训练中已经发生。训练结束后按原 runner 继续独立校验所有dev/checkpoint/采样完整前缀和 best/last 导出，再对 dev-best 进行一次 test；本报告不代替这些终态门槛。

## 复盘

当前审查未形成需要修改源码或重启训练的发现。当前 AC401→AC802 的提升只是 run 中间结果，不能据此宣称相较 baseline/旧 Sorted1 的最终收益或将变化单独归因于早停。
