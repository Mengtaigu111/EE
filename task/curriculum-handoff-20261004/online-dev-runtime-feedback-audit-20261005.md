# 在线 dev 当前运行：反馈、scheduler、checkpoint 与恢复审查

审查日期：2026-10-05。目标 run：`C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-20261005-072128`。只读现有源码、运行产物和本地 transformers 4.52.3；未加载 Qwen3-8B、未启动 GPU、未改变运行进程或源码。本报告为独立新增文件。规格基准：`task/curriculum-handoff-20261004/06-online-dev-training-plan.md`。

## 结论

**本次范围内，未发现已证明会破坏当前 fresh、连续训练的严重控制或保存问题。** 当前已完成的 401/802 两次反馈、对应完整 checkpoint、首轮课程消费和 scheduler 更新计数相互一致。不能据此宣称尚未结束的全部训练已验收。

恢复相关限制须单独看待：底层 Trainer 的 checkpoint 恢复经过 CPU 模型逐参数验证，但正式入口不提供恢复、独立 verifier 不拼接分段日志；这不影响当前从原模型开始的一次连续运行。断点后不能直接把底层恢复测试视为完整 runner 恢复支持。

## 规格与控制链核对

以下文件均在 `tasks/stage2-event-tree/curriculum/` 下；行号以审查时文件为准。

| 核对项 | 实现证据 | 审查结论 |
|---|---|---|
| 3208 optimizer updates、401 步评估/保存、97 步 warmup | `train_sorted_first_online_dev.py:62,95–106` | 与计划一致。当前 config 中 max_steps=3208、eval_steps=save_steps=401、warmup_steps=97、dev_count=1593。 |
| 独立 raw best 与 anchor；0.001、bad3 降率、bad6 停 | `dev_feedback.py:32–54` | 精确平分取更晚 step；微小进步更新 raw best 而不清零 bad；累计达到 anchor+0.001 才重置；bad3 降一次、不重置等待，bad6 stop 优先。 |
| scheduler 每次 optimizer update 一次 | `dev_feedback.py:73–106`；本地 HF `trainer.py:2606–2622` | 自定义类不是 ReduceLROnPlateau；HF 正常 optimizer 更新后仅一次 `.step()`，然后 global_step+1。反馈 reduce_lr 仅改 scale/LR，不推进 last_epoch。 |
| 每 epoch 最后不足 16 行正常 flush | 本地 HF `trainer.py:5323–5327` | HF4.52.3 使用 ceil，12828/16 每轮为 802 updates，四轮恰好 3208。没有 floor=801 导致 epoch/步数错位。 |
| evaluate →反馈→best→完整 checkpoint | `online_dev_trainer.py:59–109`；本地 HF `trainer.py:3093–3104,3201–3206` | on_evaluate 只调用一次，随后强制 should_save，并保留 stop；自定义 `_determine_best_metric` 防止 HF 严格大于规则覆盖平局；HF 保存时 best_global_step 来自控制器，因此指向正确 step。 |
| final best 与 last 区分 | `train_sorted_first_online_dev.py:137–145` | last 保存驻留终态；best 从控制器选定的完整 checkpoint 复制 adapter。不依赖 load_best_model_at_end，也不会将最后模型冒充最佳。 |
| 完整 safe 状态 | `curriculum_trainer.py:177–227`；`online_dev_trainer.py:101–110` | optimizer/scheduler/RNG 用 JSON+safetensors，Trainer 状态和 adapter 被 integrity manifest 封存，反馈状态追加并封存。 |

当前 run 的 `dev_feedback.py`、`online_dev_trainer.py`、`train_sorted_first_online_dev.py` 与其 source_snapshot 字节一致。prompt SHA256 为 `2bcc66773a71904973ae4f85b865e60eddfacf4230515618576d3217e6776b41`，resume_from_checkpoint=None。未发现此控制链引入新的 prompt/课程修改；prompt、评估生成和评分本身应结合其他独立审查确认。

## 当前真实运行证据

对 401、802 两个 checkpoint 的 continuity.json 全部条目独立重新计算 SHA256，均匹配，包括 feedback_state.json。

| step | official offset AC_f1 | anchor | bad_count | best_step | scheduler last_epoch | plateau_scale / LR |
|---|---:|---:|---:|---:|---:|---|
| 401 | 0.12488497436571577 | 同分数 | 0 | 401 | 401 | 1.0 / 1e-4 |
| 802 | 0.3218475073313783 | 同分数 | 0 | 802 | 802 | 1.0 / 1e-4 |

反馈日志用 verifier 的独立规则重放，两次均一致。checkpoint-401 的 observed_microbatches=6416，checkpoint-802=12828，分别对应半轮和完整首轮。读取采样日志时的快照共有15215行，尾行 optimizer_step_before=951；完整首轮恰好12828行，其 index 顺序与 0…12827 完全相等。这是读取时的活动快照，不能当作任务终止步数。

截至该快照尚未触发 plateau；主 run 的实际 bad3/bad6 路径仍需未来产物验证，现有分支证据来自下面的真实 CPU Trainer 测试。

## 仅涉及恢复或诊断的限制

1. **正式 CLI/runner 没有恢复入口与分段验收。** `train_sorted_first_online_dev.py:26–31,63,117–121` 只接受全新 output 并调用 `train()`；`online_dev_config.py:89–92` 拒绝 initial manifest 中非空 resume。底层 Trainer `online_dev_trainer.py:112–120` 可恢复控制器和 microbatch 计数，但 feedback/observed 文件独占创建（`:39`、`sorted_first_sampler.py:82`），需要显式新段路径。`verify_sorted_first_online_dev.py:138,159` 只读单 feedback.jsonl/observed-order.jsonl；`replay_feedback:23` 要求日志从第一个 cadence 开始，`audit_sampling_prefix:94` 要求 microbatch 从0开始。CPU 测试的新 output 恢复段因此不能直接通过完整 verifier。对当前 fresh run 无影响；若中断后要继续，需先设计恢复入口与分段拼接审计，禁止直接重启原 runner。

2. **恢复后的 HF 聚合 train_loss 不等价于完整训练平均 loss。** CPU 恢复测试中，连续6步 train_loss=3.1591537396113076，而从 step4 恢复只跑5/6两步后为1.0527296861012776，尽管 adapter 参数、反馈、scheduler 完全一致。原因是本地 HF `trainer.py:2688–2691` 将本次进程累计 loss 除以含恢复前步数的 global_step。当前 fresh run 无此偏差；未来恢复报告应依拼接的逐步 loss 重算，不把 resumed train_loss 与完整 run 直接比较。HF train_runtime/throughput 同样不应代替累计真实耗时。

3. **完整性 hash 不等于全面语义校验。** `validate_online_checkpoint`（`online_dev_trainer.py:16–30`）检查 hash、格式及 feedback.last_step 不超过 global_step，但不在该函数内核对 feedback.best_step/score 与 Trainer 状态、scheduler LR/scale 的全部关系；`FeedbackController.load_state_dict`（`dev_feedback.py:67–70`）直接恢复字段。当前各 checkpoint 的实际语义对照一致；独立训练 verifier（`:165–182`）另补 scheduler/feedback 对照。这属于底层恢复 validator 的边界，不是当前 checkpoint 已损坏的证据。

以上三项不足以成为停止或修改当前训练的依据。尤其第1项是初版明确选择 fresh-only 的范围限制，不应误报为当前实验控制失败。

## CPU 验证

在 qwen3meld-run 环境使用 `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，执行：

```bash
cd /home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m unittest \
  test_dev_feedback test_online_dev_trainer test_sorted_first_online_dev -q
```

结果：**20 tests，5.073s，OK，退出码0。** 仅使用 tiny GPT2 LoRA 和临时 CPU checkpoint，未加载8B。覆盖 warmup97、Accelerate wrapper、持久 plateau scale、微小改善/平局/anchor、bad3降率与bad6早停、一次 on_evaluate、early-stop checkpoint、非有限/缺失指标拒绝、checkpoint hash、防重步，以及恢复后控制器/scheduler/模型逐参数与连续训练一致。

验证仍有边界：既有 CPU 恢复测试位于同一 epoch 内，未单独证明断点跨 epoch 的正式分段 runner 恢复，也未验证 GPU/bf16恢复；本审查未重复运行全量 GPU实验或推理。当前训练最终验收仍应等待 terminal state、全部反馈 checkpoint、独立全量评分重算和 best/last export hash 校验。
