# 在线 dev 训练方案独立审查

**结论：Approved。** 未发现需要先修改方案才能实施的严重阻断。本结论仅审查方案与本地源码的兼容性，不表示新训练实现或 GPU 实验已通过验证。

审查范围：`06-online-dev-training-plan.md`、`07-online-dev-execution-prompt.md`，现有 `train_sorted_first.py`、`run_sorted_first.py`、`curriculum_trainer.py`、`sorted_first_sampler.py`，以及 qwen3meld-run 环境中 transformers 4.52.3 的 `trainer.py`。未修改旧训练源码、历史 run，未启动 GPU。

- 冻结要求完整：原 AppendixNoCopy、原数据、原始 Qwen3-8B 新 LoRA、第一轮固定 Sorted1 / 后三轮 seed+epoch 随机顺序均保留；新增控制与 3208 步预算明确。微小提升与平局后 step 的 raw best 独立于 0.001 anchor；bad3 降率不清零、bad6 停止的轨迹一致，首次 401 建 anchor 后持续无改善确实在 1604 降率、2807 停止。
- HF 钩子可行：本地 `Trainer._maybe_log_save_evaluate()`（3064 行）先评估、再 `_determine_best_metric()`、再保存；停止标志在保存后的训练循环检查（2638 行）。方案明确要求避免 HF 默认严格 greater 比较覆盖平局规则，能够通过新子类实现。每个 optimizer 更新的 scheduler 调用位于 2616 行；自定义非 ReduceLROnPlateau scheduler 能满足每步一次，反馈只改变平台 scale。
- 安全 checkpoint 可扩展：`CurriculumTrainer` 已使用 JSON+safetensors 保存 optimizer/scheduler/RNG，完整保存后封存 continuity hash。方案明确要求把 feedback 状态加入完整性校验，并恢复计数、scale、评估 step 与采样位置，因此没有“仅 adapter 续训”的设计缺口。
- 采样预算可行：本地 HF `set_initial_training_values()`（5323 行）按 ceil 计算每 epoch 更新数，12828/16 对应 802 次；3208 是四轮。现有 sampler 携带 epoch 到实际 `training_step` 记录，能复用顺序并以新 verifier 校验早停前缀。方案已明确不能复用旧完整五轮 validator。
- 完成门槛覆盖 CPU 控制流、旧预测评分复算、GPU 训练/生成/恢复/adapter 重载冒烟以及独立验收；test 在训练结束后使用 raw best，未参与控制。

实施时需遵守的既有接口约束（不构成方案阻断）：HF 构造器在 `eval_strategy='steps'` 时要求非空 `eval_dataset`（452 行），即使覆盖 evaluate 仍应传入 dev 数据；若实现续训，现有 `SortedFirstTrainer` 构造器的 observation 文件独占创建及从零计数需在新子类/新日志段中处理，不能直接对原观察文件再次实例化。方案的 CPU Trainer 测试与完整 state 恢复/采样前缀测试应实际覆盖这些接口。
