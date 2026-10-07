# 融合 loss 分支独立 review

审阅人：/root/fast_independent_review。范围：追加计划、kernel_support.py、check_kernel_equivalence.py、benchmark_liger.py、config/run/verify 与训练调用点，并逐行对照本机 Transformers 4.52 的 Qwen3 forward/ForCausalLMLoss 和 vendored Liger 0.6.2 实现。

静态检查未发现 P1/P2 阻塞；这不是 GPU 门槛通过声明。

## 已核验的语义边界

- 仅 fused_linear_cross_entropy=True；RoPE/RMSNorm/SwiGLU 和独立 cross_entropy 均 False，native 分支不导入 Liger。验证版本必须0.6.2、实际导入路径必须在独立 vendor 内。
- LigerForCausalLMLoss 与 native4.52 均在 labels 右侧 pad(ignore_index=-100)，取 labels[...,1:]，将最后位置忽略；有 num_items_in_batch 时 loss 为有效答案 token CE 总和除以整个梯度累积组 token 数。Trainer 已检查 num_items_in_batch 可进入模型。
- 冻结 LM head 时 vendor 算子不分配 grad_weight，但仍计算 grad_input = grad_logits @ weight，使 LoRA 梯度能够回传；head 不在 LoRA target 列表。
- vendor forward 在 train+labels 时省略完整 logits；labels=None 的 eval/generation 仍执行同样 lm_head(kept_hidden_states)，并传递相同 cache 等输出。Qwen3 state_dict/adapter 格式未改变。正式冒烟中的独立原生predictor重载会继续检查实际生成一致。
- 新 manifest 可选 training_kernel native/liger，run 与两阶段config记录 kernel；verify核对这三处与manifest相同。source_files新增全部 vendor/liger_kernel/**/*.py 及 dist-info/METADATA，run 会复制源码快照并冻结hash。prompt/data/gold/course/反馈/选模/test的边界沿用上一独立review。

## 必须由实际运行满足的剩余门槛

- GPU tiny BF16 检查必须真实通过：不同答案长度共享num_items分母，非零LoRA B保证A/B均有梯度，loss与逐参数梯度在明确容差内；不修改权重，labels=None logits 与 greedy generation 必须exact。
- 完整尺寸16样本有效组 benchmark 与原生同 groups/lengths/参数/环境；最长32压力、finite loss/grad、显存门槛与至少5%耗时收益。失败时保留native，不把微型模型检查当作8B性能证据。
- 最终配置两阶段GPU冒烟与标准adapter reload逐条检查通过后才能正式启动。融合算法有新的浮点计算路径，结果是可比较的新实验而非旧run逐位复现。

独立亲自执行 CPU suite：34 passed in 18.35s（含此前25项及新增kernel边界/配置测试）。CPU证明配置与损失归一化边界；尚不能证明实际Triton GPU正确性和8B速度。
