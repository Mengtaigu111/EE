# Seed52 独立 review

审阅人：/root/fast_independent_review。与原 type-cot-online-20261006 逐文件diff，并检查共享sampling/evaluation/checkpoint实现；未发现 P1/P2 阻塞。

## 实际差异和冻结边界

- 新outer manifest seed=52，原tree_reference seed=42仅用于锁定原始数据、prompt和参考规格。实际树cfg显式覆盖seed=52，独立verify取消seed与reference相等要求，改为核对outer/run/CoT/tree全部52。
- CoT与树 set_seed、TrainingArguments seed/data_seed、sampling_seed 均52；CoT随机首轮使用52，树首轮仍固定完整Sorted1顺序，第二轮及之后使用52+epoch。独立sampling审计同样显式使用52，正确处理12828的epoch尾12样本和partial epoch。
- smoke标准重载predictor与最终test predictor传入seed52。在线dev为原resident evaluator，greedy且不重新set_seed，保存/恢复进入时RNG；没有隐藏的seed42初始化或随机采样。
- 唯一其他行为差异：移除不参与训练控制/选模的独立CoT dev诊断。未引入fast microbatch、vendor或Liger；原native loss与安全state机制不变。
- 保留原用户定稿CoT prompt及gold、原AppendixNoCopy、完整训练数据、首轮排序/后续随机、microbatch1/accum16/bf16/SDPA/gradient checkpointing；CoT802/cosine warmup25，树3208/plateau warmup97。
- dense_trainer.py与原版本完全一致：13个完整dev点、original official-offset AC control、min_delta .001/bad3降率/bad6停止、不因降率重置计数、devbest选模及训练结束后的test保持。
- CoT原始base/new LoRA初始化、本次CoT final-adapter来源限制及训练入口权重exact检查、新树optimizer/scheduler/controller、安全checkpoint和source_snapshot/hash冻结保留。历史文件未改。

## 实际验证

亲自运行 qwen3meld-run Python 的新CPU suite：7 passed in 9.92s。包括seed52冻结配置拒绝seed42、source不含fast/vendor、完整/partial/尾组采样、拒绝seed42后续epoch，及真实tiny Qwen3 native Trainer 对CoT与树的seed52实际采样记录与loss_kwargs检查。

审阅未启动GPU。正式launch仍需root完成本次seed52两阶段GPU冒烟、标准adapter reload逐条一致及安全状态/反馈/评分核验。
