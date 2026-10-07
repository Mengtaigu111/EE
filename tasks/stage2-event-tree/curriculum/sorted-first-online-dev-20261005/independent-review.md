# Sorted1 online-dev 独立实施审查

审查日期：2026-10-05。审查依据：`task/curriculum-handoff-20261004/06-online-dev-training-plan.md`。本文件审查源码与 CPU 证据，不代表 GPU 冒烟或正式实验已通过。

## 第一阶段：规格审查

**结论：通过，允许进入方案要求的 GPU 冒烟；正式启动仍须以成功冒烟为前置条件。**

已逐项核对新配置、训练入口、反馈控制器、驻留评估、Trainer、runner 与独立 verifier。原 AppendixNoCopy prompt、冻结句子样本、Sorted1 首轮顺序及后续 seed+epoch 随机课程保持一致；从原 Qwen3-8B 新建 LoRA，明确无旧 adapter 续训。正式预算3208 optimizer updates、验证间隔401、warmup97、平台降率与等待规则符合规格。dev 生成仅使用 prompt，同一驻留模型评估，不加载第二份8B；只用官方 offset AC 控制反馈，test 不参与选择。

前轮提出的验收缺口现已关闭：

- 正式 dev 验收绑定冻结原路径、hash、1593句和完整内容；smoke 独立重算原 dev 前两篇摘要的所有句子与文档 gold，避免相对子集的“full”被误当全量 dev。
- 每次 dev completion 的 status/count/原预测 hash 独立核验，原预测、merge 结果、全18项指标与 SciEvents 产物重算一致。
- 训练验收同时核对 best/last adapter 权重与配置对应的完整 checkpoint；HF best path 受独立反馈回放约束。
- 每个 checkpoint 的 scheduler update count、warmup、base LR、plateau scale、实际 scheduler LR 与 optimizer 各组 LR 对照反馈轨迹核验。
- 历史冻结 hash 清单缺项及新源码清单缺项均拒绝；control score 必须等于同条记录的官方 AC。

实际采样验收支持早停后的正确课程前缀，并检查 microbatch 连续性与 optimizer 边界。smoke 规格为最长32个真实训练样本、4次更新、2/4步评估、完整两篇 dev 摘要；训练进程退出后独立加载 checkpoint-4，以逐句原始输出精确比较驻留生成。

## 第二阶段：代码质量审查

**结论：通过；未发现需阻止 GPU 冒烟的实现问题。**

控制器、评估、训练连接与验收分工清楚，旧源码不修改。评估输出独占创建、异常状态恢复、固定模块显式加载、非有限指标拒绝、checkpoint反馈状态封存等实现与实验可靠性需求相符。新 verifier 对反馈规则独立重放，避免仅复用控制器自证。训练结束保留 last，另导出 dev-best；test 在训练和 dev 独立验收通过后才可访问。

非阻断建议：`test_online_dev_trainer.py` 的 `setUp()` 将 torch CPU 线程数设为1而未恢复。当前测试运行于独立进程，不影响随后单独启动的实验；未来若同进程合并更多测试，可通过 `addCleanup(torch.set_num_threads, original_threads)` 改善隔离。

## 本轮验证证据及范围

使用 `/home/myluo/miniconda3/envs/qwen3meld-run/bin/python` 运行：

- `unittest discover -s tasks/stage2-event-tree/curriculum -p 'test_*online_dev*.py'`：21项通过，实际日志 `/tmp/online-dev-spec-rereview-tests.log`。
- `unittest discover -s tasks/stage2-event-tree/curriculum -p test_dev_feedback.py`：9项通过，实际日志 `/tmp/online-dev-spec-rereview-feedback.log`。
- `git diff --check`：通过。

合计30项 CPU 测试。包括真实小型 CPU LoRA Trainer 的训练中验证/降率/早停/安全保存恢复，以及历史 Sorted1 checkpoint-2406 完整 dev 原预测重算18项与历史分数精确一致。fake generation 与注入分数仅验证控制流，不作为实验成绩。

本审查未运行 GPU、未验证 Qwen3 实际峰值显存，也未宣告正式训练完成。方案 Task5 的 GPU 驻留评估→恢复训练→保存→独立重载比较仍是启动正式任务的必要闸门。
