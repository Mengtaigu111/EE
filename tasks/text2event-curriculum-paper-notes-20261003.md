# Text2Event 课程学习：论文与官方代码核对

核对日期：2026-10-03。仅分析，未改训练实现或启动实验。

## 来源

- ACL 正式论文：[2021.acl-long.217 PDF](https://aclanthology.org/2021.acl-long.217.pdf)，重点 §3、§4.1、§4.4/Table 5。
- [官方仓库](https://github.com/luyaojie/Text2Event)，读取 main 对应 commit `dbe5a0ae933d639980c8e1f62c141511c12cefd6`。
- [课程学习训练脚本原文](https://raw.githubusercontent.com/luyaojie/Text2Event/dbe5a0ae933d639980c8e1f62c141511c12cefd6/run_seq2seq_with_pretrain.bash)。
- [普通训练脚本原文](https://raw.githubusercontent.com/luyaojie/Text2Event/dbe5a0ae933d639980c8e1f62c141511c12cefd6/run_seq2seq.bash)。
- [树/平铺子结构转换原文](https://raw.githubusercontent.com/luyaojie/Text2Event/dbe5a0ae933d639980c8e1f62c141511c12cefd6/data_convert/format/text2tree.py)。
- [转换数据主程序原文](https://raw.githubusercontent.com/luyaojie/Text2Event/dbe5a0ae933d639980c8e1f62c141511c12cefd6/data_convert/convert_text_to_tree.py)。
- [约束解码原文](https://raw.githubusercontent.com/luyaojie/Text2Event/dbe5a0ae933d639980c8e1f62c141511c12cefd6/extraction/extract_constraint.py)。
- [Trainer 入口原文](https://raw.githubusercontent.com/luyaojie/Text2Event/dbe5a0ae933d639980c8e1f62c141511c12cefd6/run_seq2seq.py)。

## 已证实机制

论文 §3 的前阶段学习 label-span：事件类型与 trigger、角色与 argument；随后学习完整事件关系。没有先筛选 easy sentences。目的是降低事件嵌套结构与普通预训练输出之间的差异，并避免过度学习低语义信息的括号。

论文文字称每个抽出的子结构构成一个 sentence/substructure pair；**公开代码并非逐个子结构单独一条**。`Text2Tree.annotate_span` 将同一句全部 type-trigger 与 role-argument 平铺到同一个虚拟 ROOT，例如 `((Transport returned) (Artifact The man) (Arrest-Jail capture) ...)`；完整树则把角色挂回其所属事件。不能混淆论文文字与代码行为。

`convert_file_tuple` 同时输出 full 和 `_span` 两套数据，输入句子相同。默认与调用参数均 `ignore_nonevent=False`，无事件句也保留。前阶段并不排除复杂多事件句，而是简化这些句子的输出目标。因此全部复杂句的类型、trigger、role、argument 在第一阶段就获得监督，暂时不学它们的嵌套归属。

## 预算和阶段切换

论文 §4.1 明确子结构 5 epochs、完整树 30 epochs；公开 `run_seq2seq_with_pretrain.bash` defaults `pretrain=5`、`epoch=30`。普通 `run_seq2seq.bash` defaults `epoch=30`。这说明公开默认脚本的课程前阶段是额外训练，并未从后面的完整树训练扣掉预算；**Table 5 未明确报告等更新步数/等计算量控制，所以不能声称论文消融已排除预算收益**。

两阶段使用两个独立 Python 调用。第一阶段 `source_prefix='span: '`、`decoding_format=treespan`，按 `eval_trigger-F1` 选最好模型；结束后只复制 config、权重、tokenizer 文件到 `span_pretrained`。第二阶段 `model_name_or_path=span_pretrained`、新的 output_dir、prefix `event: `，按 `eval_role-F1` 选最好模型。未复制 optimizer/scheduler/RNG，也未以先前 checkpoint resume。Trainer 入口 `run_seq2seq.py` 行645直接令 `checkpoint=None`，行647调用 `trainer.train(resume_from_checkpoint=checkpoint)`；因此第二阶段重新建立优化器/调度器。课程脚本默认 `constant_with_warmup`、warmup_steps=0，不是我们贯穿 4010 步的 cosine。

这些是 T5/ACE/ERE 的设置，不能直接把 5+30 epochs 移植到 Qwen3-8B LoRA/SciEvents；结构、数据密度、参数更新方式不同。

## 消融：不是必然提升

Table 5 是 ACE05-EN **dev**、T5-base，数值不是 SciEvents test：

| Train比例 | TC，无CL | TC，有CL | TC差值 | AC，无CL | AC，有CL | AC差值 |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 17.9 | 24.6 | +6.7 | 3.7 | 8.6 | +4.9 |
| 5% | 52.1 | 52.8 | +0.7 | 30.9 | 33.6 | +2.7 |
| 25% | 65.0 | 65.5 | +0.5 | 44.7 | 44.0 | −0.7 |
| 100% | 69.6 | 71.4 | +1.8 | 52.6 | 53.3 | +0.7 |

作者支持课程有用，但表中 AC 在 25% 下下降，100% 下增益只有 0.7 个百分点；不应把“用了课程”当作必定更好。论文正文的概括平均提升与表中这四项简单平均不一致；比较应使用具体表项，不能转述为已自行复算验证的平均收益。

## 约束解码与可移植边界

§2.3 使用 trie/state 随输出位置限制类型、角色、文本 span、括号。官方代码类型位置查 `type_tree`、角色位置查 **全局** `role_tree`；span 延续限制为原文连续 token 序列；state 限制括号/结束。因此不能根据公开代码夸大为完整 type-specific role 约束：它没有按当前事件类型查 type-to-role 映射。

形式合法和引用原文不等于语义选择正确；即使当前输出全都可解析，也不意味着该解码机制没有作用。是否给我们的 Qwen 增益需要单独控制解码对照。

## 对我们实验的解释（推断，待验证）

我们 easy + ROOT → all train 是样本课程；Text2Event 是输出子结构 → 完整结构的任务课程。二者机制不同。当前前 814 更新排除了 B2 复杂样本，也没有把 B1 的整树目标变成 label-span 目标，因此并没有复现该论文的主要课程机制。

我们固定总 4010 更新，Stage 1 用掉 814，留给全量约 3.985 epochs；在 best 2406 时，全量阶段约 1.985 epochs。基线同总步数至 2406 约 3 full epochs。因而此时 B2 少约一轮监督，B1 累计多约一轮；这可以解释精确率下降的可能机制，但不是因果证明。不能单凭当前单 seed 结果断言 Stage 1 2 epochs 已过拟合。

合理后续对照应拆开：①保持当前方法但不压缩完整树学习预算，识别预算取舍；②从全部 train 的 label-span 简化目标学起，再完整树，识别子结构机制；③保持 total updates 对照，以区分额外计算。开发前应明确任务模式/目标格式和是否保留无事件比例；本次只做分析，不更改已完成实验。
