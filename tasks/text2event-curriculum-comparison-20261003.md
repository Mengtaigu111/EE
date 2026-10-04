# Text2Event 与 AppendixNoCopy-Easy2 课程对比

本次仅分析，不改训练代码、不启动新实验。主要结论：当前方法是句子难度筛选课程，Text2Event是输出子结构课程；两者不能视为同一算法。Stage1两轮是否过多仍为假设，不能从单seed下降证明。

## 论文证据

来源：[ACL正式PDF](https://aclanthology.org/2021.acl-long.217.pdf)，§3、§4.1、§4.4/Table5；[作者仓库](https://github.com/luyaojie/Text2Event)。官方代码详细证据见同目录text2event-curriculum-paper-notes-20261003.md。

- §3先生成(type,trigger)和(role,argument)简单子结构，后生成完整事件树；目的是缓解自然语言生成到结构生成的差距，避免括号等非语义符号干扰。
- §4.1为5轮子结构、30轮完整结构。与我们的等4010步对照不同，论文没有把Table5课程收益证明为等优化步数的排序收益。不能按T5/ACE的epoch直接配置Qwen/SciEvents。
- Table5是T5-base、ACE05-EN dev；100%数据Arg-C从52.6到53.3；25%从44.7到44.0；5%30.9到33.6；1%3.7到8.6。全量收益小，部分比例退化；不是每种设置都提升。
- §2.3另有生成时约束解码；它与课程是两个变量，不应将其当作这次baseline与curriculum差距的原因。

## 本地冻结实现

文件：tasks/stage2-event-tree/curriculum/appendix-easy2-20261003/mean_curriculum_manifest.json；build_mean_curriculum_data.py；build_appendix_easy2_config.py；difficulty-analysis/analyze_mean_percentile_difficulty.py；analyze_train_schema_completeness.py。

- B1=5427，B2=5277，ROOT=2124。Stage1为B1+固定1077 ROOT=6504，两轮814updates，completion仍为完整事件树；Stage2原始全train12828，3196updates，总4010与baseline相同。
- 难度score=0.5*((1-1/N)+(1-mean(type-local role-completeness percentile)))；role completeness=观察到的合法role种类数/schema合法role总数。这是结构代理，没有验证为模型实测困难程度；合法role并非全部必填。高填充率可能意味着更多长论元，低填充率可能是合法简短事件。平均percentile也可能掩盖同句较难事件。
- baseline约5轮完整train，当前约4轮完整train加2轮easy；best2406时baseline各组约3轮，当前easy约4、B2约2。计数为近似平均暴露，不能保证每条样本恰好同次数。
- Cosine与optimizer/RNG在814完整延续，复杂组首次参与时LR约9.2368e-5，未重置；缺少初始814步复杂样本监督是事实，其效果归因尚待消融。
- ROOT比例Stage1 16.559%、full16.558%，无证据把结果下降归因于ROOT比例过高。

## 本地结果与限制

来源：runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/run.json；goal-23/main-verification.json；evaluation-v2-20261003/results/baseline/test-checkpoint-2406.metrics.json。两组同prompt、同总步数、同修复评估、seed42；dev AC均选2406。

| 指标 | baseline | curriculum |
|---|---:|---:|
| dev AC F1 |41.2637|40.5298|
| test AC F1 |41.8797|41.1416|
| test TC P |64.4146|61.6634|
| test TC R |62.7042|63.6029|
| test AC P |42.9702|41.5714|
| test AC R |40.8431|40.7207|

精确率下降明显于召回下降；但AC统计依赖正确trigger/type，不能单凭该表将差距定位为argument边界错误。现有数据转换、全状态恢复和指标复算验证均通过，不是已知流水线故障。单seed不能判断显著性，样本难度与目标难度不一致、复杂监督不足均为候选解释。

## 建议与验证

优先从全train（包括B2）构造type-trigger和role-argument的简单子结构监督，再训练完整树；不引入gold trigger作为线上已知输入，不删多实例，不改原始gold/prompt文件。官方代码flat子结构与论文per-substructure实例构造须分别说明；跨句/不连续argument保留本任务真实表示，不能套ACE span假设。

先按正确trigger/type匹配后的论元边界/角色误差、事件类型与复杂性分组确认损失位置，再定正式方案。若实验，预先固定dev AC选模，统一seed/prompt/评估，并明确等总步数对照与额外预训练预算对照；增加预算时需有等预算full-tree baseline。只调整Stage1轮数可以验证时长假设，不能当作复现Text2Event机制。不得据本报告直接宣称新设计一定优于baseline。

交叉核验补充：作者官方实现annotate_span保留句子、把全部type-trigger与role-argument子结构平铺到ROOT（含原无事件句），并非论文文字描述的每子结构一条。两阶段新Trainer只加载模型，run_seq2seq.py显式resume_from_checkpoint=None；默认constant_with_warmup。完整结构阶段按eval_role-F1选模；本项目按dev AC F1，目标大体对应但不能视为评分细节完全相同。具体commit/行号见paper notes。子代理已交叉核对本报告主要论文判断，本地指标与步数已程序化复核。
