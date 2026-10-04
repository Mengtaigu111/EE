# 评测与文本回填修复计划

需求：解决offset/token差异中已证实的评分覆盖、实例复用与表示/对齐问题；不为提高分数改gold，不覆盖官方代码、现有模型、历史run。用户已授权诊断、修复、复算，无须重训。

根因已复现：原gold自评offset AC98.635648/token AC96.694070而非100。utils.calculate_F1按trigger/argument表达字典覆盖不同实例，重复预测可被多个gold复用。现有04-score-consistency-audit.md已记录，原实现和旧分数需保留。

假设与验证：
1. 用保留实例的一对一匹配，可使gold自评各指标100%，重复预测不产生额外TP，事件/论元顺序不影响结果。
2. gold句子completion→parse→merge→offset过程中仍可能失分；不连续/逆序/重复文本等分类对照原gold确证。
3. 真实模型token TI较offset高可能包含跨位置同文/重复复用假命中，不代表所有差值为低估。固定预测集合与匹配器，分离模式差异、过滤/去重、排列影响。

规格：新instance_scorer.py，offset=原offset序列，token=精确tokens。TI按trigger multiset1:1；TC按(type,trigger)1:1；同TC组事件用最大权配对（先最大化AC、再AI、再EC、稳定tie）；论元与role multiset1:1。EC沿用表达属于同文档trigger集合的关系定义，明确token同文异位仍无法恢复真实关系。完整输出6F1+P/R+TP/FP/FN。官方评分独立原样对照，不称新分数为官方。

任务：
- [x] TDD实例评分器：gold自评、同trigger异type/同type异args、同span异role、重复预测、顺序不变、唯一键与官方一致、reverse offsets保序。
- [x] 可复现gold自评/文本树往返全量诊断，分类实际gold损失及完整例子，不使用gold给真实预测选择位置。
- [x] 新离线runner重算baseline/Easy2/Sorted1已有dev/test raw/merged，完整覆盖/hash/官方一致闸门、存实例修复分数与误差例子。
- [x] 实际TI token/offset差异逐doc计数说明；dev新规则选模与旧规则分开，不能用test选模型。
- [x] 独立CPU tests/review/结果复算，原源码hash不变，写简短报告并更新交接索引。

验收：所有gold self指标100；重复预测降低precision；无冲突fixture旧新一致；全量真实输入可重复、baseline旧分数精确重现；同时报告旧官方与新实例，分数可能升也可能降。不可声称恢复模型未输出的位置信息。执行仅CPU，优先tmux离线长诊断。

结果复盘：41项CPU测试通过；19组原sentence预测独立复算、6 test artifacts、73历史/运行hash通过；dev/test原gold全部18项两模式100%。40处旧候选遗漏全部覆盖但均存在歧义；新位置协议全gold无损，未改旧prompt或重训。三组最佳仍2406。偏差同时有漏分与虚高，见report.md及verification.json。
