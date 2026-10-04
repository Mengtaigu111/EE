# 评测修复交接

2026-10-04 已完成CPU修复及重算，不重训、不修改原gold/官方实现/历史run。

- 新实例评分器保留同trigger/同文本重复实例，一对一匹配。覆盖TI/TC/AI/AC/EC_I/EC_C全P/R/F1和TP/FP/FN；这是独立诊断口径，官方数值仍保留。
- baseline、Easy2、Sorted1全部19组dev/test评估复算；dev最佳仍2406。Sorted1新offset：TI74.46、TC62.57、AI43.95、AC41.75、EC47.73%；新raw token：74.62、62.79、44.53、42.34、48.17%。基线新raw token：74.91、63.29、44.76、42.45、47.74%。
- 旧token AC确有漏分，旧TI也有重复命中导致虚高。不能将所有差异视为低估。固定已对齐预测，Sorted1同文异位额外token TI命中6个（3个跨句）。
- 旧文本表示丢失95个gold论元的offset顺序/重复项；另有40个gold位置未被旧候选算法枚举。新lossless_forest可选位置协议保序；新ambiguity_resolver枚举候选、显式暴露歧义。不能恢复旧输出未提供的位置，也未接入现有训练prompt。

[完整结果、验证与代码入口](../../tasks/stage2-event-tree/scorer-repair-20261004/report.md)。旧官方分数见01，04是修复前的只读核查。后续报告必须并列注明官方/新实例、offset/raw-token，不能混比。
