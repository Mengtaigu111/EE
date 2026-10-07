# Seed52 指定checkpoint test独立review

审阅人：/root/fast_independent_review。只读检查新run_evaluation.py/verify_artifacts.py/manifest.json/plan.md，与昨日seed42补测脚本逐行diff；未发现P1/P2阻塞。

- 新runner串行评估用户指定2205→1804→3208，checkpoint均来自正式seed52 run的tree/checkpoints，并校验安全checkpoint步骤与3208 horizon。未启动训练、未写原run或best-adapter。
- 生成明确seed52、batch4、greedy、max_new_tokens319；保持原Qwen3/bf16/AppendixNoCopy数据/官方修复评分协议。新验证器diff仅seed42→52。
- 原devbest2406只从已保存原run读取并加入报告，最终表1804/2205/2406/3208统一official-offset六项，不混入其他评分口径，不按test重新选模型。其他四模式与全部18 P/R/F1独立存储。
- 实际重新计算manifest全部215文件hash通过；所有4个比较checkpoint目录文件均已冻结锁定，包括adapter与完整安全state。原run/source/data锁及每轮verify_locks防止评估中静默变化。
- 原artifact验证逻辑保留：1627句/250文档、顺序/metadata、completion hash、实际adapter路径/hash、生成配置、raw重新解析、merge重建、18官方指标与SciEvents/per-doc、7complex指标、四口径独立复算。
- recipe中的旧8点checkpoint_steps属于继承的来源规格，本runner不使用它进行训练或控制；补测不会改变正式run13点dev反馈事实。

根代理将实际执行已有2406全部artifact CPU复算和错误batch负例；该两项结果及GPU启动由根代理确认。本review没有GPU操作，不声明新3个checkpoint test已完成。
