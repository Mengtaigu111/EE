# Seed52 完成后的独立复核

审阅人：/root/fast_independent_review。只读检查正式run、report、train_metrics、初始状态、handoff、反馈轨迹和seed42正式run，并实际重新计算hash与控制轨迹。没有启动GPU或新实验。

## 完成与冻结证据

正式run为 C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-52-20261007-000051。exit=0、run.status=completed；CoT802更新，树3208更新。2026-10-07北京时间00:01:01至15:40:59，共15.6659小时（约15小时40分钟）。移除CoT独立dev诊断后比seed42的17.1447小时少约1小时29分钟；没有证据表明训练更新本身提速。

实际逐项校验通过：100份源码文件hash及其source_snapshot、49份冻结输入hash、manifest hash；CoT/tree seed52、microbatch1/accum16。CoT初始化证据为fresh base/new LoRA、optimizer state空、global_step=0、scheduler_last_epoch=0；树handoff来自本run CoT final-adapter，504张量exact、optimizer空、新scheduler/plateau_scale=1和controller eval_count=0/global_step=0。无Liger或batch2路径。

## 反馈控制轨迹

独立手动按min_delta=.001、bad3降率、bad6停逐点复算，13个时间点和日志一致：1403达到有效改善后，1604/1804/2005连续3次无有效改善，2005降率1e-4→5e-5；2205为bad4；2406 dev AC40.5186386%，比有效锚点1403有足够改善，因此重置bad0并成为raw最佳。2606/2807/3007无改善，3007降率5e-5→2.5e-5；3208为bad4，budget_exhausted。不是早停，也没有跳过预算。

Test使用dev最佳checkpoint-2406，实际预测配置seed52/batch4/greedy319；adapter hash与该checkpoint及best-adapter一致。test_controls_training=false。

## 统一官方 offset F1 对比（百分比）

以下只比较各seed的dev-selected checkpoint，六项均来自 official_offset，未混入实例或raw-token评分。

| Run | dev-best step | dev AC | Test TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CoT seed42 | 2205 | 41.0751 | 75.5592 | 64.1541 | 44.7401 | 42.4606 | 47.9205 | 47.9205 |
| CoT seed52 | 2406 | 40.5186 | 75.0246 | 63.1060 | 43.6123 | 40.9853 | 47.0862 | 47.0862 |
| 52减42（百分点） | — | -0.5565 | -0.5346 | -1.0480 | -1.1278 | -1.4753 | -0.8343 | -0.8343 |
| 两seed平均 | — | 40.7969 | 75.2919 | 63.6301 | 44.1762 | 41.7229 | 47.5034 | 47.5034 |

## 可支持的结论和边界

seed52的dev及test六项均低于seed42，test AC相差1.4753个百分点。seed42上相对历史基线的AC提升没有在seed52上复现，当前结果不支持“稳定提升”的表述。两seed平均AC41.7229%是描述性汇总，不能据此声称显著优于或差于只有seed42的普通基线41.88%。

两seed也不能分离CoT自身因果效果：普通基线/旧online对照与本方案的验证频率、学习率决策或预算不同；仍需同seed、同密集dev/控制规则、同树阶段配置的不做CoT对照，额外CoT802更新预算必须并列披露。避免根据test另选checkpoint来替换这次devbest。

本独立复核未逐条重跑全部18项评分；root另执行完整verify(include_test=True)复算dev/test，完成声明应引用其实际结果。
