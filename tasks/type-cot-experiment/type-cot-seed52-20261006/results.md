# Seed52 CoT→树最终结果

正式实验2026-10-07 15:40:59北京时间完成exit0，总15小时39分57秒；CoT802+树3208=4010更新。停止原因为预算用尽，未提前早停。重新执行verify --include-test通过：原始源码/采样/13dev全部18官方指标/反馈与全checkpoint状态/最终test四口径独立复算。

统一官方offset F1百分比，test均使用各自dev AC选择的checkpoint。

| seed | best cp | dev AC | test TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2205 | 41.0751 | 75.5592 | 64.1541 | 44.7401 | 42.4606 | 47.9205 | 47.9205 |
| 52 | 2406 | 40.5186 | 75.0246 | 63.1060 | 43.6123 | 40.9853 | 47.0862 | 47.0862 |

Test AC seed52相对42下降1.4753个百分点。相对普通seed42基线41.88%，seed52为40.9853%，约低0.89个百分点；不同seed不构成配对无CoT对照。

AC precision 42.7886%→39.3311%，recall42.1375%→42.7847%；体现预测论元分类的precision下降，不能仅据此指定类型/边界等错误根因。

反馈：1403 devAC40.3339为anchor，1604/1804/2005连续三次无有效改善，2005 LR1e-4→5e-5；2406 devAC40.5186%有效改善并选为best、等待清零；3007第三次无改善LR→2.5e-5；3208bad4，达到最大预算而结束。

seed42 testAC42.46%的提升未在seed52复现；两seed不足以证明稳定提升或CoT独立因果。两个seed各自dev选择CP，不按额外test挑best。

耗时比seed42少约1小时29分，主要省去不控制训练的CoT dev诊断；树非dev与13次dev耗时基本相同，没有采用batch2/Liger，没有实现更快训练内核。

全部13checkpoint的六项dev F1与最终test四口径见 [C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-52-20261007-000051/report.md](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-52-20261007-000051/report.md)；全部P/R/F1见run.json及tree/online-dev逐CPmetrics。

新完整验收：completed-run-verification.json；独立复核另见review-final.md。
