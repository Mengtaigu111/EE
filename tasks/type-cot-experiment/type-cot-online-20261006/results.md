# CoT1 → Sorted1 dense online dev：最终结果

北京时间2026-10-06 17:39完成，exit0，总耗时约17小时9分钟。
Run：tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-42-20261006-003100。

本次重新执行verify(run, include_test=True)通过，证据completed-run-verification.json。
全12828条CoT随机一轮/802更新、树51312条四轮/3208更新的真实采样，全部13次dev的18项官方P/R/F1、checkpoint安全状态及反馈规则、最终test完整性/原始到位置合并一致性/四口径评分独立复核通过。

dev选择树checkpoint-2205，AC41.0751%，约2.75轮树训练；其前已有一轮CoT。test只评这个dev-best checkpoint。

下表统一官方offset F1百分比；历史基线/Sorted1采用此前修复后公布的两位小数，不混用实例或raw token评分。

| 实验 | dev AC | test TI | TC | AI | AC | EC_I/EC_C |
|---|---:|---:|---:|---:|---:|---:|
| 普通基线 | 41.26 | 75.41 | 63.55 | 44.20 | 41.88 | 48.07 |
| Sorted1 | 40.55 | 75.19 | 63.10 | 44.16 | 41.98 | 48.19 |
| 上一组在线dev | 40.68 | 75.56 | 62.91 | 44.30 | 41.80 | 48.66 |
| 本次CoT1＋加密dev | 41.08 | 75.56 | 64.15 | 44.74 | 42.46 | 47.92 |

本次test AC较普通基线约+0.58个百分点、Sorted1约+0.48、上一在线dev约+0.66；TC/AI改善，EC退步。dev AC仍低于普通基线约0.18个百分点。仅single seed，CoT、验证频率及反馈时机同时变化，不证明CoT独立收益或稳定改善。

反馈轨迹：1203 devAC40.7538%；1804连续3次无有效改善，LR1e-4→5e-5；2005 devAC41.0190%达到有效改善、等待清零；2205 devAC41.0751%更新raw best，但增加0.0561个百分点不足0.10阈值，等待计数1；2606 LR→2.5e-5；3208等待6触发stop，同时已经达到最大预算。因此记录early_stop，但没有提前节省树阶段更新，总更新802+3208=4010。

CoT终点dev类型多重集F167.23%，只是诊断，不参与最终选模。其他test评分：官方raw-token AC42.0349%，实例offset AC42.6180%，实例raw-token AC43.1852%；不得与上表混用。全部18项test P/R/F1及四种口径在run.json/report.md保留。
