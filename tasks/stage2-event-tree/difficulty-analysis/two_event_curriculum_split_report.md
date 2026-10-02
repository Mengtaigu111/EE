# Two-event train sentence completeness split

仅分析 sentence-C train 的既有 gold completeness 表。事件 percentile 为 event type 内的 midrank percentile：小于当前 completeness 的事件数加同值事件数的一半，再除以该类型总事件数。同一类型、同一 completeness 的 event 得到相同 percentile。2-event sentence score 取两事件 percentile 的最小值。

## Q1. 2-event min-percentile distribution

| Statistic | Value |
|---|---:|
| Count | 3916 |
| Mean | 0.380150 |
| SD | 0.197342 |
| Min | 0.001506 |
| Max | 0.990309 |
| P10 | 0.144932 |
| P25 | 0.259790 |
| P50 / median | 0.368763 |
| P75 | 0.408584 |
| P90 | 0.690467 |

Ten-bin distribution:

| Bin | Count | Percent |
|---|---:|---:|
| [0.0, 0.1) | 315 | 8.04% |
| [0.1, 0.2) | 144 | 3.68% |
| [0.2, 0.3) | 1215 | 31.03% |
| [0.3, 0.4) | 386 | 9.86% |
| [0.4, 0.5) | 1084 | 27.68% |
| [0.5, 0.6) | 112 | 2.86% |
| [0.6, 0.7) | 329 | 8.40% |
| [0.7, 0.8) | 195 | 4.98% |
| [0.8, 0.9) | 82 | 2.09% |
| [0.9, 1.0] | 54 | 1.38% |

## Q2. Median threshold and ties

Median threshold = **0.368763**. High uses score ≥ threshold and Low uses score < threshold; median ties were kept together (no random split). Exactly equal to threshold: 278 (7.10%).

| Group | Count | Percent of 2-event |
|---|---:|---:|
| High completeness | 2134 | 54.49% |
| Low completeness | 1782 | 45.51% |

## Q3. Final D1 / D2 distribution

Train totals: 12828 sentences; ROOT=2124; positive=10704; 1 event=4749; 2 events=3916; 3+ events=2039; 2-event High=2134; 2-event Low=1782.

| Group | Rule | Sentences | % of positive | % of all train |
|---|---|---:|---:|---:|
| D1 | 1-event + 2-event High | 6883 | 64.30% | 53.66% |
| D2 | 2-event Low + 3+ event | 3821 | 35.70% | 29.79% |
| ROOT | 0-event | 2124 | N/A | 16.56% |

## Q4. Balance and source of imbalance

D1:D2 = 6883:3821 = 1.801:1. not severe by 70:30 criterion; no threshold adjustment was made. D1 has 4749 one-event sentences plus 2134 two-event High; D2 has 1782 two-event Low plus 2039 3+ sentences. The main source of the D1 excess is the 1-event versus 3+ pool size difference (2710 samples); the median split itself contributes only a 352 sample High-over-Low difference.

## Q5. Implementation diagnostics

For 2-event sentences, median(min/mean/max event percentile) = 0.368763/0.542601/0.690467. Median within-sentence max-minus-min gap is 0.273816; 75 pairs have one event below 0.25 and the other at or above 0.75.

Percentile ties are expected because completeness is discrete. Per-type event count, unique percentile count, largest tied block and its share:

| Type | Events | Unique percentiles | Largest tie | Tie share |
|---|---:|---:|---:|---:|
| CMP | 1536 | 8 | 522 | 33.98% |
| FAC | 2528 | 6 | 1080 | 42.72% |
| FIN | 1480 | 2 | 786 | 53.11% |
| ITT | 1400 | 3 | 1322 | 94.43% |
| MDS | 2385 | 5 | 1363 | 57.15% |
| PRP | 2094 | 3 | 1090 | 52.05% |
| PUR | 3320 | 4 | 2693 | 81.11% |
| RWF | 1549 | 6 | 820 | 52.94% |
| RWS | 463 | 6 | 208 | 44.92% |
| WKS | 2733 | 6 | 1252 | 45.81% |

In the 2-event min-score distribution, 278 samples lie exactly at the threshold, so the split is determined by the documented tie rule rather than forced to 50:50. High/Low counts and percentile bins are shown above. Type-specific tie blocks are preserved; the assignment does not randomly split event-level ties.

## Verification

2-event sentence rows=3916; D1+D2=10704 equals positive sentences=10704; ROOT+D1+D2=12828 equals total=12828. Percentile range/order, event-count joins, and bin totals are asserted in two_event_curriculum_split_verification.json.

No model training/inference, dev/test data, redefinition of completeness, or threshold tuning was used.
