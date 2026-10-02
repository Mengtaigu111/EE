# Train Appendix F.2 Schema Completeness

## Q1. Overall gold event distribution

Train has 12,828 target sentences and 19,488 gold events. Mean 0.4894; median 0.4000; SD 0.2170; P25/P75 0.3333/0.6000; P10/P90 0.2857/0.8000. Full schema: 1,795 (9.21%).

| Completeness | Events | Percent |
|---:|---:|---:|
| 0.0000 | 50 | 0.26% |
| 0.1429 | 219 | 1.12% |
| 0.1667 | 409 | 2.10% |
| 0.2000 | 364 | 1.87% |
| 0.2857 | 1287 | 6.60% |
| 0.3333 | 5086 | 26.10% |
| 0.4000 | 2434 | 12.49% |
| 0.4286 | 1459 | 7.49% |
| 0.5000 | 1275 | 6.54% |
| 0.5714 | 747 | 3.83% |
| 0.6000 | 1995 | 10.24% |
| 0.6667 | 1697 | 8.71% |
| 0.7143 | 290 | 1.49% |
| 0.8000 | 320 | 1.64% |
| 0.8333 | 4 | 0.02% |
| 0.8571 | 57 | 0.29% |
| 1.0000 | 1795 | 9.21% |

## Q2. By event type

| Type | Events | Mean | Median | SD | P25 | P75 | Full | Full % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PUR | 3320 | 0.3947 | 0.3333 | 0.1327 | 0.3333 | 0.3333 | 4 | 0.12% |
| PRP | 2094 | 0.8268 | 1.0000 | 0.1914 | 0.6667 | 1.0000 | 1090 | 52.05% |
| WKS | 2733 | 0.5149 | 0.6000 | 0.1484 | 0.4000 | 0.6000 | 3 | 0.11% |
| ITT | 1400 | 0.3500 | 0.3333 | 0.0769 | 0.3333 | 0.3333 | 0 | 0.00% |
| RWS | 463 | 0.4460 | 0.5000 | 0.1336 | 0.3333 | 0.5000 | 0 | 0.00% |
| RWF | 1549 | 0.3218 | 0.3333 | 0.1277 | 0.1667 | 0.3333 | 0 | 0.00% |
| MDS | 2385 | 0.4592 | 0.4000 | 0.1333 | 0.4000 | 0.6000 | 0 | 0.00% |
| FIN | 1480 | 0.7345 | 0.5000 | 0.2495 | 0.5000 | 1.0000 | 694 | 46.89% |
| CMP | 1536 | 0.5120 | 0.5714 | 0.1544 | 0.4286 | 0.5714 | 4 | 0.26% |
| FAC | 2528 | 0.3658 | 0.2857 | 0.1222 | 0.2857 | 0.4286 | 0 | 0.00% |

## Q3. Most common role-set realizations (Top 10/type)

| Type | Rank | Observed roles | Events | Within type | Completeness |
|---|---:|---|---:|---:|---:|
| PUR | 1 | {Aim} | 2671 | 80.45% | 33.33% |
| PUR | 2 | {Aim, Condition} | 600 | 18.07% | 66.67% |
| PUR | 3 | {Condition} | 18 | 0.54% | 33.33% |
| PUR | 4 | {Aim, Dataset} | 13 | 0.39% | 66.67% |
| PUR | 5 | {} | 10 | 0.30% | 0.00% |
| PUR | 6 | {Aim, Condition, Dataset} | 4 | 0.12% | 100.00% |
| PUR | 7 | {Dataset} | 4 | 0.12% | 33.33% |
| PRP | 1 | {Proposer, Content, Target} | 1090 | 52.05% | 100.00% |
| PRP | 2 | {Proposer, Content} | 805 | 38.44% | 66.67% |
| PRP | 3 | {Content, Target} | 111 | 5.30% | 66.67% |
| PRP | 4 | {Content} | 81 | 3.87% | 33.33% |
| PRP | 5 | {Proposer, Target} | 4 | 0.19% | 66.67% |
| PRP | 6 | {Proposer} | 2 | 0.10% | 33.33% |
| PRP | 7 | {Target} | 1 | 0.05% | 33.33% |
| WKS | 1 | {Researcher, Content, Target} | 682 | 24.95% | 60.00% |
| WKS | 2 | {Researcher, Content} | 632 | 23.12% | 40.00% |
| WKS | 3 | {Researcher, Content, Condition} | 370 | 13.54% | 60.00% |
| WKS | 4 | {Content, Target} | 244 | 8.93% | 40.00% |
| WKS | 5 | {Researcher, Content, Condition, Target} | 186 | 6.81% | 80.00% |
| WKS | 6 | {Content} | 159 | 5.82% | 20.00% |
| WKS | 7 | {Content, Condition} | 140 | 5.12% | 40.00% |
| WKS | 8 | {Researcher, Content, Dataset} | 91 | 3.33% | 60.00% |
| WKS | 9 | {Content, Condition, Target} | 80 | 2.93% | 60.00% |
| WKS | 10 | {Researcher, Content, Dataset, Target} | 32 | 1.17% | 80.00% |
| ITT | 1 | {Target} | 1322 | 94.43% | 33.33% |
| ITT | 2 | {Target, Condition} | 71 | 5.07% | 66.67% |
| ITT | 3 | {} | 4 | 0.29% | 0.00% |
| ITT | 4 | {Target, Dataset} | 3 | 0.21% | 66.67% |
| RWS | 1 | {BaseComponent, TriedComponent} | 72 | 15.55% | 33.33% |
| RWS | 2 | {Subject, Target, BaseComponent} | 54 | 11.66% | 50.00% |
| RWS | 3 | {Subject, BaseComponent, TriedComponent} | 53 | 11.45% | 50.00% |
| RWS | 4 | {Target, BaseComponent, TriedComponent} | 40 | 8.64% | 50.00% |
| RWS | 5 | {Subject, Target, BaseComponent, TriedComponent} | 32 | 6.91% | 66.67% |
| RWS | 6 | {Target, BaseComponent} | 32 | 6.91% | 33.33% |
| RWS | 7 | {Subject, BaseComponent} | 31 | 6.70% | 33.33% |
| RWS | 8 | {Subject, Target, TriedComponent} | 19 | 4.10% | 50.00% |
| RWS | 9 | {Subject, BaseComponent, Condition} | 16 | 3.46% | 50.00% |
| RWS | 10 | {BaseComponent} | 13 | 2.81% | 16.67% |
| RWF | 1 | {Concern, Fault} | 603 | 38.93% | 33.33% |
| RWF | 2 | {Fault} | 273 | 17.62% | 16.67% |
| RWF | 3 | {Target, Concern, Fault} | 137 | 8.84% | 50.00% |
| RWF | 4 | {Target, Fault} | 106 | 6.84% | 33.33% |
| RWF | 5 | {Concern} | 100 | 6.46% | 16.67% |
| RWF | 6 | {Concern, Fault, Condition} | 95 | 6.13% | 50.00% |
| RWF | 7 | {Fault, Condition} | 44 | 2.84% | 33.33% |
| RWF | 8 | {} | 31 | 2.00% | 0.00% |
| RWF | 9 | {Target, Concern} | 25 | 1.61% | 33.33% |
| RWF | 10 | {Concern, Condition} | 19 | 1.23% | 33.33% |
| MDS | 1 | {BaseComponent, TriedComponent} | 692 | 29.01% | 40.00% |
| MDS | 2 | {BaseComponent, TriedComponent, Target} | 459 | 19.25% | 60.00% |
| MDS | 3 | {BaseComponent, Target} | 421 | 17.65% | 40.00% |
| MDS | 4 | {BaseComponent, TriedComponent, Condition} | 185 | 7.76% | 60.00% |
| MDS | 5 | {TriedComponent, Target} | 128 | 5.37% | 40.00% |
| MDS | 6 | {BaseComponent} | 126 | 5.28% | 20.00% |
| MDS | 7 | {BaseComponent, Condition} | 78 | 3.27% | 40.00% |
| MDS | 8 | {BaseComponent, TriedComponent, Condition, Target} | 77 | 3.23% | 80.00% |
| MDS | 9 | {BaseComponent, Condition, Target} | 64 | 2.68% | 60.00% |
| MDS | 10 | {TriedComponent} | 48 | 2.01% | 20.00% |
| FIN | 1 | {Content} | 785 | 53.04% | 50.00% |
| FIN | 2 | {Finder, Content} | 694 | 46.89% | 100.00% |
| FIN | 3 | {Finder} | 1 | 0.07% | 50.00% |
| CMP | 1 | {Arg1, Result, Arg2} | 228 | 14.84% | 42.86% |
| CMP | 2 | {Metrics, Arg1, Result, Arg2} | 107 | 6.97% | 57.14% |
| CMP | 3 | {Arg1, Result} | 81 | 5.27% | 28.57% |
| CMP | 4 | {Arg1, Result, Arg2, Condition} | 81 | 5.27% | 57.14% |
| CMP | 5 | {Metrics, Arg1, Extent, Result, Arg2} | 76 | 4.95% | 71.43% |
| CMP | 6 | {Arg1, Extent, Result, Arg2} | 59 | 3.84% | 57.14% |
| CMP | 7 | {Metrics, Arg1, Result} | 54 | 3.52% | 42.86% |
| CMP | 8 | {Arg1, Arg2} | 52 | 3.39% | 28.57% |
| CMP | 9 | {Result, Arg2} | 52 | 3.39% | 28.57% |
| CMP | 10 | {Metrics, Arg1, Arg2} | 49 | 3.19% | 42.86% |
| FAC | 1 | {Subject, Object} | 598 | 23.66% | 28.57% |
| FAC | 2 | {Subject, Object, Condition} | 397 | 15.70% | 42.86% |
| FAC | 3 | {Subject, Object, Target} | 204 | 8.07% | 42.86% |
| FAC | 4 | {Subject, Condition} | 174 | 6.88% | 28.57% |
| FAC | 5 | {Subject} | 150 | 5.93% | 14.29% |
| FAC | 6 | {Subject, Target} | 120 | 4.75% | 28.57% |
| FAC | 7 | {Subject, Extent, Object} | 87 | 3.44% | 42.86% |
| FAC | 8 | {Subject, Object, Dataset} | 75 | 2.97% | 42.86% |
| FAC | 9 | {Subject, Extent, Object, Condition} | 66 | 2.61% | 57.14% |
| FAC | 10 | {Subject, Object, Target, Condition} | 64 | 2.53% | 57.14% |

## Q4. Role presence and missingness

| Type | Role | Present | Present % | Missing | Missing % |
|---|---|---:|---:|---:|---:|
| PUR | Aim | 3288 | 99.04% | 32 | 0.96% |
| PUR | Condition | 622 | 18.73% | 2698 | 81.27% |
| PUR | Dataset | 21 | 0.63% | 3299 | 99.37% |
| PRP | Proposer | 1901 | 90.78% | 193 | 9.22% |
| PRP | Content | 2087 | 99.67% | 7 | 0.33% |
| PRP | Target | 1206 | 57.59% | 888 | 42.41% |
| WKS | Researcher | 2067 | 75.63% | 666 | 24.37% |
| WKS | Content | 2670 | 97.69% | 63 | 2.31% |
| WKS | Condition | 820 | 30.00% | 1913 | 70.00% |
| WKS | Dataset | 200 | 7.32% | 2533 | 92.68% |
| WKS | Target | 1279 | 46.80% | 1454 | 53.20% |
| ITT | Target | 1396 | 99.71% | 4 | 0.29% |
| ITT | Condition | 71 | 5.07% | 1329 | 94.93% |
| ITT | Dataset | 3 | 0.21% | 1397 | 99.79% |
| RWS | Subject | 250 | 54.00% | 213 | 46.00% |
| RWS | Target | 229 | 49.46% | 234 | 50.54% |
| RWS | BaseComponent | 396 | 85.53% | 67 | 14.47% |
| RWS | TriedComponent | 276 | 59.61% | 187 | 40.39% |
| RWS | Condition | 78 | 16.85% | 385 | 83.15% |
| RWS | Dataset | 10 | 2.16% | 453 | 97.84% |
| RWF | Target | 332 | 21.43% | 1217 | 78.57% |
| RWF | Extent | 56 | 3.62% | 1493 | 96.38% |
| RWF | Concern | 1040 | 67.14% | 509 | 32.86% |
| RWF | Fault | 1343 | 86.70% | 206 | 13.30% |
| RWF | Condition | 209 | 13.49% | 1340 | 86.51% |
| RWF | Dataset | 11 | 0.71% | 1538 | 99.29% |
| MDS | BaseComponent | 2118 | 88.81% | 267 | 11.19% |
| MDS | TriedComponent | 1658 | 69.52% | 727 | 30.48% |
| MDS | Condition | 471 | 19.75% | 1914 | 80.25% |
| MDS | Dataset | 25 | 1.05% | 2360 | 98.95% |
| MDS | Target | 1204 | 50.48% | 1181 | 49.52% |
| FIN | Finder | 695 | 46.96% | 785 | 53.04% |
| FIN | Content | 1479 | 99.93% | 1 | 0.07% |
| CMP | Metrics | 652 | 42.45% | 884 | 57.55% |
| CMP | Arg1 | 1300 | 84.64% | 236 | 15.36% |
| CMP | Extent | 487 | 31.71% | 1049 | 68.29% |
| CMP | Result | 1221 | 79.49% | 315 | 20.51% |
| CMP | Arg2 | 1204 | 78.39% | 332 | 21.61% |
| CMP | Condition | 403 | 26.24% | 1133 | 73.76% |
| CMP | Dataset | 238 | 15.49% | 1298 | 84.51% |
| FAC | Subject | 2270 | 89.79% | 258 | 10.21% |
| FAC | Extent | 382 | 15.11% | 2146 | 84.89% |
| FAC | Object | 1878 | 74.29% | 650 | 25.71% |
| FAC | Target | 677 | 26.78% | 1851 | 73.22% |
| FAC | Condition | 964 | 38.13% | 1564 | 61.87% |
| FAC | Dataset | 276 | 10.92% | 2252 | 89.08% |
| FAC | Reason | 26 | 1.03% | 2502 | 98.97% |

## Q5. Sentence-level minimum completeness

Zero-event sentences have NA completeness. The following gives the exact observed value distribution within each event-count group.

| Events | Min completeness | Sentences | Within-group % |
|---|---:|---:|---:|
| 1 | 0.0000 | 22 | 0.46% |
| 1 | 0.1429 | 47 | 0.99% |
| 1 | 0.1667 | 153 | 3.22% |
| 1 | 0.2000 | 89 | 1.87% |
| 1 | 0.2857 | 251 | 5.29% |
| 1 | 0.3333 | 1681 | 35.40% |
| 1 | 0.4000 | 590 | 12.42% |
| 1 | 0.4286 | 290 | 6.11% |
| 1 | 0.5000 | 186 | 3.92% |
| 1 | 0.5714 | 162 | 3.41% |
| 1 | 0.6000 | 472 | 9.94% |
| 1 | 0.6667 | 436 | 9.18% |
| 1 | 0.7143 | 94 | 1.98% |
| 1 | 0.8000 | 77 | 1.62% |
| 1 | 0.8333 | 1 | 0.02% |
| 1 | 0.8571 | 18 | 0.38% |
| 1 | 1.0000 | 180 | 3.79% |
| 2 | 0.0000 | 11 | 0.28% |
| 2 | 0.1429 | 81 | 2.07% |
| 2 | 0.1667 | 150 | 3.83% |
| 2 | 0.2000 | 117 | 2.99% |
| 2 | 0.2857 | 409 | 10.44% |
| 2 | 0.3333 | 1463 | 37.36% |
| 2 | 0.4000 | 506 | 12.92% |
| 2 | 0.4286 | 431 | 11.01% |
| 2 | 0.5000 | 187 | 4.78% |
| 2 | 0.5714 | 101 | 2.58% |
| 2 | 0.6000 | 203 | 5.18% |
| 2 | 0.6667 | 218 | 5.57% |
| 2 | 0.7143 | 30 | 0.77% |
| 2 | 0.8000 | 2 | 0.05% |
| 2 | 0.8571 | 6 | 0.15% |
| 2 | 1.0000 | 1 | 0.03% |
| 3+ | 0.0000 | 10 | 0.49% |
| 3+ | 0.1429 | 82 | 4.02% |
| 3+ | 0.1667 | 62 | 3.04% |
| 3+ | 0.2000 | 116 | 5.69% |
| 3+ | 0.2857 | 372 | 18.24% |
| 3+ | 0.3333 | 960 | 47.08% |
| 3+ | 0.4000 | 149 | 7.31% |
| 3+ | 0.4286 | 170 | 8.34% |
| 3+ | 0.5000 | 32 | 1.57% |
| 3+ | 0.5714 | 34 | 1.67% |
| 3+ | 0.6000 | 41 | 2.01% |
| 3+ | 0.6667 | 6 | 0.29% |
| 3+ | 0.7143 | 3 | 0.15% |
| 3+ | 0.8571 | 1 | 0.05% |
| 3+ | 1.0000 | 1 | 0.05% |

The next table groups the same values by train-positive-sentence terciles; these are descriptive groups, not curriculum thresholds.

Cuts: Low ≤ 0.3333; Medium > 0.3333 and ≤ 0.4000; High > 0.4000.

| Events | High n (%) | Medium n (%) | Low n (%) | Total |
|---|---:|---:|---:|---:|
| 1 | 1916 (40.35%) | 590 (12.42%) | 2243 (47.23%) | 4749 |
| 2 | 1179 (30.11%) | 506 (12.92%) | 2231 (56.97%) | 3916 |
| 3+ | 288 (14.12%) | 149 (7.31%) | 1602 (78.57%) | 2039 |

## Q6. Raw completeness as curriculum difficulty

Raw completeness is a role-coverage ratio, not a calibrated cross-type difficulty scale. Types with more legal roles have finer possible score increments; one absent role changes the score by 1/|schema|. A role that rarely appears in the training annotations can mechanically lower completeness for its type, without implying that the role is unimportant. The exact-value distribution shows how much scores concentrate on discrete fractions. Since type-specific baselines, schema sizes, and role presence differ, raw values are not automatically comparable between types. Within-type midrank percentiles place each type near the same 50th-percentile center by construction; this removes type baselines but also discards absolute role coverage. For illustration, compare each type’s raw mean and average within-type percentile below. The percentile is diagnostic only and is not used for curriculum labels.

| Type | Raw mean completeness | Mean within-type percentile |
|---|---:|---:|
| PUR | 0.3947 | 0.5000 |
| PRP | 0.8268 | 0.5000 |
| WKS | 0.5149 | 0.5000 |
| ITT | 0.3500 | 0.5000 |
| RWS | 0.4460 | 0.5000 |
| RWF | 0.3218 | 0.5000 |
| MDS | 0.4592 | 0.5000 |
| FIN | 0.7345 | 0.5000 |
| CMP | 0.5120 | 0.5000 |
| FAC | 0.3658 | 0.5000 |

## Reproducibility and scope

Uses only original sentence-C train gold plus the supplied Appendix F.2 reference. No dev/test data, predictions, model inference, role weights, or source-data edits. Quantile groups are descriptive, not proposed thresholds.
