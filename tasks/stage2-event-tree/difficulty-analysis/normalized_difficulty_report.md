# Normalized Curriculum Difficulty on Train

Fixed formula: DIF_norm = 0.5 * [(1 − 1/N_e) + (1 − P_min)]. Percentiles are train-only type-specific tie-aware midranks. Lower is easier. ROOT has NA.

## Q1. Overall normalized score

| Statistic | Value |
|---|---:|
| Count | 10704.000000 |
| Mean | 0.451534 |
| SD | 0.210049 |
| Min | 0.000161 |
| Max | 0.918018 |
| P10 | 0.158800 |
| P25 | 0.262500 |
| P50 | 0.482731 |
| P75 | 0.621295 |
| P90 | 0.704629 |

| DIF bin | Sentences | Percent |
|---|---:|---:|
| [0.0,0.1) | 565 | 5.28% |
| [0.1,0.2) | 923 | 8.62% |
| [0.2,0.3) | 1653 | 15.44% |
| [0.3,0.4) | 1565 | 14.62% |
| [0.4,0.5) | 878 | 8.20% |
| [0.5,0.6) | 1538 | 14.37% |
| [0.6,0.7) | 2066 | 19.30% |
| [0.7,0.8) | 1247 | 11.65% |
| [0.8,0.9) | 263 | 2.46% |
| [0.9,1.0] | 6 | 0.06% |

## Q2. Event count and normalized difficulty

| Event count | Samples | Mean | Median | P25 | P75 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4749 | 0.2550 | 0.2625 | 0.1588 | 0.3542 | 0.0002 | 0.4999 |
| 2 | 3916 | 0.5599 | 0.5656 | 0.5457 | 0.6201 | 0.2548 | 0.7492 |
| 3 | 1441 | 0.6829 | 0.7006 | 0.6290 | 0.7046 | 0.3433 | 0.8332 |
| 4+ | 598 | 0.7450 | 0.7432 | 0.6906 | 0.7713 | 0.4922 | 0.9180 |

### Adjacent-count overlap

| Pair | A min–max | B min–max | Shared range | A samples in range | B samples in range | Exact shared score values |
|---|---:|---:|---:|---:|---:|---:|
| 1_vs_2 | 0.0002–0.4999 | 0.2548–0.7492 | 0.2548–0.4999 | 2882 | 772 | 1 |
| 2_vs_3 | 0.2548–0.7492 | 0.3433–0.8332 | 0.3433–0.7492 | 3824 | 1201 | 0 |

Counts on both sides within a shared score range demonstrate realized overlap. Integer boundaries are not forced by event count; actual score distributions determine overlap.

## Q3. Equal-size B1/B2 and event-count composition

B1=5,352; B2=5,352. Boundary score=0.482731; 112 samples tie there, split B1/B2=40/72. This tie split is administrative only; ordering is by normalized score then sample_id. Tie-preserving alternative (below/equal/above)=5312/112/5280.

| Formula/bucket | 1 event | 2 events | 3 events | 4+ events | Total |
|---|---:|---:|---:|---:|---:|
| B1 | 4630 (86.51%) | 700 (13.08%) | 22 (0.41%) | 0 (0.00%) | 5352 |
| B2 | 119 (2.22%) | 3216 (60.09%) | 1419 (26.51%) | 598 (11.17%) | 5352 |
| Old B1 | 4749 (88.73%) | 603 (11.27%) | 0 (0.00%) | 0 (0.00%) | 5352 |
| Old B2 | 0 (0.00%) | 3313 (61.90%) | 1441 (26.92%) | 598 (11.17%) | 5352 |

The old composition rows are recomputed from the prior bucket assignment CSV; they are shown as a direct comparison.

## Q4. Event-type assignment skew

Rates use all events of a type as the denominator.

| Type | B1 events | % type in B1 | B2 events | % type in B2 |
|---|---:|---:|---:|---:|
| PUR | 316 | 9.52% | 3004 | 90.48% |
| PRP | 697 | 33.29% | 1397 | 66.71% |
| WKS | 943 | 34.50% | 1790 | 65.50% |
| ITT | 1288 | 92.00% | 112 | 8.00% |
| RWS | 151 | 32.61% | 312 | 67.39% |
| RWF | 774 | 49.97% | 775 | 50.03% |
| MDS | 487 | 20.42% | 1898 | 79.58% |
| FIN | 170 | 11.49% | 1310 | 88.51% |
| CMP | 517 | 33.66% | 1019 | 66.34% |
| FAC | 753 | 29.79% | 1775 | 70.21% |

B1 type-rate max/min/range/SD: old=92.29%/8.86%/83.43%/0.2244; normalized=92.00%/9.52%/82.48%/0.2219. Range changed by -0.95 percentage points and SD by -0.0025; the shift is small, and type skew remains.

## Q5. Component influence and old/new buckets


Spearman(normalized DIF, event-only)=0.8790; Spearman(normalized DIF, schema-only)=0.7390. The new correlation gap is 0.1400, versus the old-formula reference gap of 0.2749 (0.9293−0.6544): event count still has the stronger rank association, but the two are closer. These are rank associations, not causal component attributions. Spearman(old DIF, normalized DIF)=0.9840.

Old/new bucket agreement=97.78%.

| Old bucket | New B1 | New B2 |
|---|---:|---:|
| B1 | 5233 | 119 |
| B2 | 119 | 5233 |

Previous manual D1/D2 agreement=85.29%.

| Previous manual rule | New B1 | New B2 |
|---|---:|---:|
| D1 | 5330 | 1553 |
| D2 | 22 | 3799 |

## Q6. Cross-count ordering examples

These are score comparisons only; no semantic interpretation is attached.

### High-difficulty 1-event examples
- ee97bd265f3ab415ece8a1fc5fd35e8b::sentence-4: P_min=0.0002, DIF_norm=0.4999
- 09e8a58fe50453a2401747d5e9c40e18::sentence-0: P_min=0.0014, DIF_norm=0.4993
- 3a1b72322a7b1460c636945d1d260a22::sentence-0: P_min=0.0014, DIF_norm=0.4993
- 67cd94de1749af7c53b535135272fbdc::sentence-0: P_min=0.0014, DIF_norm=0.4993
- 91b8a0929d6aec4b1c17e9bfbd0c43d7::sentence-0: P_min=0.0014, DIF_norm=0.4993

### 2-event sentence easier than a 1-event sentence
- 2-event 030a7f4c3782cbc87643f2aa68a7594c::sentence-5 (0.2548) < 1-event ee97bd265f3ab415ece8a1fc5fd35e8b::sentence-4 (0.4999)
- 2-event 8a07d860794a96ad9a1c13ce9ac55ec1::sentence-8 (0.2548) < 1-event ee97bd265f3ab415ece8a1fc5fd35e8b::sentence-4 (0.4999)
- 2-event eee2471900b04ff861bd2e6494b99d6b::sentence-8 (0.2548) < 1-event ee97bd265f3ab415ece8a1fc5fd35e8b::sentence-4 (0.4999)
- 2-event c74f7ba287b1c1241b617c8ba1c5d951::sentence-2 (0.2550) < 1-event ee97bd265f3ab415ece8a1fc5fd35e8b::sentence-4 (0.4999)
- 2-event eed953c88714d4fe6759bbaec5c6aa8a::sentence-2 (0.2585) < 1-event ee97bd265f3ab415ece8a1fc5fd35e8b::sentence-4 (0.4999)

### 3-event sentence easier than a 2-event sentence
- 3-event 2fbb4456fc76dd83fc54b0c6bb439207::sentence-3 (0.3433) < 2-event cd88f204890aa2b5bc6e3eb1d1472d70::sentence-2 (0.7492)
- 3-event 2404839b071bfa1cc48ff2742de35495::sentence-6 (0.3688) < 2-event cd88f204890aa2b5bc6e3eb1d1472d70::sentence-2 (0.7492)
- 3-event 53ff141b141332aa9869994fbdacc838::sentence-6 (0.3688) < 2-event cd88f204890aa2b5bc6e3eb1d1472d70::sentence-2 (0.7492)
- 3-event 61350e040561e1b668b54905a8982d3a::sentence-6 (0.3801) < 2-event cd88f204890aa2b5bc6e3eb1d1472d70::sentence-2 (0.7492)
- 3-event b5f4dc81177eb036e5b4f4eafc728227::sentence-3 (0.3801) < 2-event cd88f204890aa2b5bc6e3eb1d1472d70::sentence-2 (0.7492)

## Q7–Q8. Interpretation and implementation checks

The normalized formula allows cross-count score overlap: inspect the observed ranges and intersection counts above. Component correlations and bucket type rates quantify whether event multiplicity still dominates or type skew remains; the formula itself was not adjusted. Percentile and boundary ties are preserved/reported, with only the 50% bucket boundary split administratively. ROOT remains unscored.

Counts reconcile: ROOT=2,124, positive=10,704, total=12,828; B1/B2=5,352/5,352. Verification details are in normalized_difficulty_verification.json.
