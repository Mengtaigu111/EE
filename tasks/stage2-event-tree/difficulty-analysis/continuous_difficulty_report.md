# Continuous Curriculum Difficulty on Train

DIF(s)=(N_e−1)+(1−P_min), where each event percentile is its train event-type midrank percentile and P_min is the least complete event percentile in the sentence. Lower DIF is easier. ROOT/no-event sentences have no score and remain outside positive buckets.

## Q1. Positive-sentence score distribution

| Statistic | Value |
|---|---:|
| Count | 10704.000000 |
| Mean | 1.408320 |
| SD | 1.027629 |
| Min | 0.000323 |
| Max | 8.631237 |
| P10 | 0.317600 |
| P25 | 0.525000 |
| P50 | 1.354818 |
| P75 | 1.742591 |
| P90 | 2.740210 |

| DIF bin | Sentences | Percent |
|---|---:|---:|
| [0,0.25) | 565 | 5.28% |
| [0.25,0.5) | 1302 | 12.16% |
| [0.5,0.75) | 2454 | 22.93% |
| [0.75,1.0) | 428 | 4.00% |
| [1.0,1.25) | 173 | 1.62% |
| [1.25,1.5) | 599 | 5.60% |
| [1.5,1.75) | 2638 | 24.64% |
| [1.75,2.0) | 506 | 4.73% |
| [2.0,2.5) | 100 | 0.93% |
| [2.5,3.0) | 1341 | 12.53% |
| [3.0,+inf) | 598 | 5.59% |

## Q1–Q2. Event count intervals and within-count ordering

For fixed event count, DIF=(N_e)−P_min, so lower P_min gives strictly higher DIF. Across event counts the coarse intervals are [N−1,N]; adjacent counts can share only integer endpoints. Endpoint cases are counted below.

| Event count | Samples | Mean DIF | Median DIF | P25 | P75 | DIF=N endpoint |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4749 | 0.5100 | 0.5250 | 0.3176 | 0.7085 | 0 |
| 2 | 3916 | 1.6198 | 1.6312 | 1.5914 | 1.7402 | 0 |
| 3 | 1441 | 2.6992 | 2.7345 | 2.5914 | 2.7426 | 0 |
| 4+ | 598 | 4.0468 | 3.7402 | 3.6803 | 3.9779 | 0 |

Raw counts for 4+ are retained: 4=456, 5=105, 6=29, 7=4, 8=3, 9=1.

## Q3. Equal-size buckets and boundary tie

B1 and B2 each contain 5,352 positive sentences. Boundary DIF=1.35481771; 69 samples equal the boundary, of which 26 are assigned to B1 and 43 to B2. This deterministic split exists only to make equal-size buckets; it does not imply a difficulty difference within the tie. Ordering is by DIF, then event count, then sample_id for stable output.

Tie-preserving alternative: 5326 strictly below (easy), 69 tied (kept as one group), 5309 strictly above (hard).

## Q4. Event-count composition

| Bucket | 1 event | 2 events | 3 events | 4+ events | Total |
|---|---:|---:|---:|---:|---:|
| B1 | 4749 (88.73%) | 603 (11.27%) | 0 (0.00%) | 0 (0.00%) | 5352 |
| B2 | 0 (0.00%) | 3313 (61.90%) | 1441 (26.92%) | 598 (11.17%) | 5352 |

## Q4. Event-type composition

Percentages are shares of all gold events within each bucket (sentences may contribute multiple events).

| Event type | B1 events | B1 % | B2 events | B2 % |
|---|---:|---:|---:|---:|
| PUR | 294 | 4.94% | 3026 | 22.36% |
| PRP | 718 | 12.06% | 1376 | 10.17% |
| WKS | 992 | 16.66% | 1741 | 12.86% |
| ITT | 1292 | 21.70% | 108 | 0.80% |
| RWS | 133 | 2.23% | 330 | 2.44% |
| RWF | 718 | 12.06% | 831 | 6.14% |
| MDS | 475 | 7.98% | 1910 | 14.11% |
| FIN | 147 | 2.47% | 1333 | 9.85% |
| CMP | 452 | 7.59% | 1084 | 8.01% |
| FAC | 734 | 12.33% | 1794 | 13.26% |

Composition differs by bucket: ITT is the largest B1 share (21.70%); PUR is the largest B2 share (22.36%). These are bucket event shares, not type-specific assignment rates.

Type assignment rates (denominator is all events of that type):

| Event type | B1 events | % of type in B1 | B2 events | % of type in B2 |
|---|---:|---:|---:|---:|
| PUR | 294 | 8.86% | 3026 | 91.14% |
| PRP | 718 | 34.29% | 1376 | 65.71% |
| WKS | 992 | 36.30% | 1741 | 63.70% |
| ITT | 1292 | 92.29% | 108 | 7.71% |
| RWS | 133 | 28.73% | 330 | 71.27% |
| RWF | 718 | 46.35% | 831 | 53.65% |
| MDS | 475 | 19.92% | 1910 | 80.08% |
| FIN | 147 | 9.93% | 1333 | 90.07% |
| CMP | 452 | 29.43% | 1084 | 70.57% |
| FAC | 734 | 29.03% | 1794 | 70.97% |

Type allocation is not uniform: 92.29% of ITT events fall in B1, while 91.14% of PUR and 90.07% of FIN events fall in B2. This is a visible composition skew induced by type-specific completeness profiles, not an assignment bug; it should be considered when interpreting buckets.

### Within-type percentile composition

| Bucket | Events | Mean percentile | Median | P25 | P75 |
|---|---:|---:|---:|---:|---:|
| B1 | 5955 | 0.5535 | 0.5345 | 0.3688 | 0.7397 |
| B2 | 13533 | 0.4764 | 0.4086 | 0.2915 | 0.6905 |

## Q5. Agreement with previous D1/D2 assignment

Positive-sentence agreement rate (previous D1↔new B1 and previous D2↔new B2): 85.70%.

Rows are previous labels and columns are new labels; alignment agreement counts D1↔B1 and D2↔B2.

| Previous rule | New B1 | New B2 |
|---|---:|---:|
| D1 | 5352 | 1531 |
| D2 | 0 | 3821 |

## Q6–Q8. Component influence, ranking and implementation risks

Spearman(DIF, event-count-only)=0.9293; Spearman(DIF, schema-only)=0.6544. Event count sets nonoverlapping coarse bands except shared integer endpoints; schema incompleteness orders samples within a fixed count.

Percentile composition: B1 mean/median/P25/P75=0.5535/0.5345/0.3688/0.7397; B2=0.4764/0.4086/0.2915/0.6905.

The two-event percentile distribution is discrete and has exact DIF ties; splitting at the 50% boundary is deterministic but membership among equal-score samples is an administrative bucket assignment, not a learned difficulty distinction. Schema percentile ties also arise because role completeness takes a small number of discrete values. ROOT is not assigned score zero. No formula weights or threshold were tuned.

Counts: total train=12,828, positive=10,704, ROOT=2,124, B1=5,352, B2=5,352.
