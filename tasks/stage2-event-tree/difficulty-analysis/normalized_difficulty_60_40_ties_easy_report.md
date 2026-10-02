# Normalized difficulty 60%:40% split

Positive=10704; ROOT=2124; B1=6699 (62.5841%); B2=4005 (37.4159%).

Round the requested easy count to the nearest integer; sort by normalized score, then sample_id. ROOT remains unscored.

| Bucket | 1 event | 2 events | 3 events | 4+ events |
|---|---:|---:|---:|---:|
| B1 | 4749 (70.89%) | 1856 (27.71%) | 84 (1.25%) | 10 (0.15%) |
| B2 | 0 (0.00%) | 2060 (51.44%) | 1357 (33.88%) | 588 (14.68%) |

Boundary=0.545707831; tied=1037, B1/B2=1037/0. Tie splitting is administrative only.

Tie-preserving alternative below/equal/above={'below': 5662, 'equal': 1037, 'above': 4005}.

| Event type | B1 events | B2 events | % of type in B1 |
|---|---:|---:|---:|
| CMP | 559 | 977 | 36.39% |
| FAC | 839 | 1689 | 33.19% |
| FIN | 209 | 1271 | 14.12% |
| ITT | 1343 | 57 | 95.93% |
| MDS | 685 | 1700 | 28.72% |
| PRP | 1092 | 1002 | 52.15% |
| PUR | 1384 | 1936 | 41.69% |
| RWF | 1031 | 518 | 66.56% |
| RWS | 241 | 222 | 52.05% |
| WKS | 1370 | 1363 | 50.13% |

Type B1 assignment-rate range=81.81%; agreement with old 50:50 buckets=87.42%; agreement with manual D1/D2=96.52%.

Boundary tie policy=easy. The difficulty formula and scores are unchanged.
