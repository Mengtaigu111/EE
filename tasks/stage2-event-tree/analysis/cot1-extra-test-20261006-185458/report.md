# CoT1 supplementary test checkpoints1604/1804/3208

All rows use fixed official offset scoring, greedy batch4/max_new_tokens319. Original dev-selected checkpoint2205 and history are unchanged.

| checkpoint | TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|
| 1604 | 75.6388 | 62.4780 | 43.8068 | 41.2499 | 47.2643 | 47.2643 |
| 1804 | 75.8387 | 64.0846 | 45.1084 | 42.4080 | 49.0308 | 49.0308 |
| 2205 | 75.5592 | 64.1541 | 44.7401 | 42.4606 | 47.9205 | 47.9205 |
| 3208 | 75.3898 | 62.8628 | 43.2195 | 40.8422 | 46.8085 | 46.5785 |

All18 P/R/F1, four separate scoring modes, raw-output consistency, input/adapter hashes and7 complex metrics independently verified per checkpoint in state.json.

Batch8 benchmark on48 test examples changed2 raw/parsed trees and only saved7.35% time; batch4 was retained for consistency with existing test protocol.
