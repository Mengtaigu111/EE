# Sorted1 online dev training

Fresh original Qwen3-8B + new LoRA; unchanged AppendixNoCopy prompt/data/course. Online official offset dev AC controls plateau LR and stopping. 3208 update maximum and new schedule differ from historical 4010-update cosine runs; this is a combined online-feedback + four-epoch-limit experiment, single seed42.

Status: completed; best step 1604; last step 3208; stop reason budget_exhausted. Smoke: False.

| dev step | TI | TC | AI | AC | EC_I | EC_C | LR after | bad count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 401 | 48.9166 | 25.2868 | 13.2247 | 12.4885 | 7.0896 | 6.9030 | 0.0001 | 0 |
| 802 | 68.6957 | 51.0342 | 34.5833 | 32.1848 | 35.1830 | 35.0649 | 0.0001 | 0 |
| 1203 | 73.4152 | 61.2253 | 42.2211 | 39.4170 | 44.9784 | 44.9784 | 0.0001 | 0 |
| 1604 | 75.3803 | 62.1893 | 43.3266 | 40.6765 | 47.7928 | 47.7928 | 0.0001 | 0 |
| 2005 | 74.5562 | 61.4108 | 42.1763 | 39.6756 | 46.2754 | 46.2754 | 0.0001 | 1 |
| 2406 | 74.4288 | 61.6119 | 43.0481 | 40.6231 | 48.1159 | 48.1159 | 0.0001 | 2 |
| 2807 | 74.2660 | 61.0953 | 42.3294 | 40.4032 | 44.5913 | 44.5913 | 5e-05 | 3 |
| 3208 | 75.1829 | 62.1113 | 43.0719 | 40.1821 | 48.0969 | 48.0969 | 5e-05 | 4 |

| test mode | TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|
| official_offset | 75.5644 | 62.9058 | 44.3047 | 41.8031 | 48.6647 | 48.6647 |
| official_raw_token | 76.1697 | 62.5223 | 43.9470 | 41.4743 | 46.9545 | 46.9545 |
| instance_offset | 74.9454 | 62.4876 | 44.2452 | 41.7622 | 48.2143 | 48.2143 |
| instance_raw_token | 75.0744 | 62.6215 | 44.8751 | 42.3773 | 48.4884 | 48.4884 |

Instance scores are separate diagnostics; compare only identical scoring modes. All 18 P/R/F1 and instance counts are retained in run.json.

Historical official-offset references (unchanged predictions): baseline AC41.88%, Sorted1 AC41.98%; both dev-best2406. Do not attribute a difference only to early stopping.

Training + online-dev wall seconds: 38880.98; peak allocated GiB: 18.769.

Each dev step retains raw predictions, merged/aligned events, official metrics, diagnostics and complete safe state. Best and last adapters are separate.
