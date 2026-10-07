# Type-CoT1 → Sorted1 dense online dev

Fresh original Qwen3-8B/new LoRA; user-final CoT prompt, one full shuffled epoch. Only this run CoT weights transfer to a new tree optimizer/scheduler/controller. Original AppendixNoCopy data/course; denser full-dev checks; official offset dev AC controls training.

Status: completed; CoT updates 802; tree updates 3208; best tree step 2205; stop early_stop; smoke False.

| tree step | dev TI | TC | AI | AC | EC_I | EC_C | LR after | bad count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 401 | 55.1859 | 37.1387 | 23.4323 | 22.7670 | 10.8716 | 10.8716 | 0.0001 | 0 |
| 802 | 68.3484 | 52.1036 | 35.4288 | 32.7583 | 35.0540 | 35.0540 | 0.0001 | 0 |
| 1203 | 75.0152 | 62.6629 | 43.5403 | 40.7538 | 46.0485 | 46.0485 | 0.0001 | 0 |
| 1403 | 74.3366 | 62.0840 | 42.8596 | 40.4483 | 47.1121 | 47.1121 | 0.0001 | 1 |
| 1604 | 75.2573 | 62.4010 | 43.4646 | 40.4278 | 47.6357 | 47.6357 | 0.0001 | 2 |
| 1804 | 75.0203 | 62.8885 | 43.6611 | 40.6926 | 47.5688 | 47.5688 | 5e-05 | 3 |
| 2005 | 74.9951 | 62.4161 | 43.6133 | 41.0190 | 47.5756 | 47.4615 | 5e-05 | 0 |
| 2205 | 74.3992 | 62.4791 | 43.5104 | 41.0751 | 46.7081 | 46.7081 | 5e-05 | 1 |
| 2406 | 74.7427 | 62.5421 | 43.0444 | 40.2587 | 47.9226 | 47.9226 | 5e-05 | 2 |
| 2606 | 74.0077 | 61.7893 | 43.1094 | 40.2284 | 45.9841 | 45.9841 | 2.5e-05 | 3 |
| 2807 | 75.0704 | 62.7340 | 42.9121 | 40.5670 | 46.9742 | 46.9742 | 2.5e-05 | 4 |
| 3007 | 75.1862 | 62.9861 | 42.5263 | 39.7544 | 46.7501 | 46.7501 | 2.5e-05 | 5 |
| 3208 | 75.6789 | 63.5028 | 43.3913 | 40.8409 | 48.1524 | 48.0370 | 2.5e-05 | 6 |

| test mode | TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|
| official_offset | 75.5592 | 64.1541 | 44.7401 | 42.4606 | 47.9205 | 47.9205 |
| official_raw_token | 76.2319 | 63.8113 | 44.3056 | 42.0349 | 46.0111 | 46.0111 |
| instance_offset | 75.0467 | 63.8042 | 44.8732 | 42.6180 | 47.6368 | 47.6368 |
| instance_raw_token | 75.1658 | 63.9303 | 45.4553 | 43.1852 | 47.8315 | 47.8315 |

CoT training seconds 9045.22; tree training+dev seconds 45924.46.
CoT dev is a type-multiset diagnostic only. Test never controls training. A no-CoT run with this same feedback schedule is required to isolate CoT effects.
