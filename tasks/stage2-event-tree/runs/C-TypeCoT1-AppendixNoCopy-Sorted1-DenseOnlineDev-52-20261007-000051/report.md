# Type-CoT1 → Sorted1 dense online dev

Fresh original Qwen3-8B/new LoRA; user-final CoT prompt, one full shuffled epoch. Only this run CoT weights transfer to a new tree optimizer/scheduler/controller. Original AppendixNoCopy data/course; denser full-dev checks; official offset dev AC controls training.

Status: completed; CoT updates 802; tree updates 3208; best tree step 2406; stop budget_exhausted; smoke False.

| tree step | dev TI | TC | AI | AC | EC_I | EC_C | LR after | bad count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 401 | 51.8628 | 34.0752 | 21.1115 | 20.2359 | 10.7206 | 10.7206 | 0.0001 | 0 |
| 802 | 68.6937 | 52.2698 | 34.6444 | 32.1520 | 36.4641 | 36.4641 | 0.0001 | 0 |
| 1203 | 72.7195 | 60.4741 | 42.1552 | 39.5087 | 43.2397 | 43.2397 | 0.0001 | 0 |
| 1403 | 75.0656 | 61.5851 | 42.7515 | 40.3339 | 45.4284 | 45.3134 | 0.0001 | 0 |
| 1604 | 74.5393 | 60.8609 | 42.4078 | 39.8253 | 45.2986 | 45.2986 | 0.0001 | 1 |
| 1804 | 73.5342 | 60.3589 | 41.6420 | 39.0991 | 44.1832 | 44.1832 | 0.0001 | 2 |
| 2005 | 74.2335 | 61.2569 | 42.4123 | 40.1220 | 47.3531 | 47.3531 | 5e-05 | 3 |
| 2205 | 75.2779 | 62.4696 | 42.3074 | 39.9253 | 46.9965 | 46.9965 | 5e-05 | 4 |
| 2406 | 75.2337 | 61.9797 | 42.9887 | 40.5186 | 47.2222 | 47.2222 | 5e-05 | 0 |
| 2606 | 74.8181 | 61.5104 | 41.9886 | 39.6817 | 47.0171 | 47.0171 | 5e-05 | 1 |
| 2807 | 74.7082 | 62.0336 | 42.1541 | 39.7214 | 47.5091 | 47.5091 | 5e-05 | 2 |
| 3007 | 73.8168 | 61.5792 | 41.7673 | 39.2809 | 46.2379 | 46.2379 | 2.5e-05 | 3 |
| 3208 | 74.8332 | 62.2924 | 42.3117 | 39.9547 | 47.7086 | 47.7086 | 2.5e-05 | 4 |

| test mode | TI | TC | AI | AC | EC_I | EC_C |
|---|---:|---:|---:|---:|---:|---:|
| official_offset | 75.0246 | 63.1060 | 43.6123 | 40.9853 | 47.0862 | 47.0862 |
| official_raw_token | 75.7504 | 62.7637 | 43.1753 | 40.5693 | 45.3596 | 45.3596 |
| instance_offset | 74.3984 | 62.6824 | 43.6275 | 41.0317 | 46.6394 | 46.6394 |
| instance_raw_token | 74.5268 | 62.7760 | 44.2343 | 41.6234 | 46.7986 | 46.7986 |

CoT training seconds 9050.76; tree training+dev seconds 45786.14.
Seed52; CoT dev diagnostic omitted. Test never controls training. A no-CoT run with this same feedback schedule is required to isolate CoT effects.
