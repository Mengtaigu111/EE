# Sorted-first full-train results

First epoch sorted; four later epochs shuffled. Original AppendixNoCopy prompt, raw Qwen3-8B, one continuous 4010-update run, seed42.

| Offset dev checkpoint | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| 802 | 68.3402 | 51.7344 | 34.3421 | 31.9993 | 35.8792 |
| 1604 | 75.2103 | 61.7730 | 42.9995 | 40.3216 | 46.4789 |
| 2406 | 74.4803 | 61.9746 | 43.1546 | 40.5506 | 47.7595 |
| 3208 | 73.8972 | 61.2766 | 42.2059 | 39.8282 | 47.6647 |
| 4010 | 72.9167 | 61.3181 | 41.7236 | 38.9716 | 46.7153 |

Selected step 2406 by full dev offset AC F1; later-step tie.

| Selected test metric | TI | TC | AI | AC | EC |
|---|---:|---:|---:|---:|---:|
| offset | 75.1903 | 63.1004 | 44.1582 | 41.9811 | 48.1900 |
| raw token | 75.9053 | 62.8370 | 43.6837 | 41.5216 | 46.8803 |

Raw token uses original SciEvents gold, parsed raw predictions, no alignment or deduplication. Official shared-trigger limitations retained; single-seed exploratory comparison.

Corrected baseline test offset: {"TI_f1": 0.7541390728476821, "TC_f1": 0.6354791968536535, "AI_f1": 0.44203353357840935, "AC_f1": 0.41879652049143573, "EC_I_f1": 0.48068669527896996, "EC_C_f1": 0.48068669527896996, "TI_p": 0.764261744966443, "TC_p": 0.6441460344104071, "AI_p": 0.45345842531272995, "AC_p": 0.4297018770702981, "EC_I_p": 0.47058823529411764, "EC_C_p": 0.47058823529411764, "TI_r": 0.744281045751634, "TC_r": 0.6270424836601307, "AI_r": 0.4311701941577751, "AC_r": 0.4084309952772433, "EC_I_r": 0.49122807017543857, "EC_C_r": 0.49122807017543857}

Manifest: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/curriculum/sorted-first-20261003/manifest.json`; SHA256 `4c95a90ddebe565c82ed7e2fceb39c7e88467a3114dba6e0130db285e96aa3ff`.
Sampling audit: `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003/training/sampling-audit.json`.
