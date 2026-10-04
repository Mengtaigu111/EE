# Official token test comparison

Both dev-best checkpoint-2406, seed42. Raw sentence forests reparsed and merged into 250 abstracts; no alignment or deduplication. Official token scorer and CLI both checked.

| Metric | Baseline | Curriculum | Delta pp |
|---|---:|---:|---:|
| TI_f1 | 76.10 | 75.39 | -0.71 |
| TC_f1 | 63.32 | 62.44 | -0.87 |
| AI_f1 | 43.92 | 43.10 | -0.82 |
| AC_f1 | 41.61 | 40.80 | -0.81 |
| EC_C_f1 | 46.83 | 46.51 | -0.32 |

Token scores are exact text/token sequence matches, not semantic similarity. Official repeated-trigger-text dictionary collision behavior is retained. Token versus offset score differences cannot be interpreted as pure alignment loss.

Sources and full precision values: results.json. Independent official CLI checks: verification.json and *.official-cli.log.
