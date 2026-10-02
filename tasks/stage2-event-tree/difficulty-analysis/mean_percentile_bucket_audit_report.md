# Mean percentile B1/B2 audit

Type multiset comparison per sentence: preserves duplicate types, ignores triggers and arguments. Optimistic inventory diagnosis, not official TC or instance-aligned type accuracy.

No train prediction JSONL found under stage2-event-tree/runs. Dev diagnosis uses train percentiles and fixed train threshold; no dev refitting. Group composition is observational and confounded by event count/type.

```json
{
  "train_sentence_counts": {
    "B1": 5427,
    "ROOT": 2124,
    "B2": 5277
  },
  "train_type_distribution": [
    {
      "event_type": "CMP",
      "total": 1536,
      "B1": 491,
      "B2": 1045,
      "B1_percent": 31.966145833333332
    },
    {
      "event_type": "FAC",
      "total": 2528,
      "B1": 722,
      "B2": 1806,
      "B1_percent": 28.560126582278482
    },
    {
      "event_type": "FIN",
      "total": 1480,
      "B1": 170,
      "B2": 1310,
      "B1_percent": 11.486486486486486
    },
    {
      "event_type": "ITT",
      "total": 1400,
      "B1": 1298,
      "B2": 102,
      "B1_percent": 92.71428571428571
    },
    {
      "event_type": "MDS",
      "total": 2385,
      "B1": 477,
      "B2": 1908,
      "B1_percent": 20.0
    },
    {
      "event_type": "PRP",
      "total": 2094,
      "B1": 691,
      "B2": 1403,
      "B1_percent": 32.99904489016237
    },
    {
      "event_type": "PUR",
      "total": 3320,
      "B1": 500,
      "B2": 2820,
      "B1_percent": 15.060240963855422
    },
    {
      "event_type": "RWF",
      "total": 1549,
      "B1": 816,
      "B2": 733,
      "B1_percent": 52.679147837314396
    },
    {
      "event_type": "RWS",
      "total": 463,
      "B1": 168,
      "B2": 295,
      "B1_percent": 36.28509719222462
    },
    {
      "event_type": "WKS",
      "total": 2733,
      "B1": 1032,
      "B2": 1701,
      "B1_percent": 37.76070252469813
    }
  ],
  "type_B1_rate_range_pp": 81.22779922779922,
  "threshold": 0.4275338928340865,
  "baseline_prediction": "/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-42-20260918/checkpoint-eval/dev-checkpoint-2406.predictions.jsonl",
  "dev_input": "/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/data/sentence-c/dev.jsonl",
  "dev_bucket_metrics": {
    "B1": {
      "sentences": 664,
      "inventory_errors": 240,
      "inventory_error_rate": 0.3614457831325301,
      "count_errors": 189,
      "count_error_rate": 0.28463855421686746,
      "count_matched_sentences": 475,
      "type_errors_given_correct_count": 51,
      "type_error_rate_given_correct_count": 0.10736842105263159,
      "gold_events": 784,
      "predicted_events": 784,
      "tp": 614,
      "fp": 170,
      "fn": 170,
      "type_inventory_micro_f1": 0.7831632653061225
    },
    "B2": {
      "sentences": 658,
      "inventory_errors": 386,
      "inventory_error_rate": 0.5866261398176292,
      "count_errors": 271,
      "count_error_rate": 0.41185410334346506,
      "count_matched_sentences": 387,
      "type_errors_given_correct_count": 115,
      "type_error_rate_given_correct_count": 0.2971576227390181,
      "gold_events": 1661,
      "predicted_events": 1455,
      "tp": 1163,
      "fp": 292,
      "fn": 498,
      "type_inventory_micro_f1": 0.7464698331193839
    },
    "ROOT": {
      "sentences": 271,
      "inventory_errors": 63,
      "inventory_error_rate": 0.23247232472324722,
      "count_errors": 63,
      "count_error_rate": 0.23247232472324722,
      "count_matched_sentences": 208,
      "type_errors_given_correct_count": 0,
      "type_error_rate_given_correct_count": 0.0,
      "gold_events": 0,
      "predicted_events": 92,
      "tp": 0,
      "fp": 92,
      "fn": 0,
      "type_inventory_micro_f1": 0.0
    }
  },
  "dev_by_event_count": {
    "1": {
      "B1": {
        "sentences": 544,
        "inventory_errors": 191,
        "inventory_error_rate": 0.35110294117647056,
        "count_errors": 159,
        "count_error_rate": 0.2922794117647059,
        "count_matched_sentences": 385,
        "type_errors_given_correct_count": 32,
        "type_error_rate_given_correct_count": 0.08311688311688312,
        "gold_events": 544,
        "predicted_events": 557,
        "tp": 421,
        "fp": 136,
        "fn": 123,
        "type_inventory_micro_f1": 0.7647593097184378
      },
      "B2": {
        "sentences": 27,
        "inventory_errors": 18,
        "inventory_error_rate": 0.6666666666666666,
        "count_errors": 17,
        "count_error_rate": 0.6296296296296297,
        "count_matched_sentences": 10,
        "type_errors_given_correct_count": 1,
        "type_error_rate_given_correct_count": 0.1,
        "gold_events": 27,
        "predicted_events": 28,
        "tp": 15,
        "fp": 13,
        "fn": 12,
        "type_inventory_micro_f1": 0.5454545454545454
      }
    },
    "2": {
      "B1": {
        "sentences": 120,
        "inventory_errors": 49,
        "inventory_error_rate": 0.4083333333333333,
        "count_errors": 30,
        "count_error_rate": 0.25,
        "count_matched_sentences": 90,
        "type_errors_given_correct_count": 19,
        "type_error_rate_given_correct_count": 0.2111111111111111,
        "gold_events": 240,
        "predicted_events": 227,
        "tp": 193,
        "fp": 34,
        "fn": 47,
        "type_inventory_micro_f1": 0.8265524625267666
      },
      "B2": {
        "sentences": 381,
        "inventory_errors": 192,
        "inventory_error_rate": 0.5039370078740157,
        "count_errors": 119,
        "count_error_rate": 0.3123359580052493,
        "count_matched_sentences": 262,
        "type_errors_given_correct_count": 73,
        "type_error_rate_given_correct_count": 0.2786259541984733,
        "gold_events": 762,
        "predicted_events": 705,
        "tp": 559,
        "fp": 146,
        "fn": 203,
        "type_inventory_micro_f1": 0.7620995228357191
      }
    },
    "3": {
      "B1": {
        "sentences": 0,
        "inventory_errors": 0,
        "inventory_error_rate": null,
        "count_errors": 0,
        "count_error_rate": null,
        "count_matched_sentences": 0,
        "type_errors_given_correct_count": 0,
        "type_error_rate_given_correct_count": null,
        "gold_events": 0,
        "predicted_events": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "type_inventory_micro_f1": null
      },
      "B2": {
        "sentences": 169,
        "inventory_errors": 110,
        "inventory_error_rate": 0.650887573964497,
        "count_errors": 86,
        "count_error_rate": 0.5088757396449705,
        "count_matched_sentences": 83,
        "type_errors_given_correct_count": 24,
        "type_error_rate_given_correct_count": 0.2891566265060241,
        "gold_events": 507,
        "predicted_events": 428,
        "tp": 361,
        "fp": 67,
        "fn": 146,
        "type_inventory_micro_f1": 0.772192513368984
      }
    },
    "4": {
      "B1": {
        "sentences": 0,
        "inventory_errors": 0,
        "inventory_error_rate": null,
        "count_errors": 0,
        "count_error_rate": null,
        "count_matched_sentences": 0,
        "type_errors_given_correct_count": 0,
        "type_error_rate_given_correct_count": null,
        "gold_events": 0,
        "predicted_events": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "type_inventory_micro_f1": null
      },
      "B2": {
        "sentences": 55,
        "inventory_errors": 42,
        "inventory_error_rate": 0.7636363636363637,
        "count_errors": 32,
        "count_error_rate": 0.5818181818181818,
        "count_matched_sentences": 23,
        "type_errors_given_correct_count": 10,
        "type_error_rate_given_correct_count": 0.43478260869565216,
        "gold_events": 220,
        "predicted_events": 181,
        "tp": 144,
        "fp": 37,
        "fn": 76,
        "type_inventory_micro_f1": 0.7182044887780549
      }
    },
    "5": {
      "B1": {
        "sentences": 0,
        "inventory_errors": 0,
        "inventory_error_rate": null,
        "count_errors": 0,
        "count_error_rate": null,
        "count_matched_sentences": 0,
        "type_errors_given_correct_count": 0,
        "type_error_rate_given_correct_count": null,
        "gold_events": 0,
        "predicted_events": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "type_inventory_micro_f1": null
      },
      "B2": {
        "sentences": 15,
        "inventory_errors": 14,
        "inventory_error_rate": 0.9333333333333333,
        "count_errors": 11,
        "count_error_rate": 0.7333333333333333,
        "count_matched_sentences": 4,
        "type_errors_given_correct_count": 3,
        "type_error_rate_given_correct_count": 0.75,
        "gold_events": 75,
        "predicted_events": 58,
        "tp": 47,
        "fp": 11,
        "fn": 28,
        "type_inventory_micro_f1": 0.706766917293233
      }
    },
    "6": {
      "B1": {
        "sentences": 0,
        "inventory_errors": 0,
        "inventory_error_rate": null,
        "count_errors": 0,
        "count_error_rate": null,
        "count_matched_sentences": 0,
        "type_errors_given_correct_count": 0,
        "type_error_rate_given_correct_count": null,
        "gold_events": 0,
        "predicted_events": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "type_inventory_micro_f1": null
      },
      "B2": {
        "sentences": 9,
        "inventory_errors": 8,
        "inventory_error_rate": 0.8888888888888888,
        "count_errors": 4,
        "count_error_rate": 0.4444444444444444,
        "count_matched_sentences": 5,
        "type_errors_given_correct_count": 4,
        "type_error_rate_given_correct_count": 0.8,
        "gold_events": 54,
        "predicted_events": 45,
        "tp": 32,
        "fp": 13,
        "fn": 22,
        "type_inventory_micro_f1": 0.6464646464646465
      }
    },
    "8": {
      "B1": {
        "sentences": 0,
        "inventory_errors": 0,
        "inventory_error_rate": null,
        "count_errors": 0,
        "count_error_rate": null,
        "count_matched_sentences": 0,
        "type_errors_given_correct_count": 0,
        "type_error_rate_given_correct_count": null,
        "gold_events": 0,
        "predicted_events": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "type_inventory_micro_f1": null
      },
      "B2": {
        "sentences": 2,
        "inventory_errors": 2,
        "inventory_error_rate": 1.0,
        "count_errors": 2,
        "count_error_rate": 1.0,
        "count_matched_sentences": 0,
        "type_errors_given_correct_count": 0,
        "type_error_rate_given_correct_count": null,
        "gold_events": 16,
        "predicted_events": 10,
        "tp": 5,
        "fp": 5,
        "fn": 11,
        "type_inventory_micro_f1": 0.38461538461538464
      }
    }
  },
  "metric_definition": "Type multiset comparison per sentence: preserves duplicate types, ignores triggers and arguments. Optimistic inventory diagnosis, not official TC or instance-aligned type accuracy.",
  "limitation": "No train prediction JSONL found under stage2-event-tree/runs. Dev diagnosis uses train percentiles and fixed train threshold; no dev refitting. Group composition is observational and confounded by event count/type."
}
```
