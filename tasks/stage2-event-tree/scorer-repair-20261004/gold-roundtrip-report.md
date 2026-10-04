# Gold sentence-tree roundtrip audit

Full test: 250 documents / 1,627 sentence completions. Fixed evaluation-v2 parser and aligner; merge deduplicate=False. Gold is diagnostic only.

Correspondence never uses a global scorer: exact original target-event multiset coverage, stable canonical event ordering, canonical original argument ordering, exact parsed completion equality, and pre-sort aligner ordering are asserted. Merge output multiset is checked against that independently paired aligner output.

## All official metrics (%)

| input | metric | F1 | precision | recall |
|---|---|---:|---:|---:|
| original_gold_self_offset | TI | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_offset | TC | 99.754902 | 99.754902 | 99.754902 |
| original_gold_self_offset | AI | 98.635648 | 98.635648 | 98.635648 |
| original_gold_self_offset | AC | 98.635648 | 98.635648 | 98.635648 |
| original_gold_self_offset | EC_I | 98.997494 | 98.997494 | 98.997494 |
| original_gold_self_offset | EC_C | 98.997494 | 98.997494 | 98.997494 |
| original_gold_self_token | TI | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_token | TC | 98.774510 | 98.774510 | 98.774510 |
| original_gold_self_token | AI | 96.711562 | 96.711562 | 96.711562 |
| original_gold_self_token | AC | 96.694070 | 96.694070 | 96.694070 |
| original_gold_self_token | EC_I | 95.522388 | 95.522388 | 95.522388 |
| original_gold_self_token | EC_C | 95.522388 | 95.522388 | 95.522388 |
| gold_completion_raw_token | TI | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_raw_token | TC | 98.774510 | 98.774510 | 98.774510 |
| gold_completion_raw_token | AI | 95.076519 | 95.068206 | 95.084835 |
| gold_completion_raw_token | AC | 95.059029 | 95.050717 | 95.067343 |
| gold_completion_raw_token | EC_I | 95.587321 | 95.527950 | 95.646766 |
| gold_completion_raw_token | EC_C | 95.587321 | 95.527950 | 95.646766 |
| gold_completion_roundtrip_offset | TI | 99.877451 | 99.877451 | 99.877451 |
| gold_completion_roundtrip_offset | TC | 99.632353 | 99.632353 | 99.632353 |
| gold_completion_roundtrip_offset | AI | 95.539619 | 95.539619 | 95.539619 |
| gold_completion_roundtrip_offset | AC | 95.522127 | 95.522127 | 95.522127 |
| gold_completion_roundtrip_offset | EC_I | 98.127341 | 97.761194 | 98.496241 |
| gold_completion_roundtrip_offset | EC_C | 98.127341 | 97.761194 | 98.496241 |
| gold_completion_roundtrip_token | TI | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_roundtrip_token | TC | 98.774510 | 98.774510 | 98.774510 |
| gold_completion_roundtrip_token | AI | 95.076519 | 95.068206 | 95.084835 |
| gold_completion_roundtrip_token | AC | 95.059029 | 95.050717 | 95.067343 |
| gold_completion_roundtrip_token | EC_I | 95.587321 | 95.527950 | 95.646766 |
| gold_completion_roundtrip_token | EC_C | 95.587321 | 95.527950 | 95.646766 |

## Instance-preserving metrics (%)

| input | metric | F1 | precision | recall |
|---|---|---:|---:|---:|
| original_gold_self_offset | TI | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_offset | TC | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_offset | AI | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_offset | AC | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_offset | EC_I | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_offset | EC_C | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_token | TI | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_token | TC | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_token | AI | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_token | AC | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_token | EC_I | 100.000000 | 100.000000 | 100.000000 |
| original_gold_self_token | EC_C | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_raw_token | TI | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_raw_token | TC | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_raw_token | AI | 98.338289 | 98.338289 | 98.338289 |
| gold_completion_raw_token | AC | 98.338289 | 98.338289 | 98.338289 |
| gold_completion_raw_token | EC_I | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_raw_token | EC_C | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_roundtrip_offset | TI | 99.877451 | 99.877451 | 99.877451 |
| gold_completion_roundtrip_offset | TC | 99.877451 | 99.877451 | 99.877451 |
| gold_completion_roundtrip_offset | AI | 96.886479 | 96.886479 | 96.886479 |
| gold_completion_roundtrip_offset | AC | 96.886479 | 96.886479 | 96.886479 |
| gold_completion_roundtrip_offset | EC_I | 99.126092 | 98.756219 | 99.498747 |
| gold_completion_roundtrip_offset | EC_C | 99.126092 | 98.756219 | 99.498747 |
| gold_completion_roundtrip_token | TI | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_roundtrip_token | TC | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_roundtrip_token | AI | 98.338289 | 98.338289 | 98.338289 |
| gold_completion_roundtrip_token | AC | 98.338289 | 98.338289 | 98.338289 |
| gold_completion_roundtrip_token | EC_I | 100.000000 | 100.000000 | 100.000000 |
| gold_completion_roundtrip_token | EC_C | 100.000000 | 100.000000 | 100.000000 |

## Proven losses and boundaries

Ordered-offset mismatches: 3 triggers and 173 arguments. Of the argument mismatches, 69 preserve the complete sorted gold span and differ only because canonicalization removed order or multiplicity. Actual sorted-span mismatches: 104 arguments. Canonical sequence loss affects 95 arguments in total; categories overlap.

Original gold self-scoring reproduces the existing dictionary-overwrite defect. Raw gold completions also lose original reverse/nonmonotonic token order: canonical_events sorts and uniquifies offsets before producing text. This is an information-loss bug for an offset-sequence-preserving evaluation contract, not an alignment error that text can undo.

The aligner additionally chooses among repeated same-text spans, prefers contiguous occurrences before any discontinuous candidates, greedily enumerates subsequences, and prefers argument text matching a trigger. These policies can select a different position even when text is correct. Repeated text, same-position event multiplicity, discontinuity and original reverse order are frequently unidentifiable from this completion schema. A wrong gold position alone is not proof that a gold-free deterministic alternative could recover it.

The subsequence fixture proves incomplete candidate enumeration, but its omitted span is also text-ambiguous. Enumerating all subsequences removes that algorithmic limitation without proving a uniquely correct location.

The new independent gold-free resolver enumerates contiguous and noncontiguous candidates together. With the default 256-candidate cap, it covers every gold span in all 40 legacy omitted-candidate cases (36 greedy-subsequence omissions and 4 contiguous-preference suppressions). All 40 resolve to ambiguous, none truncate; maximum observed candidate count is 141. It returns offsets=None in every case. This proves improved candidate completeness, not recovered positions or higher real-model scores. No actual prediction or historical aligner is changed.

Do not add gold-self loss to real model scores. Official scorer collisions can undercount and overcount; token text can match the wrong location. These measurements establish loss mechanisms, not the net bias in actual predictions.

## Counts (categories overlap)

```json
{
  "offset_duplicate_trigger_expression_groups": 30,
  "offset_duplicate_trigger_expression_instances": 68,
  "offset_docs_with_duplicate_trigger_expression": 26,
  "offset_duplicate_argument_expression_groups": 1,
  "token_duplicate_trigger_expression_groups": 79,
  "token_duplicate_trigger_expression_instances": 176,
  "token_docs_with_duplicate_trigger_expression": 65,
  "same_trigger_tokens_distinct_offset_groups": 52,
  "token_duplicate_argument_expression_groups": 2,
  "canonical_completion_exact_sentence_rows": 1627,
  "events": 2448,
  "event_fields_exact_ignoring_argument_order_and_nugget_metadata": 2283,
  "trigger_total": 2448,
  "trigger_original_tokens_match_document_ordered_offsets": 2448,
  "trigger_original_text_matches_token_field": 2448,
  "trigger_offset_sequence_exact": 2445,
  "trigger_tokens_exact": 2448,
  "trigger_text_field_exact": 2448,
  "trigger_canonical_sorted_offsets_exact": 2445,
  "trigger_canonical_text_exact": 2448,
  "argument_total": 5717,
  "argument_original_tokens_match_document_ordered_offsets": 5717,
  "argument_original_text_matches_token_field": 5717,
  "argument_offset_sequence_exact": 5544,
  "argument_tokens_exact": 5622,
  "argument_text_field_exact": 5622,
  "argument_canonical_sorted_offsets_exact": 5613,
  "argument_canonical_text_exact": 5717,
  "argument_wrong_argument_position": 173,
  "argument_same_tokens_different_offsets": 78,
  "argument_same_text_field_different_offsets": 78,
  "argument_canonical_sorted_span_position_mismatch": 104,
  "argument_gold_sorted_span_in_candidates_but_not_selected": 64,
  "argument_multiple_same_text_candidates": 83,
  "trigger_wrong_trigger_position": 3,
  "trigger_same_tokens_different_offsets": 3,
  "trigger_same_text_field_different_offsets": 3,
  "trigger_canonical_sorted_span_position_mismatch": 3,
  "trigger_gold_sorted_span_in_candidates_but_not_selected": 3,
  "trigger_multiple_same_text_candidates": 3,
  "argument_gold_sorted_span_absent_from_candidates": 40,
  "argument_valid_gold_subsequence_omitted_by_greedy_enumeration": 36,
  "argument_discontinuous_gold": 117,
  "argument_position_mismatch_only_original_order_or_multiplicity_loss": 69,
  "argument_canonical_sorted_unique_loses_original_offset_sequence": 95,
  "argument_reverse_or_nonmonotonic_original_order": 95,
  "argument_canonical_text_differs_from_original_tokens": 95,
  "argument_contiguous_candidate_suppresses_discontinuous_search": 4,
  "argument_canonical_unique_loses_multiplicity": 4,
  "documents_with_position_mismatch": 93,
  "documents_with_trigger_position_mismatch": 3,
  "documents_with_argument_position_mismatch": 93,
  "default_dedup_events_removed": 1,
  "new_resolver_legacy_omission_cases": 40,
  "new_resolver_legacy_omission_gold_candidate_covered": 40,
  "new_resolver_legacy_omission_status_ambiguous": 40,
  "new_resolver_max_candidates_observed": 141
}
```

## Gold-free repair options

- Preserve original offset order when constructing supervised text targets; existing completions cannot recover the removed order.
- For future model outputs, serialize explicit ordered token indices or span IDs and event-instance IDs. Validate them against input tokens without consulting gold.
- Retain all alignment candidates and mark ambiguity; evaluate text quality alongside offset quality. If choosing candidates, freeze a policy using dev only and report tradeoffs.
- Use instance-preserving one-to-one scoring for collisions and retain old official scores for traceability. Avoid deduplication when scoring event-instance multiplicity.

## Actual examples

```json
{
  "doc_id": "f500e30b1dd8d82a97fcce9fd26bcd74",
  "title": "Cross-replication Reliability - An Empirical Approach to Interpreting Inter-rater Reliability",
  "sentence_id": "f500e30b1dd8d82a97fcce9fd26bcd74::sentence-6",
  "event_index": 7,
  "event_type": "WKS",
  "kind": "argument",
  "argument_index": 0,
  "original": {
    "text": "we",
    "nugget_type": "OG",
    "argument_type": "Researcher",
    "tokens": [
      "we"
    ],
    "offsets": [
      148
    ]
  },
  "canonical_text": "we",
  "canonical_offsets": [
    148
  ],
  "aligned": {
    "text": "we",
    "tokens": [
      "we"
    ],
    "offsets": [
      169
    ],
    "argument_type": "Researcher"
  },
  "candidate_method": "exact",
  "candidates": [
    [
      88
    ],
    [
      140
    ],
    [
      148
    ],
    [
      169
    ]
  ],
  "categories": [
    "wrong_argument_position",
    "same_tokens_different_offsets",
    "same_text_field_different_offsets",
    "canonical_sorted_span_position_mismatch",
    "gold_sorted_span_in_candidates_but_not_selected",
    "multiple_same_text_candidates"
  ],
  "trigger_original": {
    "text": "analyze",
    "tokens": [
      "analyze"
    ],
    "offsets": [
      162
    ]
  },
  "trigger_aligned": {
    "text": "analyze",
    "tokens": [
      "analyze"
    ],
    "offsets": [
      162
    ]
  }
}
```

```json
{
  "doc_id": "fd4526d8ccd6b35ae6a285c2e34721f3",
  "title": "Making Pre-trained Language Models Better Few-shot Learners",
  "sentence_id": "fd4526d8ccd6b35ae6a285c2e34721f3::sentence-2",
  "event_index": 4,
  "event_type": "PUR",
  "kind": "trigger",
  "argument_index": null,
  "original": {
    "text": "fine - tuning",
    "tokens": [
      "fine",
      "-",
      "tuning"
    ],
    "offsets": [
      94,
      95,
      96
    ]
  },
  "canonical_text": "fine - tuning",
  "canonical_offsets": [
    94,
    95,
    96
  ],
  "aligned": {
    "text": "fine - tuning",
    "tokens": [
      "fine",
      "-",
      "tuning"
    ],
    "offsets": [
      79,
      80,
      81
    ]
  },
  "candidate_method": "exact_scope",
  "candidates": [
    [
      79,
      80,
      81
    ],
    [
      94,
      95,
      96
    ]
  ],
  "categories": [
    "wrong_trigger_position",
    "same_tokens_different_offsets",
    "same_text_field_different_offsets",
    "canonical_sorted_span_position_mismatch",
    "gold_sorted_span_in_candidates_but_not_selected",
    "multiple_same_text_candidates"
  ],
  "trigger_original": {
    "text": "fine - tuning",
    "tokens": [
      "fine",
      "-",
      "tuning"
    ],
    "offsets": [
      94,
      95,
      96
    ]
  },
  "trigger_aligned": {
    "text": "fine - tuning",
    "tokens": [
      "fine",
      "-",
      "tuning"
    ],
    "offsets": [
      79,
      80,
      81
    ]
  }
}
```

```json
{
  "doc_id": "cf863b7553fb478ab0719d422f732e7b",
  "title": "Neural Pipeline for Zero-Shot Data-to-Text Generation",
  "sentence_id": "cf863b7553fb478ab0719d422f732e7b::sentence-1",
  "event_index": 3,
  "event_type": "WKS",
  "kind": "argument",
  "argument_index": 1,
  "original": {
    "text": "data - to - text generation datasets",
    "nugget_type": "DST",
    "argument_type": "Dataset",
    "tokens": [
      "data",
      "-",
      "to",
      "-",
      "text",
      "generation",
      "datasets"
    ],
    "offsets": [
      1,
      2,
      3,
      4,
      5,
      44,
      45
    ]
  },
  "canonical_text": "data - to - text generation datasets",
  "canonical_offsets": [
    1,
    2,
    3,
    4,
    5,
    44,
    45
  ],
  "aligned": {
    "text": "data - to - text generation datasets",
    "tokens": [
      "data",
      "-",
      "to",
      "-",
      "text",
      "generation",
      "datasets"
    ],
    "offsets": [
      1,
      2,
      3,
      4,
      5,
      9,
      45
    ],
    "argument_type": "Dataset"
  },
  "candidate_method": "subsequence",
  "candidates": [
    [
      1,
      2,
      3,
      4,
      5,
      9,
      45
    ]
  ],
  "categories": [
    "wrong_argument_position",
    "same_tokens_different_offsets",
    "same_text_field_different_offsets",
    "canonical_sorted_span_position_mismatch",
    "gold_sorted_span_absent_from_candidates",
    "valid_gold_subsequence_omitted_by_greedy_enumeration",
    "discontinuous_gold"
  ],
  "trigger_original": {
    "text": "avoid finetuning",
    "tokens": [
      "avoid",
      "finetuning"
    ],
    "offsets": [
      34,
      35
    ]
  },
  "trigger_aligned": {
    "text": "avoid finetuning",
    "tokens": [
      "avoid",
      "finetuning"
    ],
    "offsets": [
      34,
      35
    ]
  }
}
```

```json
{
  "doc_id": "cc90cf8a89b308caf0758defbb3df0d2",
  "title": "Improving the Robustness of Question Answering Systems to Question Paraphrasing",
  "sentence_id": "cc90cf8a89b308caf0758defbb3df0d2::sentence-1",
  "event_index": 2,
  "event_type": "PUR",
  "kind": "argument",
  "argument_index": 0,
  "original": {
    "text": "robustness of question answering models",
    "nugget_type": "TAK",
    "argument_type": "Aim",
    "tokens": [
      "robustness",
      "of",
      "question",
      "answering",
      "models"
    ],
    "offsets": [
      31,
      32,
      4,
      5,
      34
    ]
  },
  "canonical_text": "question answering robustness of models",
  "canonical_offsets": [
    4,
    5,
    31,
    32,
    34
  ],
  "aligned": {
    "text": "question answering robustness of models",
    "tokens": [
      "question",
      "answering",
      "robustness",
      "of",
      "models"
    ],
    "offsets": [
      4,
      5,
      31,
      32,
      34
    ],
    "argument_type": "Aim"
  },
  "candidate_method": "subsequence",
  "candidates": [
    [
      4,
      5,
      31,
      32,
      34
    ]
  ],
  "categories": [
    "position_mismatch_only_original_order_or_multiplicity_loss",
    "canonical_sorted_unique_loses_original_offset_sequence",
    "reverse_or_nonmonotonic_original_order",
    "canonical_text_differs_from_original_tokens",
    "wrong_argument_position",
    "discontinuous_gold"
  ],
  "trigger_original": {
    "text": "explore",
    "tokens": [
      "explore"
    ],
    "offsets": [
      29
    ]
  },
  "trigger_aligned": {
    "text": "explore",
    "tokens": [
      "explore"
    ],
    "offsets": [
      29
    ]
  }
}
```

```json
{
  "doc_id": "68d883fd2cfe9e80ea9193d5b1e0b588",
  "title": "DEAM: Dialogue Coherence Evaluation using AMR-based Semantic Manipulations",
  "sentence_id": "68d883fd2cfe9e80ea9193d5b1e0b588::sentence-7",
  "event_index": 11,
  "event_type": "FAC",
  "kind": "argument",
  "argument_index": 2,
  "original": {
    "text": "coherent dialogues",
    "nugget_type": "FEA",
    "argument_type": "Object",
    "tokens": [
      "coherent",
      "dialogues"
    ],
    "offsets": [
      215,
      218
    ]
  },
  "canonical_text": "coherent dialogues",
  "canonical_offsets": [
    215,
    218
  ],
  "aligned": {
    "text": "coherent dialogues",
    "tokens": [
      "coherent",
      "dialogues"
    ],
    "offsets": [
      82,
      83
    ],
    "argument_type": "Object"
  },
  "candidate_method": "exact",
  "candidates": [
    [
      82,
      83
    ]
  ],
  "categories": [
    "wrong_argument_position",
    "same_tokens_different_offsets",
    "same_text_field_different_offsets",
    "canonical_sorted_span_position_mismatch",
    "gold_sorted_span_absent_from_candidates",
    "discontinuous_gold",
    "contiguous_candidate_suppresses_discontinuous_search"
  ],
  "trigger_original": {
    "text": "distinguish",
    "tokens": [
      "distinguish"
    ],
    "offsets": [
      213
    ]
  },
  "trigger_aligned": {
    "text": "distinguish",
    "tokens": [
      "distinguish"
    ],
    "offsets": [
      213
    ]
  }
}
```

```json
{
  "doc_id": "435cd3d22bfa0f9ad55eb40d8491d2c9",
  "title": "Speak to your Parser: Interactive Text-to-SQL with Natural Language Feedback",
  "sentence_id": "435cd3d22bfa0f9ad55eb40d8491d2c9::sentence-2",
  "event_index": 3,
  "event_type": "MDS",
  "kind": "argument",
  "argument_index": 0,
  "original": {
    "text": "correct the system when system generates an inaccurate interpretation of an initial utterance",
    "nugget_type": "LIM",
    "argument_type": "Condition",
    "tokens": [
      "correct",
      "the",
      "system",
      "when",
      "system",
      "generates",
      "an",
      "inaccurate",
      "interpretation",
      "of",
      "an",
      "initial",
      "utterance"
    ],
    "offsets": [
      69,
      70,
      71,
      72,
      71,
      74,
      75,
      76,
      77,
      78,
      79,
      80,
      81
    ]
  },
  "canonical_text": "correct the system when generates an inaccurate interpretation of an initial utterance",
  "canonical_offsets": [
    69,
    70,
    71,
    72,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81
  ],
  "aligned": {
    "text": "correct the system when generates an inaccurate interpretation of an initial utterance",
    "tokens": [
      "correct",
      "the",
      "system",
      "when",
      "generates",
      "an",
      "inaccurate",
      "interpretation",
      "of",
      "an",
      "initial",
      "utterance"
    ],
    "offsets": [
      69,
      70,
      71,
      72,
      74,
      75,
      76,
      77,
      78,
      79,
      80,
      81
    ],
    "argument_type": "Condition"
  },
  "candidate_method": "subsequence",
  "candidates": [
    [
      69,
      70,
      71,
      72,
      74,
      75,
      76,
      77,
      78,
      79,
      80,
      81
    ]
  ],
  "categories": [
    "position_mismatch_only_original_order_or_multiplicity_loss",
    "canonical_sorted_unique_loses_original_offset_sequence",
    "canonical_unique_loses_multiplicity",
    "reverse_or_nonmonotonic_original_order",
    "canonical_text_differs_from_original_tokens",
    "wrong_argument_position",
    "discontinuous_gold"
  ],
  "trigger_original": {
    "text": "providing",
    "tokens": [
      "providing"
    ],
    "offsets": [
      61
    ]
  },
  "trigger_aligned": {
    "text": "providing",
    "tokens": [
      "providing"
    ],
    "offsets": [
      61
    ]
  }
}
```

## Reproduction

`/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/stage2-event-tree/scorer-repair-20261004/audit_gold_roundtrip.py`

CPU only. JSON contains all position mismatch records, all representation differences, full contexts, exact source SHA-256, fixed module origins and all 18 official metrics for each condition. The script asserts sources unchanged after scoring. No aligner, scorer, old data or real predictions are modified.
