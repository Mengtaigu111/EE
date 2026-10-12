# 全量原始 offset 与 Action 保留性审计

结果：passed。核对 16048 条句子、24381 个事件、56411 个论元；失败 0 条。

所有 mention 按原始 offset **给定顺序**索引 raw document 后，均与 gold tokens/text 相等；不排序片段、不合并相同文本事件或论元。Action 的事件/论元 multiset 与原始 gold 完全相等。

## 计数

```json
{
  "rows": 16048,
  "events": 24381,
  "same_role_text_distinct_offset_groups": 14,
  "triggers": 24381,
  "trigger_noncontiguous": 0,
  "trigger_nonmonotonic": 0,
  "trigger_multiple_contiguous_text_occurrences": 2702,
  "trigger_gold_ordered_diagnostic_false": 0,
  "arguments": 56411,
  "argument_noncontiguous": 2492,
  "argument_nonmonotonic": 819,
  "argument_multiple_contiguous_text_occurrences": 12421,
  "argument_gold_ordered_diagnostic_false": 662,
  "same_type_trigger_text_distinct_offset_groups": 57
}
```

## 定位边界

当前评分基于文本存在匹配，不需要把预测文本逆推到唯一 offset。训练数据仍保留原始 gold offsets；Action/Tree 只输出文本，因此重复文本、多种等效片段选择无法唯一恢复坐标。该信息丢失属于格式表达能力的限制，不能通过选择第一个匹配位置修复。

`ordered_match` 只判断顺序子序列是否存在，返回布尔值，既不返回 offset，也不保证连续性或唯一性。非单调 gold 可能被诊断为无序列匹配，这不等于 gold/action 错误；该诊断不会删预测，也不影响 presence 分数或奖励。

若后续要求精确 span/offset 指标，应单独增加片段坐标输出协议与对应监督；不能声称现在纯文本 Action 能唯一复原全部原始 offset。

## 真实样本

### repeated_text

```json
{
  "split": "train",
  "id": "febd573ada9568c635f6d8aeada27ec5::sentence-0",
  "document_id": "febd573ada9568c635f6d8aeada27ec5",
  "mention_kind": "argument",
  "event_type": "ITT",
  "role": "Target",
  "offsets": [
    0,
    1
  ],
  "text": "program induction",
  "tokens_at_offsets": [
    "program",
    "induction"
  ],
  "contiguous_candidate_offsets": [
    [
      0,
      1
    ],
    [
      104,
      105
    ]
  ],
  "ordered_diagnostic": true,
  "target_sentence": "program induction for answering complex questions over knowledge bases ( kbs ) aims to decompose a question into a multi - step program , whose execution against the kb produces the final answer ."
}
```

```text
NEW(ITT)
TRG("decompose")
ARG(Target, "program induction")
END
STOP
```

### nonmonotonic

```json
{
  "split": "train",
  "id": "febd573ada9568c635f6d8aeada27ec5::sentence-3",
  "document_id": "febd573ada9568c635f6d8aeada27ec5",
  "mention_kind": "argument",
  "event_type": "MDS",
  "role": "TriedComponent",
  "offsets": [
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    7,
    8
  ],
  "text": "valuable program annotations on the rich - resourced knowledge bases",
  "tokens_at_offsets": [
    "valuable",
    "program",
    "annotations",
    "on",
    "the",
    "rich",
    "-",
    "resourced",
    "knowledge",
    "bases"
  ],
  "contiguous_candidate_offsets": [],
  "ordered_diagnostic": false,
  "target_sentence": "in this paper , we propose the approach of program transfer , which aims to leverage the valuable program annotations on the rich - resourced kbs as external supervision signals to aid program induction for the low - resourced kbs that lack program annotations ."
}
```

```text
NEW(PRP)
TRG("propose")
ARG(Proposer, "we")
ARG(Content, "approach of program transfer")
END
NEW(MDS)
TRG("leverage")
ARG(TriedComponent, "valuable program annotations on the rich - resourced knowledge bases")
ARG(Target, "aid")
END
NEW(PUR)
TRG("aid")
ARG(Aim, "program induction")
END
STOP
```

### discontinuous

```json
{
  "split": "train",
  "id": "febd573ada9568c635f6d8aeada27ec5::sentence-7",
  "document_id": "febd573ada9568c635f6d8aeada27ec5",
  "mention_kind": "argument",
  "event_type": "WKS",
  "role": "Content",
  "offsets": [
    7,
    8,
    191
  ],
  "text": "knowledge bases ontology",
  "tokens_at_offsets": [
    "knowledge",
    "bases",
    "ontology"
  ],
  "contiguous_candidate_offsets": [],
  "ordered_diagnostic": true,
  "target_sentence": "during the searching , we incorporate the kb ontology to prune the search space ."
}
```

```text
NEW(WKS)
TRG("incorporate")
ARG(Researcher, "we")
ARG(Content, "knowledge bases ontology")
ARG(Condition, "during the searching")
ARG(Target, "prune")
END
NEW(PUR)
TRG("prune")
ARG(Aim, "search space")
END
STOP
```

### duplicate_event_text

```json
{
  "split": "train",
  "id": "f0836257dd4df8ac15ef193f4376fda5::sentence-7",
  "gold_events": [
    {
      "event_type": "FIN",
      "arguments": [
        {
          "text": "we",
          "nugget_type": "OG",
          "argument_type": "Finder",
          "tokens": [
            "we"
          ],
          "offsets": [
            230
          ]
        },
        {
          "text": "yields",
          "nugget_type": "E-CMP",
          "argument_type": "Content",
          "tokens": [
            "yields"
          ],
          "offsets": [
            239
          ]
        }
      ],
      "trigger": {
        "text": "show",
        "tokens": [
          "show"
        ],
        "offsets": [
          232
        ]
      }
    },
    {
      "event_type": "CMP",
      "arguments": [
        {
          "text": "putting all information sources together",
          "nugget_type": "APP",
          "argument_type": "Arg1",
          "tokens": [
            "putting",
            "all",
            "information",
            "sources",
            "together"
          ],
          "offsets": [
            234,
            235,
            236,
            237,
            238
          ]
        },
        {
          "text": "current state - of - the - art",
          "nugget_type": "APP",
          "argument_type": "Arg2",
          "tokens": [
            "current",
            "state",
            "-",
            "of",
            "-",
            "the",
            "-",
            "art"
          ],
          "offsets": [
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251
          ]
        },
        {
          "text": "huge improvements",
          "nugget_type": "STR",
          "argument_type": "Result",
          "tokens": [
            "huge",
            "improvements"
          ],
          "offsets": [
            240,
            241
          ]
        }
      ],
      "trigger": {
        "text": "yields",
        "tokens": [
          "yields"
        ],
        "offsets": [
          239
        ]
      }
    },
    {
      "event_type": "FIN",
      "arguments": [
        {
          "text": "matters",
          "nugget_type": "E-FAC",
          "argument_type": "Content",
          "tokens": [
            "matters"
          ],
          "offsets": [
            226
          ]
        }
      ],
      "trigger": {
        "text": "show",
        "tokens": [
          "show"
        ],
        "offsets": [
          221
        ]
      }
    },
    {
      "event_type": "FAC",
      "arguments": [
        {
          "text": "what was written",
          "nugget_type": "TAK",
          "argument_type": "Subject",
          "tokens": [
            "what",
            "was",
            "written"
          ],
          "offsets": [
            223,
            224,
            225
          ]
        },
        {
          "text": "most",
          "nugget_type": "STR",
          "argument_type": "Object",
          "tokens": [
            "most"
          ],
          "offsets": [
            227
          ]
        }
      ],
      "trigger": {
        "text": "matters",
        "tokens": [
          "matters"
        ],
        "offsets": [
          226
        ]
      }
    }
  ]
}
```

```text
NEW(FIN)
TRG("show")
ARG(Content, "matters")
END
NEW(FAC)
TRG("matters")
ARG(Subject, "what was written")
ARG(Object, "most")
END
NEW(FIN)
TRG("show")
ARG(Finder, "we")
ARG(Content, "yields")
END
NEW(CMP)
TRG("yields")
ARG(Arg1, "putting all information sources together")
ARG(Result, "huge improvements")
ARG(Arg2, "current state - of - the - art")
END
STOP
```
