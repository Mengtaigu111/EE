# 2026-10-03 dev评估修复结果

独立版本：`tasks/stage2-event-tree/evaluation-v2-20261003/`。未修改旧run锁定的13份共享源码、checkpoint、数据、prompt、旧预测或旧指标。
v2从sentence raw_output重新解析，按目标句scope对齐，保留文本occurrence，按事件类型、trigger offsets与argument role/offsets去重。官方评分器`utils.calculate_F1`原样使用。

## Dev同口径重评

| 组别 | step | 旧AC F1(%) | 新AC F1(%) | 差值(pp) |
|---|---:|---:|---:|---:|
| UserV2 Easy2课程 | 802 | 33.8498 | 34.3523 | +0.5026 |
| UserV2 Easy2课程 | 814 | 35.2335 | 35.7027 | +0.4692 |
| UserV2 Easy2课程 | 1604 | 39.5032 | 39.9000 | +0.3968 |
| UserV2 Easy2课程 | 2406 | 40.2860 | 40.7179 | +0.4320 |
| UserV2 Easy2课程 | 3208 | 38.4877 | 38.9771 | +0.4894 |
| UserV2 Easy2课程 | 4010 | 38.3600 | 38.8747 | +0.5147 |
| AppendixNoCopy历史baseline | 802 | 37.7794 | 38.4788 | +0.6993 |
| AppendixNoCopy历史baseline | 1604 | 39.5112 | 39.8813 | +0.3701 |
| AppendixNoCopy历史baseline | 2406 | 40.9561 | 41.2637 | +0.3076 |
| AppendixNoCopy历史baseline | 3208 | 38.7973 | 39.1693 | +0.3720 |
| AppendixNoCopy历史baseline | 4010 | 38.1500 | 38.5608 | +0.4108 |

两组均只按新official dev AC F1选模，平局选较大step；未读取test成绩选checkpoint。

- UserV2 Easy2课程：旧best step 2406，新best step 2406，新dev AC F1 40.7179%。
- AppendixNoCopy历史baseline：旧best step 2406，新best step 2406，新dev AC F1 41.2637%。

## Test（仅各组新dev-best）

- UserV2 Easy2课程 step 2406：TI 74.1627%，TC 62.6794%，AI 43.1103%，AC 40.7333%（旧40.2050%，变化+0.5283pp），EC 48.4412%；复杂场景exit=0、七项齐全。
  - 复杂场景：Consecutive_AC_f1=11.9500%；Overlap_AC_f1=18.4700%；Overlap_TC_f1=51.9900%；Reverse_AC_f1=0.0000%；Subevent_AC_f1=48.4400%；Subevent_EC_f1=46.8600%；Subevent_TC_f1=56.1800%。
- AppendixNoCopy历史baseline step 2406：TI 75.4139%，TC 63.5479%，AI 44.2034%，AC 41.8797%（旧41.3447%，变化+0.5350pp），EC 48.0687%；复杂场景exit=0、七项齐全。
  - 复杂场景：Consecutive_AC_f1=12.2900%；Overlap_AC_f1=18.4700%；Overlap_TC_f1=53.8600%；Reverse_AC_f1=0.0000%；Subevent_AC_f1=48.0700%；Subevent_EC_f1=46.2500%；Subevent_TC_f1=56.7600%。

## 验收与边界

- 11份dev raw均具complete标记，输出SHA-256与config数据/adapter hash一致，sentence ID唯一且与数据逐行全覆盖；每份provenance保存输入、合并、指标及代码hash。
- 回归测试覆盖跨句同文本、同句共享trigger、同类型多occurrence、畸形节点恢复、raw重新解析、重复身份与复杂场景失败闸门。
- 旧流水线 `goal-22/main.exit`：0。
- 官方`utils.calculate_F1`对共享trigger offsets的覆盖是既有评分限制，本轮保留官方口径。两组prompt不同，因此比较是历史对照，不构成仅改变课程的因果消融。
- Stage3报告问题4–6不在本次范围。

独立验收输出：

```json
{
  "verified": {
    "curriculum": {
      "dev_count": 6,
      "best_step": 2406,
      "test_AC_f1": 0.4073333912590147
    },
    "baseline": {
      "dev_count": 5,
      "best_step": 2406,
      "test_AC_f1": 0.41879652049143573
    }
  },
  "locked_source_count": 13
}
```
