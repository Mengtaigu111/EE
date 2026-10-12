# Action v2 启动前验证

正式数据为`../data/action-v2-goldorder-20261012/`，采用[兼容短prompt](../prompts/action-v2-checked.txt)。旧v1数据保留。仅prompt改变，Gold/Action/Tree/Sorted1逐条不变，见[data-diff.json](data-diff.json)。

- [Gold→Action→独立Tree脚本](gold_action_tree_audit.py)：[报告](gold-action-tree-report.json)，16048条、2508文档、24381事件、56411论元，0失败。
- [offset与重复文本脚本](audit_offsets.py)：[报告](offset-audit.json)与[案例](offset-audit.md)，原始有序坐标/文本及实例次数保持；纯文本不保证唯一逆推offset，诊断不删除预测或参与presence匹配。
- [真实tokenizer/mask/768截断脚本](token_cap_audit.py)：[报告](token-cap-report.json)，全部完整目标与截取768后的解码Action一致，最长train/dev/test为295/671/520，无Gold截断；模型自由生成仍可能达到上限。
- [CPU回归](cpu-tests.log)：70 passed，含完整模型来源锁、实际生成参数、assistant监督/尾组、真实tiny LoRA SFT/GRPO更新、reference冻结和选中adapter配置锁。
- [完整preflight](preflight.json)：基座全部分片、tokenizer、数据、代码与CoT接续hash，768输出/4096上下文预算验证；采样审计sampling.py也纳入来源锁。
- [数据manifest副本](data-manifest.json)：记录数据与source SHA，JSONL与模型权重不纳入git。

复现数据生成（输出目录必须不存在，环境及原始修复句子数据/Sorted1/CoT模型作为外部输入）：

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python scripts/build_action_dataset.py \
  --prompt tasks/action-experiment/prompts/action-v2-checked.txt \
  --output tasks/action-experiment/data/action-v2-goldorder-20261012 \
  --tokenizer /opt/models/Qwen3-8B
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m pytest -q tasks/action-experiment/tests tests/test_score_tree_presence.py
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/action-experiment/audit-v2-20261012/gold_action_tree_audit.py
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/action-experiment/audit-v2-20261012/audit_offsets.py
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/action-experiment/audit-v2-20261012/token_cap_audit.py
```

以上证据说明数据和代码检查通过，不表示正式训练已经完成。启动与阶段日志见[正式队列](../launch-v2-20261012/README.md)。
