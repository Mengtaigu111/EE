# 迁移依赖盘点（2026-10-15）

当前主线是：外部 API 生成类型 CoT → user-final 识别数据 → NoRE CoT1 seed42 final adapter → Action v2 SFT best step1604 → R0/window GRPO（已停止）→ RL cp800/1600/2000 test。不是旧 RE/tree checkpoint 初始化。

## 传输范围与来源

精确去重列表见 `dependencies.json`。当前清单 599 个文件，共 19,065,770,739 bytes（17.76 GiB）；其中基座 12 文件 16,397,431,693 bytes（15.27 GiB），项目资产与代码 587 文件 2,668,339,046 bytes（2.49 GiB）。未制作大压缩包，未重算大型模型/adapter SHA。

| 资产 | 文件数 | 逻辑大小 |
|---|---:|---:|
| `tasks/action-experiment/data/action-v2-goldorder-20261012` | 10 | 297.03 MiB |
| `tasks/type-cot-experiment/data/type-cot-recognition-user-final-20261005` | 5 | 180.31 MiB |
| `tasks/type-cot-experiment/data/type-cot-recognition` | 11 | 144.19 MiB |
| `task/prompt/runs/prompt-3.10-1-train-full-merged-20260926.jsonl` | 1 | 41.67 MiB |
| `tasks/action-experiment/runs/action-sft-v2-42-20261012/best-adapter` | 11 | 181.72 MiB |
| `tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot/final-adapter` | 11 | 181.72 MiB |
| `tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/checkpoints/step-800` | 10 | 181.72 MiB |
| `tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/checkpoints/step-1600` | 10 | 181.72 MiB |
| `tasks/action-experiment/runs/action-grpo-r0-window-dev400-42-20261014/checkpoints/step-2000` | 10 | 181.72 MiB |

CoT 原始生成结果位于 `task/prompt/runs/prompt-3.10-1-train-full-merged-20260926.jsonl`。生成脚本 `task/prompt/run_prompt_3_10_full.py` 使用 prompt `task/prompt/3.10-1.md`（脚本默认 3.10.md，原生成需要显式 PROMPT_FILE）；依赖 `task/scievents_appendix_f2_exact_reference_v2.md` 和 `data/SciEvents`。原请求已有完成文本，API 模型/密钥/代理是再次生成才需要的外部依赖，密钥文件不应迁移到 GitHub。`tasks/type-cot-experiment/scripts/build_dataset.py` 将原成果生成 recognition 数据，`rewrite_prompt_dataset.py` 仅替换提示到 user-final 版本，completion 未改。两个数据目录一并保留用于重建来源。

CoT 主训练入口 `tasks/type-cot-experiment/type-cot-goldorder-20261009/run.py`、`train_cot.py`、`config.py`、`sampling.py`、`dense_trainer.py`，seed42 manifest 为 `planned-goldorder-noRE-20261009/manifests/noRE-seed42.json`。Action 主入口 `train_sft.py`、`train_grpo.py`、`action_rewards.py`、`grpo_core.py`、`evaluate_adapter.py` 等，依赖 stage2 scripts/model_utils 与 curriculum Python 源码。JSON 内递归发现的所有 SHA 锁、manifest、tokenizer 输入路径均已纳入；缺失文件为 0。

不能只迁 adapter：SFT handoff 还要父目录 provenance/train_metrics/configuration，Action 数据验收还读 token-audit、manifest、自包含 random15 文件、raw correction input、Sorted1 ordered IDs 和 ontology。CoT 旧 recipe 还引用 sorted-first-online-dev-20261005/manifest.json、sentence-c/raw document 等历史输入。本清单已追到这些依赖，不要求拷贝所有历史 runs。

## 来源锁与历史版本冲突

当前小型锁文件已计算 SHA，检测到两处历史冲突：

- `tasks/action-experiment/train_grpo.py`：SFT 时快照与当前 R0/window 代码不同。
- `tasks/type-cot-experiment/type-cot-goldorder-20261009/config.py`：CoT 运行后补齐完整来源锁，历史快照保留原版本。

`dependencies.json` 为每个锁保留来源文件、历史 hash（可能多版本）、current_sha256 和 currenthash_conflict。绝不能声称所有历史 provenance 当前均通过。每个 run 的 `source_snapshot` 保持原字节，当前 R0 执行与历史 CoT/SFT 重放应区分；历史重放用独立目录/容器装载对应快照，不能覆盖当前 R0。

旧代码大量使用 Path.resolve() 与绝对路径键，因此新机器最好创建真实 `/home/myluo/EE/EXCEEDS`、`/opt/models/Qwen3-8B` 路径，或在相同路径使用 bind mount。普通软链接会被 resolve 穿透，不能保证来源锁和 handoff 路径一致。不要直接改冻结 manifest 路径，更不能关闭校验。

## 结果与可选内容

保留当前 SFT 与 RL dev 全部报告/原始预测、已完成 test 报告及来源输入记录。step1600/800 测试输出当前尚不在 RL run 目录，现有 inputs.json 已保留；实际队列/进度需主代理现场核实。selected best-adapter 与 step2000 两个路径目前重复，保留以兼容已有输入记录，可在传输方案支持硬链接时去重。

较大 rollouts.jsonl 与概率 token/窗口日志列为 optional，可诊断但不是 adapter 推理必要输入。旧 CoT/tree 阶段、其它 seed/RE runs、所有 CoT 训练 checkpoint、SFT last-adapter、RL0/400/1200 都不属默认必要范围。当前停止 run 没有 step2195 optimizer/RNG 完整状态，迁移现有 adapter 不等于能精确断点续训。

GitHub 用于精确核心源码、测试和说明；模型、数据、生成 CoT、adapter、原始预测等使用 rsync。源码候选精确列表为 dependencies.json 的 source_files（149 项，不含历史 source_snapshot）；不要将含 API key 的本地配置、整个 conda 目录或无关历史模型提交 GitHub。
