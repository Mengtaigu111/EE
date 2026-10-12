# Action 监督数据

从已修复 gold 论元内部词序的正式句子数据构造 `NEW/TRG/ARG/END/STOP`，不调用教师 API。输入模板使用 `task/101202_action_prompt_v1.txt`，包括带 `[S1]` 等标记的完整摘要与目标句。监督字段为 `prompt` 与 `completion`，与当前 assistant-only SFT 数据接口一致；其他 gold 字段仅作审计，不能追加进学生输入。

入口：`scripts/build_action_dataset.py`。文法与文本事件转换：`scripts/action_language.py`。输出：`tasks/action-experiment/data/action-v1-goldorder-20261012/`。

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python scripts/build_action_dataset.py \
  --tokenizer /opt/models/Qwen3-8B \
  --output tasks/action-experiment/data/action-v1-goldorder-20261012
```

输出目录必须不存在。重复构建请指定独立新目录；不能覆盖已有数据。完整运行的命令和日志在 `build-20261012/`；本次使用 tmux，没有训练或生成模型推理。

- `train.jsonl`、`dev.jsonl`、`test.jsonl` 保留原 split 与行顺序。
- `train-sorted.jsonl` 和 `ordered-ids.txt` 保留原 Sorted1 的精确样本顺序及内容。
- `prompt_template.txt`、`manifest.json` 记录模板、角色顺序、来源与输出 SHA256、数量和全量验证。
- `token-audit.json` 使用实际 Qwen tokenizer，关闭 thinking、不截断，核查 assistant token 前缀并统计含结束 token 的目标长度。
- `random15.jsonl` 和 `random15.md` 固定 seed42，从 train 均匀无放回抽15条，展示完整输入、原始 gold、动作答案和精简树。独立检查见 `build-20261012/random15-review.md`。

事件按 trigger 最小 offset 稳定排序；同位置保持原 gold 顺序。角色使用新 prompt 的 Roles 顺序（有些与旧 ontology 字典顺序不同），同角色按最小 offset 稳定排序。排序只作用于事件／论元记录，绝不重排 mention 内部 token 或 offset；保留同触发同类型事件、重复角色和跨事件共享文本。无事件仅输出 `STOP`。

严格 parser 拒绝缺 END／STOP、STOP 后内容、未知类型／角色、空文本和非法状态；文本使用 JSON 转义。全量转换核查触发范围、文本与原始 offset 一致、事件与论元次数、动作读回后内容不变；任何失败均停止，不删掉异常样本。

`compact_tree` 是新方案的 `(ROOT (TYPE "trigger" (Role "argument")))`，不能直接交给旧的 Trigger 子节点树 parser。当前 Action 评分从解析后的文本事件记录接入存在匹配；新SFT/RL入口见下文。旧训练runner、数据与评分器保留。

验证：

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m pytest -q tasks/action-experiment/tests/test_action_dataset.py
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/action-experiment/build-20261012/verify_data.py
```

## 本次生成结果

| split | 样本 | 事件 | 论元 | None样本 |
|---|---:|---:|---:|---:|
| train | 12828 | 19488 | 45076 | 2124 |
| dev | 1593 | 2445 | 5618 | 271 |
| test | 1627 | 2448 | 5717 | 272 |

32项CPU测试通过；16,048条正式数据与原始文档gold全量对照通过；随机15条独立审查15/15通过。完整训练序列最长2,686 tokens，assistant目标最长671 tokens（含结束token），2条目标超过旧319上限。原4096上下文预算足够覆盖本批完整目标；未来生成上限须重新配置。

## SFT 与 RL 入口（2026-10-12）

Action 输出上限已调整到 **768 tokens**（2026-10-12暂停后更新；此前正式启动使用1024）；SFT／RL／dev／独立 test 共用该默认值。输入上下文维持4096，完整 gold 不截断。当前最长答案671 tokens、完整训练序列2686 tokens。

- `train_sft.py`：加载已完成 seed42 CoT LoRA，重新建立 optimizer/scheduler。Sorted1 首轮顺序、后续 seed+epoch shuffle；微batch1、累积16、最多4轮3208步、warmup97。仅监督 assistant token（含EOS），按真实积累组的目标token数归一化，正确处理尾批。原 plateau 规则保留：TC anchor改善至少0.001才重置耐心，3次减半LR、6次早停。
- `train_grpo.py` / `grpo_core.py`：从 Action SFT `best-adapter` 开始，原始train seed42 shuffle一次，保留全部None；G4逐条采样、默认积累4个prompt组，lr1e-5、clip0.2、beta0.01、单次rollout更新。序列长度归一化的GRPO，sample-std组优势，同分优势0；含EOS、排除pad，截断样本保留已采样token并记录。
- `action_evaluation.py`：严格解析动作，直接用文本事件记录接入已有presence签名／命中计数。奖励 `0.6 TC + 0.3 TI + 0.1 TypeCount`；有效双方空为1，解析失败为0。论元奖励按用户指令延期。未定位的合法预测仍参与评分，解析失败在corpus评分中保留显式unmatched FP；诊断定位不参与删预测或借gold选位置。
- `evaluate_adapter.py`：只加载已完成run中dev选出的best adapter，独立评估dev/test，不以test选模。所有阶段按自定义文本存在匹配 TC→TI 排名，完全同分保留较早checkpoint。
- `preflight.py`：只检查输入、完整基座文件hash、CoT/SFT来源和实际tokenizer前缀／预算，不加载模型或训练。

RL 的基座共享，policy/reference分别加载相同 SFT LoRA，reference始终冻结。更新前核对actor/reference概率相同，结束核对reference权重不变。RL使用采样时实际scores计算行为logprobs，不重新格式化或重分词；与teacher-force数值差单独记录。LoRA dropout在RL禁用，temperature1/topP1/topK0，无合法动作mask。

特别处理了 Transformers 4.52 的模型默认参数回填：所有generate通过 `generation_arguments()` 显式设置 `use_model_defaults=False`，避免 Qwen 原默认 temperature0.6/topP0.95覆盖本轮设置或把greedy dev改为sampling。

基座完整5分片及tokenizer、数据、代码、初始化adapter都做内容锁；SFT→RL另校验继承的完整基座、数据和评分实现一致。选中adapter的权重和LoRA配置均校验hash，防止同权重但alpha等配置被替换。保存raw动作、实际token、行为/reference概率、解析／截断／保守定位诊断、dev指标、best/last adapter及预算。训练不读取test答案进行更新或选模。

### 运行

以下命令用于新的正式run。此前1024上限的SFT已按用户要求中止，RL尚未启动；本次768配置未重新启动训练。输出目录必须不存在；不得覆盖旧run。使用已验证的conda Python，在tmux中运行。

```bash
cd /home/myluo/EE/EXCEEDS
# 只读预检查，报告文件须不存在
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/action-experiment/preflight.py \
  --stage sft --output /tmp/action-sft-preflight.json

# 正式SFT；完成后再启动下面的RL
tmux new-session -d -s ee-action-sft \
  '/home/myluo/miniconda3/envs/qwen3meld-run/bin/python /home/myluo/EE/EXCEEDS/tasks/action-experiment/train_sft.py --output /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-42'

# 必须等待SFT train_metrics.json为completed且best-adapter通过来源验证
tmux new-session -d -s ee-action-grpo \
  '/home/myluo/miniconda3/envs/qwen3meld-run/bin/python /home/myluo/EE/EXCEEDS/tasks/action-experiment/train_grpo.py --init-adapter /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-sft-42/best-adapter --output /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-42'

# 两组各自选中的best可独立测test；结果不回传训练控制
tmux new-session -d -s ee-action-grpo-test \
  '/home/myluo/miniconda3/envs/qwen3meld-run/bin/python /home/myluo/EE/EXCEEDS/tasks/action-experiment/evaluate_adapter.py --run /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-42 --split test --output /home/myluo/EE/EXCEEDS/tasks/action-experiment/runs/action-grpo-42/test'
```

SFT与RL的 `--smoke` 会减少数据并标记为非正式；smoke SFT adapter不能初始化正式RL。没有安装TRL，也没有更换现有环境，使用已安装PyTorch/Transformers/PEFT实现目标。

### 验证边界

结果在 `verification-20261012/`：CPU回归包含真实tiny PEFT/Trainer更新、GRPO G4更新、冻结reference与重载、EOS/尾批/生成参数回填回归。实际Qwen3-8B/4090另完成SFT一步、GRPO一步；reference与原CoT权重不变，行为概率直接捕获。该短步测试使用12-token采样上限和受控测试reward，验证代码与环境，不能解释为模型效果或正式训练完成；也没有实跑完整dev、1024-token长rollout或四组对照。

实际GPU短步峰值约17.6 GiB；完整批次/长序列显存仍需正式smoke确认。默认dev batch4可用SFT的 `--dev-batch-size` 调整；独立评测支持 `--batch-size`。目前不支持中断后的optimizer/RNG自动resume，checkpoint用于评测和阶段初始化，重试用新run目录。

## 暂停后的全量审计与新版prompt

2026-10-12用户要求暂停，旧SFT在287步中止，队列未进入RL；无完整optimizer/RNG续训支持。旧run的配置和provenance保留1024，不能把旧run标记为新768配置已训练或completed。相关记录见launch-20261012/interruption.json。

全量原始Gold→Action→独立Tree检查、不连续/重复文本offset检查与真实tokenizer长目标检查归档audit-20261012/。纯文本Action不能唯一恢复重复表达的原始offset；保留原gold坐标用于审计，presence评分直接使用文本，不借gold回填预测位置，不删除诊断不匹配的预测。

用户最新task/101202_action_prompt_v2.txt已大幅压缩，真实train输入平均1831→996 tokens。该原稿尚未替换正式训练数据：生产load_schema依赖旧标题与Roles行，且简化稿删掉非连续片段组合、JSON转义及稳定顺序规则。兼容建议稿为audit-20261012/prompt-v2-recommended.txt，补回这些规则和显式Roles，未修改用户原稿；采用新prompt时需另建新数据版本及来源锁，SFT/RL同时切换，不覆盖历史数据。

RL有prompt，当前与Action SFT同一数据模板：完整摘要、目标句、事件schema和Action输出规则。RL仅用奖励更新策略，不向模型输入gold答案或额外奖励说明。

## 已采用兼容v2短prompt的新轮次

用户重新授权训练后，正式默认数据切换到`data/action-v2-goldorder-20261012/`，SFT与RL共同使用`prompts/action-v2-checked.txt`，生成上限768。模板采用用户短版v2并保留审计必需的组合/转义/排序和显式角色规则；用户原v2文件未修改。新数据与旧数据逐条仅prompt不同，三项全量审计、70项回归和完整来源preflight通过。旧287步中断run与v1数据完整保留，新run从原CoT LoRA启动。

[启动前结果](audit-v2-20261012/README.md) · [新tmux队列](launch-v2-20261012/README.md)。SFT、RL、dev/test全使用相同模板，训练期间冻结源码。git归档代码、模板、必要依赖与小型验证证据，不包含数据JSONL、模型/adapter权重或训练日志。
