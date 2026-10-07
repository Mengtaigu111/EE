# Sorted1 训练中 dev 反馈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` or `executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在最近 Sorted1 的相同 prompt、数据、课程和模型初始化上，接入在线 dev 验证、平台期降学习率与早停，上限 3,208 次更新。

**Architecture:** 新写一个继承现有 SortedFirstTrainer 的 Trainer，训练过程中使用当前驻留模型生成完整 dev，再用现有修复版解析/对齐和官方 offset 评分。一个连续 Trainer 保留 optimizer/采样状态；独立反馈控制器管理最佳分数和等待计数，替换 cosine。所有新源码与运行目录独立，不覆盖历史实现。

**Tech Stack:** `/home/myluo/miniconda3/envs/qwen3meld-run/bin/python`；现有 torch 2.4.1、transformers 4.52.3、PEFT、bf16、单 RTX4090、tmux。

**状态（2026-10-05）：** 仅方案，未实现新训练代码、未启动 GPU 训练。用户已明确：最近 Sorted1 为基础，prompt 和课程一致，只调整 dev 反馈与已确认预算。旧 Git 提交 `ee8e203`。

## 1. 冻结实验规格

| 项目 | 固定值 |
|---|---|
| 参照实验 | `C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003` |
| 初始化 | 原始 `/opt/models/Qwen3-8B`，新 LoRA；不从旧最佳 adapter 续训 |
| Prompt | 原 AppendixNoCopy，不换 UserV2；无 Region、Semantic Decision、At、显式推理 |
| Prompt SHA256 | `2bcc66773a71904973ae4f85b865e60eddfacf4230515618576d3217e6776b41` |
| 输入/输出 | 完整摘要＋目标句→该句 trigger 的事件森林；论元可跨句；无事件 `(ROOT)` |
| 数据 | 原 train12828/dev1593/test1627，不重新划分、不改 gold/样本内容 |
| 课程 | 首轮沿用冻结 Sorted1 顺序及 ROOT 插入；后3轮 `randperm(seed+epoch)` |
| 难度 | 沿用原 mean 代理和同分顺序，不重算、不通过 dev 自动切课程 |
| LoRA | r16、alpha32、dropout0.05，q/k/v/o/gate/up/down_proj |
| 优化器 | AdamW，初始学习率1e-4；保持原 betas/epsilon/weight_decay 等配置 |
| batch/长度 | microbatch1、梯度累积16，max_len4096，bf16、gradient checkpointing |
| seed/loss | 42；仅 assistant 答案 token loss |
| 最大预算 | **3208 optimizer steps，约4轮**；不是3208个microbatch，也非从旧模型再加3208步 |
| dev频率 | **401 optimizer steps**：401/802/1203/1604/2005/2406/2807/3208 |
| warmup | 比例0.03；固定 `ceil(3208×0.03)=97` 次更新，真实线性warmup |
| warmup后 | 保持当前平台学习率，dev达到降率条件才×0.5；移除cosine，不叠加两套衰减 |
| 选模/控制指标 | 完整dev、修复解析/对齐后的**官方 offset AC_f1**（0–1尺度） |
| 其他指标 | 每次同批预测保存TI/TC/AI/AC/EC_I/EC_C全部P/R/F1与解析/对齐诊断 |
| 解码 | 保持近期Sorted1：greedy、batch4、max_new_tokens319、bf16、thinking关闭 |

预算和学习率日程与旧实验不同，因此新结果属于“Sorted1＋dev反馈＋4轮上限”联合实验，不能将变化只归因于早停。

## 2. 现有来源与允许改动

根目录为 `/home/myluo/EE/EXCEEDS`，以下相对路径均以此为准。

**复用并冻结：**
- `tasks/stage2-event-tree/curriculum/sorted-first-20261003/{manifest.json,train-sorted.jsonl,ordered-ids.txt,prompt.md}`。
- 原句子数据 `tasks/stage2-event-tree/data/sentence-c-appendix-no-copy-20261002/{train,dev,test}.jsonl`；文档gold `tasks/stage2-event-tree/data/none/{dev,test}.jsonl` 及 `data/SciEvents/{dev,test}.json`。
- 近期run的 `training/config.json`、baseline `C-AppendixNoCopy-42-20261002`，仅作为配置/历史比较来源。
- `tasks/stage2-event-tree/curriculum/{sorted_first_sampler.py,curriculum_trainer.py}`；`tasks/stage2-event-tree/scripts/model_utils.py`。
- `tasks/stage2-event-tree/evaluation-v2-20261003/scripts/{ontology,linearize,align_events,merge_sentence_predictions,evaluate}.py`、官方 `utils.py`。
- 修复模块必须显式加载并校验 `__file__`，可复用 `verify_sorted_first_run.fixed_evaluation_modules()`，避免旧同名模块缓存。

**新增代码（均放 `tasks/stage2-event-tree/curriculum/`）：**
- `dev_feedback.py`：纯状态机＋线性warmup/平台学习率调度器。
- `online_dev_evaluation.py`：当前模型的dev生成、固定解析/对齐、评分与上下文恢复。
- `online_dev_trainer.py`：继承SortedFirstTrainer，连接evaluation、反馈与安全checkpoint钩子。
- `train_sorted_first_online_dev.py`：只做训练，参数/数据hash审计、实际采样证据与最终状态。
- `run_sorted_first_online_dev.py`：配置冻结、运行训练子进程、训练结束后最佳模型test评估与报告。
- `verify_sorted_first_online_dev.py`：独立重算反馈轨迹、最佳选择、采样前缀、源码hash和评分。
- 对应 `test_dev_feedback.py`、`test_online_dev_evaluation.py`、`test_online_dev_trainer.py`、`test_sorted_first_online_dev.py`。

新配置目录 `tasks/stage2-event-tree/curriculum/sorted-first-online-dev-20261005/`，复用数据而非重写样本。新run名 `C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev-42-<实际启动日期时间>`；目录必须exclusive创建，禁止复用旧run。**旧训练脚本、prompt、解析对齐、评分器、旧run均不修改。**

## 3. 反馈状态机：必须固定的行为

F1统一用0–1值，`min_delta=0.001`即0.1个百分点；不得误写0.1或0.0001。

状态：`best_score/best_step`、`anchor_score`、`bad_count`、`eval_count/last_eval_step`。首次分数用None表示未初始化，不向JSON写Infinity/NaN。

```python
# 每次成功的完整dev评估后；失败/缺失/非有限分数必须报错退出。
new_best = best_score is None or score > best_score or (score == best_score and step > best_step)
if new_best:
    best_score, best_step = score, step  # 任意微小提升也保存，精确平局取后step。
if anchor_score is None or score - anchor_score >= 0.001 - 1e-12:
    anchor_score, bad_count = score, 0
else:
    bad_count += 1
stop = bad_count >= 6
reduce_lr = bad_count == 3 and not stop
# reduce_lr: 当前学习率×0.5。保持bad_count，不重置总等待计数。
# stop优先于降率；同一平台期只有第3次bad降一次，第6次bad停。
```

anchor与raw best分开：40.50→40.54仍保存40.54，但等待不清零；累积到40.62相对anchor超过0.1pp才清零。有效改善后保持已经降低的学习率，开始新的等待窗口。

scheduler `.step()`每个optimizer更新调用一次；维护`last_epoch==global_step`。warmup后为`base_lr × plateau_scale`，`reduce_lr`只改变scale和当前LR，不增加步数。提供`state_dict/load_state_dict/get_last_lr`，禁止使用HF自动plateau再额外手动step，禁止cosine下次覆盖已降低LR。控制器只负责发事件，学习率scale由scheduler唯一持有。

初次验证401步建立anchor；若之后一直无有效改善，1604步降率、2807步早停。第3轮才达到最佳时，可能跑到3208上限而不会触发早停，这是正常情况。最终模型由raw best决定，不能用最后模型替代。

## 4. 训练循环与GPU评估

- 采用一个连续Trainer，`eval_strategy='steps'`、`eval_steps=401`、`save_strategy='steps'`、`save_steps=401`、`max_steps=3208`；保存全部验证checkpoint，无自动轮换删除。
- transformers初始化要求steps评估有非空`eval_dataset`，即使override evaluate也不能省略；传独立dev行Dataset满足契约，自定义评估仅从prompt生成，不对dev gold做teacher forcing，不混入train loader。
- **新实现不能保留旧`eval_strategy='no'`或训练结束后才循环评估dev。** `OnlineDevTrainer.evaluate()`须在同一进程使用当前PeftModel推理，不在训练模型驻留GPU时再加载另一份8B。
- 按原dev行顺序、完整1593句生成；输出字段保持原prediction schema。相同原始预测同时生成官方offset全18项；new instance/raw-token可作为独立诊断保存，但不参与降率/停训/选模。不得训练中读test成绩。
- 评估需保留并在finally恢复：模型train/eval状态、`use_cache`、Python/NumPy/CPU/CUDA RNG，tokenizer padding设置。优先独立eval tokenizer（left padding）；训练collator仍right padding。
- `inference_mode()`下临时启用cache。验证在optimizer边界进行，不反向传播、不更新optimizer、不改变采样器/梯度累计/训练RNG。GPU冒烟验证batch4显存；OOM不能偷偷改长度、缩dev或改prompt。
- 覆盖transformers evaluate/回调钩子时保证只调用一次on_evaluate；依次执行：生成与评分→反馈/降率/stop→更新best（含后step平局）→保存当前完整checkpoint。确认HF内部best逻辑不会覆盖自定义平局规则。
- 每次记录`step,epoch,18项指标,LR_before/after,raw_best,anchor,bad_count,reduce/stop,wall_seconds`；验证输出路径带step且不得覆盖别的step。
- 不重建optimizer，不重新开始epoch；实际训练观察序列按旧Sorted1同seed顺序推进。独立验证完整epoch和最后未完成epoch的正确前缀，不能套用旧“必须5轮完整”的validator。
- checkpoint含adapter、optimizer、scheduler、RNG、Trainer状态、feedback_state；沿用JSON+safetensors序列化，扩展新checkpoint integrity manifest覆盖反馈状态。早停step也必须保存。final adapter加载best，另保留last，路径清晰区分。
- 初版主任务从原模型新训练，**不是旧run续训**；如果实现新run续训，必须同时恢复控制器计数、scheduler scale、optimizer、RNG、样本位置和已评估step，禁止重复累计等待或静默重置。原SortedFirstTrainer会独占创建observation文件；新resume逻辑必须使用新的日志分段并恢复microbatch计数，verifier按段拼接审计，不能对已有日志重新open(x)。

## 5. 任务分解与验证门槛

### Task 1：配置冻结与反馈控制器

- [ ] 先写失败测试：0.001尺度、微小提升保存但不清零、累计提升清零、exact tie后step、bad3降率/bad6停止、降低后改善不恢复高LR、NaN/重复或非递增step报错。
- [ ] 实现dev_feedback模块；另验证warmup97、每update一次step、平台降率不被下一step覆盖、state往返恢复。
- [ ] 新manifest记住所有输入/旧源码hash及冻结值，验证课程顺序/prompt hash和原行完整一致。测试硬编码3208/401与不从adapter续训。

### Task 2：驻留模型dev评估

- [ ] 用轻量fake model验证生成→parse→merge→score的调用与完整ID覆盖；fake只用于控制流测试，不能作为实验成绩。
- [ ] 测试正常/异常路径都恢复训练状态、RNG、padding/cache；验证参数与旧generate一致，dev行不会进入训练loss。
- [ ] 从一组历史dev原预测重算全18项，必须与原Saved official scores一致；这一步无需GPU。
- [ ] 实现online_dev_evaluation；拒绝重复/缺失ID和数据截断。记录invalid/max-token数量，它们影响结果，不能静默丢弃以提高分数。

### Task 3：Trainer与安全checkpoint

- [ ] 实际小型CPU Trainer跨optimizer步检查：真实on_evaluate在训练结束前触发，后续LR/stop随反馈变化，best/last路径正确。
- [ ] 注入预设分数仅测试状态机分支，证明第3bad降率、第6bad停且停止在optimizer边界；实验metrics必须来自真实生成。
- [ ] 验证validation不改变实际采样前缀；早停partial epoch合法，重复/错序样本非法。
- [ ] 完整state保存/恢复测试，不得只保存adapter后重置等待。检查反馈状态包含在continuity hash中。
- [ ] 实现新的train入口和Trainer。原Sorted1训练代码不改，旧validate_recipe/sampling_audit/verify_training硬编码4010和5epochs不可直接用于新run。

### Task 4：Runner、独立verifier与报告

- [ ] 新runner只在训练子进程正常结束且独立验收通过后运行test；训练进程退出释放GPU后，再复用旧独立test预测流程。
- [ ] 对最佳checkpoint评估test一次：官方offset、官方raw-token、新实例offset/raw-token及EC_I/EC_C完整P/R/F1并列，明确各自口径。禁止用test结果换best。
- [ ] verifier由日志从头重放反馈事件、每次LR变化、raw best选择；核对实际updates≤3208、停止原因为early_stop或budget_exhausted。
- [ ] 对每个dev原预测独立复算官方全18项与日志一致，核对数据/旧源hash，实际观察顺序是原4轮计划的正确前缀。
- [ ] 报告与旧Sorted1/baseline比较时说明新增“在线反馈＋4轮上限”，早停不是必然涨分；官方与诊断口径不得混比。

### Task 5：GPU冒烟、tmux主实验与验收

- [ ] 先跑全部新CPU测试及相关旧回归，独立代码review通过。
- [ ] GPU冒烟独立run：原train中最长的32个真实样本保持课程顺序，4次更新、每2次评估，dev选择至少2篇完整摘要的全部句子，max_new_tokens319/batch4；预算例外只能由明确smoke配置触发。
- [ ] 实际证明训练→dev生成→恢复训练→dev→保存→独立加载同adapter预测结果一致；观察LR、mode恢复和GPU峰值。CPU分支测试不替代这项验证。
- [ ] 冒烟失败先定位修复，禁止直接启动主任务。
- [ ] 冒烟通过后，在新tmux session `ee-sorted1-online-dev-<实际日期时间>` 启动新runner，日志、exit code、config/source snapshot保存在新run。
- [ ] 启动后核查真实步数和首轮消费顺序；按实际训练/验证耗时更新预计完成时间。最多8次完整dev生成，比旧5次更多，不能因步数少20%就声称总耗时也少20%。
- [ ] 完成后独立verifier passed、全量test覆盖、best checkpoint存在；写报告，更新交接与todo再宣布完成。

## 6. 新窗口执行入口（命令为将来接口，当前文件尚未创建）

```bash
cd /home/myluo/EE/EXCEEDS
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m unittest discover -s tasks/stage2-event-tree/curriculum -p 'test_*online_dev*.py'
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python -m unittest discover -s tasks/stage2-event-tree/curriculum -p 'test_dev_feedback.py'
```

实现runner至少接受`--output <全新run目录> --smoke`（冒烟）或`--output <全新run目录>`（正式），内部读取独立manifest。两者都要求tmux；命令中的新目录必须由执行者使用实际日期时间生成，不能复用历史run。

当前只交付完整方案。下一窗口已获授权实施该方案，完成CPU验证、GPU冒烟和独立review后启动tmux主任务；实现中发现需改变prompt、课程、数据、评分口径等范围时，先回到规格核对，不自行扩大实验。

**方案审查：Approved。** 见[独立审查](08-online-dev-plan-review.md)；[来源与文档核验](online-dev-plan-verification.json)已验证31个历史源码hash、12个Sorted1manifest来源hash、文件链接与步数规则。这里只证明方案与冻结来源，不能当作新训练实现已通过测试。
