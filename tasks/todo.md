# 当前任务：Stage-1 Typed Mention Extraction

唯一状态源。[规格](../goal-11/spec.md) · [计划](../goal-11/plan.md) · [日志](../goal-11/experiment_log.md)

- [x] I1 数据/环境/计划核验
- [x] I2 S_G / SD_G 数据与验证
- [x] I3 对齐/解析/评测与测试
- [x] I4 LoRA/推理/队列与真实smoke：训练、adapter保存、推理、评测均已实跑；29项测试通过。
- [x] I5 tmux E0→E1→E1-S→E1 seeds43/44（E1、E1-S及三 seed均完成）
- [x] I6 独立验收与完整report：四个正式 run 的 dev/test 指标、三 seed汇总和Q1-Q3已写入报告。

## 结果复盘
全部split长度无超限，未截断gold。E0、smoke、E1、E1-S及E1三seed均已完成；队列状态为 completed，最终报告已生成。[历史状态](../goal-11/previous-task-todo.md)。

---

# 当前会话：Stage-2 Event Tree 实验理解与结果核对

目标：弄清 Stage-2 的建模对象、T0–T5 的实际代码差异、评测链路，以及当前结果能支持和不能支持的结论。

## 计划与检查项

- [x] 阅读项目教训、Stage-2 README、实验规格和权威报告
- [x] 核对数据构造、prompt、线性化、对齐、官方评分及队列状态
- [x] 对照 Stage-1 报告，确认普通 hints 与 oracle 的含义
- [x] 运行 Stage-2 单元测试并做报告/产物一致性核对：16 项通过；主结果 JSON 与报告一致
- [x] 将理解结果和当前异常/边界归档

## 结果复盘

Stage-2 是文档级 event-forest 生成：Qwen3-8B LoRA 先生成 S-expression 树，再由确定性文本对齐器回到 SciEvents token offsets，最后调用 EXCEEDS 官方 `utils.calculate_F1`。16 项单元测试和 dev gold roundtrip（TI 99.16 / TC 98.58 / AI 94.80 / AC 94.77 / EC 97.23）通过。

- T0 是“有 event schema、无 hints”的真正基线；test AC 39.88。
- T2 普通 hints 使用 Stage-1 E1-42 的真实预测，训练阶段使用按 Stage-1 dev 误差画像生成的伪噪声 hints；test AC 39.21，比 T0 低 0.66。
- T2 oracle 只把推理输入替换成 gold typed mentions，adapter 不变；test AC 56.46，比普通 T2 高 17.25，比 T0 高 16.58。因此它是“理想 mention guidance + 当前 Stage-2”的上界诊断，不是可部署结果，也不能把全部差距严格归因于 Stage-1 单一因素。
- T3 是无 hints 的自然语言事件标签消融（代码把事件码替换为 Proposal/Comparison 等），不是在 T2 上追加自然语言标签；test AC 39.27。
- T4 先用 2 epoch 平铺 trigger/argument 子结构，再用 5 epoch full tree 继续训练；test AC 37.53。
- T5 是无 schema、也无 hints 的对照；test AC 38.78。故 schema 相对 T0 的收益约为 1.10 AC 点，但在 T2 中被低质量 hints 抵消。
- 所有正式模型 invalid=0、触顶截断仅 T5 test 1 篇，未对齐数量也很小；主要问题不是树语法或 parser 崩溃，而是事件/trigger/argument 的联合识别、类型和归属。
- `T1` 正式 run 在 train 阶段以 exit -15 失败，没有有效完整指标；不能纳入结论。
- `tasks/stage2-event-tree/report.md` 对应 2026-09-17 两个队列，后一个队列 completed。工作区另有 2026-09-18 的 `V2-T2warm-*` 手工 warm-start 目录，但只有训练前三步、没有 adapter/prediction/metrics/completed 标记，不能并入 stage2b 报告。

代码归档：已生成合并包 [stage1-stage2-code-20260918.zip](../stage1-stage2-code-20260918.zip)，包含 Stage-1/Stage-2 scripts、tests、README、两份实验规格、ontology 和官方评测依赖源码，共 40 个文件；未包含数据集、模型、checkpoint、训练日志或缓存。zip 完整性检查通过，SHA256=`2d00eaf3d79c036ea507657c20c8f3defc42a3ae9485252fd11c07748fa971eb`。此前的 Stage-2 单独包仍保留。

本次对照 `task/0917-stage2-event-tree-v2.md` 后确认：stage2b 不是按该 v2 规格原样执行，而是执行了一个后续扩展协议。原规格的 T0/T1/T2 语义与实际 run 不一致：原规格 T0=Base+无 schema、T1=Base+schema、T2=Stage-1 warm start+schema；实际 T0=Base+schema、T1=Base+flat predicted hints（且 train 以 -15 失败）、T2=Base+typed predicted hints/schema（不是 warm start）。实际新增了 T3/T4/T5，并使用 5 epochs，而原规格要求首轮 3 epochs、先筛选 T0/T1/T2、再补 seed 43/44。故现有报告回答的是“schema / 显式 hints / oracle / natural labels / curriculum / no-schema”的扩展问题，不能当作原规格 Q3（schema）和 Q4（Stage-1 warm-start）的严格实验结论；尤其没有已完成的 Base vs Stage-1 warm-start 对照。

Experiment C 评估：方向值得做，能直接测试“文档级输入、句子级 decoding”是否缓解长 forest 的结构竞争；但正式运行前必须控制训练预算和 checkpoint 选择。C 的 train rows 会从 2,007 篇文档扩大到 12,828 个句子，固定 5 epochs 不再等价于 T0；并且 T0 当前没有触顶截断（test hit_max_new_tokens=0），所以 C 若提升，结论应是减少 autoregressive 结构竞争/错误传播，而不是解决长度截断。数据上约 15.1% 的 test events 有跨句 argument，约 16.7% 的 test sentences 无 event，必须在 gold roundtrip、merge 去重、局部 trigger/全局 argument 对齐和空句/无事件输出中逐项验收。C 与 T0 还要统一 final checkpoint vs dev-best 规则，并报告 1,627 次 sentence inference 相对 250 次 document inference 的总 token、耗时和显存成本。

验证记录：一次 `jq` 核对因环境未安装 jq 失败，已改用当前 conda 环境的只读 JSON 校验完成；没有改写实验产物。

---

# 当前会话：Experiment C 句子级 Event Generation 实现

目标：按 [task/0918-experiment-C-short.md](../task/0918-experiment-C-short.md) 实现 Full Abstract + Target Sentence 的训练、句级对齐、文档合并和官方评测入口；保持 T0 行为不变。

## 计划与检查项

- [x] 写入并核对实现计划
- [x] 句级数据构造：每句一行、target-only events、全文 context、无事件 `(ROOT)`
- [x] 句级 trigger scope + 全文 argument alignment
- [x] deterministic merge 与预对齐评测入口
- [x] C 队列：长度审计、显式 step budget、dev-best checkpoint、test 评测
- [x] gold sentence trees → merge → official scorer：dev TI 99.26 / TC 98.18 / AI 94.30 / AC 94.26 / EC 95.72
- [x] 最小真实数据 build/audit smoke：train 12,828 rows，max total tokens 1,051，max assistant tokens 243，未截断
- [x] 正式 8B 队列：tmux `stage2-c` 已完成，run `C-42-20260918` 状态为 `completed`，GPU 已空闲

## 实现提交与运行记录

- commit：`ea01b45`，已推送到远端 `main`。
- Stage-2 测试：22 项通过；`git diff --check` 和 Python 编译检查通过。
- C 数据目录：`tasks/stage2-event-tree/data/sentence-c/`；统计 train/dev/test 句子数为 12,828 / 1,593 / 1,627。
- 队列日志：`tasks/stage2-event-tree/C-console.log`；预计总耗时约 7–8 小时，训练 5 epochs、seed 42、默认不限制 optimizer steps。

## 设计决策

- C 使用独立脚本，不改变 T0 的文档级数据和默认对齐行为。
- merge 后输出 document-level aligned predictions，再由官方 `utils.calculate_F1` 评测。
- target sentence 只约束 trigger；argument 在全文范围内确定性对齐。

---

# 当前任务：Stage-3 Structure-Aware Event Tree Refinement v1

目标：按 [task/0918-stage3-tree-refinement-v1.md](../task/0918-stage3-tree-refinement-v1.md) 实现隔离的 D1/D2、reward scorer、R1/R0/R01 诊断闭环；不修改 Stage-2 行为。

## 计划与检查项

- [x] 读取规格、核对 Stage-2 依赖与 C 运行隔离约束
- [x] 写入设计文档：`docs/superpowers/specs/2026-09-18-stage3-tree-refinement-design.md`
- [x] 写入实现计划：`docs/superpowers/plans/2026-09-18-stage3-tree-refinement-plan.md`
- [x] Stage-3 基础模块、候选标签与诊断：隔离脚本、D1/D2 核心纯逻辑、reward 输入/损失/选择原语已实现；Stage-3 测试 74 项通过
- [ ] D1 采样候选池与 support/oracle 统计（tmux `stage3-d1-dev` 已启动，`runs/D1-dev-42-20260919`）
- [ ] D2 类型竞争与前缀一致性
- [x] 两折 reward 数据与合成负样本：`data/folds.json`、`data/folds/fold-{0,1}.jsonl`、`data/confusion_sets.json` 已生成；GPU held-out F0/F1 尚未运行
- [x] Qwen3-0.6B / RoBERTa reward scorer：训练、评分、长度审计与 CPU 测试入口已实现；实际 GPU 训练尚未运行
- [ ] R1/R0/R01 选择、阈值扫描与报告
- [ ] R1/R0/R01 选择、阈值扫描与报告
- [ ] 完整测试、编译、C 完成门禁和 GPU 队列（CPU 74 项通过；D1 GPU 正在运行）

## 当前约束

- `tmux stage2-c` 已完成；D1/D2 现在通过具名 tmux 启动，并由 C 完成/GPU 空闲门禁保护。
- 新代码只能放在 `tasks/stage3-tree-refinement/`；Stage-2 脚本保持不变。
- D1 启动前已重新检查 `run.json` 为 `completed`、项目 GPU 进程为空；当前 tmux 为 `stage3-d1-dev`。

## 结果复盘

待实现和验证后填写；不得把代码完成误写成实验完成。


## 2026-10-12 修复模型完整来源锁并Git提交

规格与设计：用户授权修复审查P2并commit。修改当前type-cot-goldorder/config.py，按safetensors index解析全部实际分片；锁config/tokenizer_config/index与实际tokenizer.json/vocab/merges/特殊token/生成配置，不把旧manifest补写成历史已经锁过。新manifest完整校验，旧弱锁manifest由新版runner拒绝，历史run/source_snapshot/manifest均保留，后续需重新生成独立manifest。SHA256流式读取，不缓存文件内容hash，避免替换绕过检查。不训练/推理/运行旧队列，不提交其他已有改动和实验数据。

计划校验与置信度：根因已独立证实；复用现有sha/verify_locks，标准库json/pathlib，无新依赖；完整源文件清单源于本地transformers/AutoTokenizer加载器及权重index，未找到已有完整锁实现；95%。已有review与本次用户授权覆盖方案，无需再次确认。

- [x] 使用小型本地模型文件fixture写权重/tokenizer/生成配置替换失败回归，确认先红。
- [x] 实现完整模型文件枚举与严格锁校验，补文档和教训。
- [x] 跑CPU回归、实际5分片完整hash覆盖验证，独立review。
- [ ] 核查暂存范围，仅提交此次修复相关文件，记录commit及复盘。

验证复盘：新增12项回归在旧实现全部失败，修复后12项通过；完整gold-order CPU套件34项通过（117.26秒），独立review无阻塞。真实Qwen枚举12项输入，覆盖全部5个分片（总16,397,431,693字节）；完整回归构造及验证实际模型manifest通过。此前记录的5份历史manifest/计划/源码快照SHA256复核未变。仅当前config、新回归、迁移说明及本次任务/教训追加进入提交范围，其他已有dirty代码与实验产物保留。

## 2026-10-12 生成 Action 正式监督数据与随机15条审查

规格：用户授权实现并跑完整数据转换。使用修复gold词序的sentence数据和101202_action_prompt_v1.txt；标准库实现严格action文法解析/序列化，不调用API或训练。事件按触发offset稳定排序（同位置保留gold次序），角色顺序按新prompt而非旧ontology字典次序，重复角色按offset稳定排序；论元内部tokens/offsets顺序绝不重排。NEW/TRG/ARG/END/STOP保持实例次数，None为STOP，compact-tree不增加Trigger节点。保留原gold与metadata，学生prompt仅来自上下文/schema。生成train/dev/test及原Sorted1顺序副本；来源hash、全量往返校验、原文本与offset一致性、真实token长度统计、随机train15条报告均归档。原数据/实验不改，不运行RL。

计划已核对：未发现已有action转换实现；复用标准库JSON字符串、当前训练JSONL字段与固定parser；新prompt的角色顺序与ontology不同，仅验证角色集合一致。现有方案与本次明确执行指令覆盖设计授权，无需再次确认。实施置信度95%。

- [x] 先写解析、重复实例、转义、不连续词序、prompt与排序边界测试，运行确认失败。
- [x] 实现独立action语言模块和生成入口，通过测试。
- [x] 生成正式三split与Sorted1，验证输入hash及全量往返/长度统计。
- [x] 固定seed42均匀抽取train15条，独立对照原始gold逐条检查并交付报告。

进展：基础22项测试通过。独立review发现tokenizer来源锁应覆盖加载前后，按规则停止首轮生成并补2项回归（确认先失败）；加固hash检查和assistant prefix一致性后，25项全部通过。正式生成重新从无输出目录启动，未把中断轮标为成功。

结果复盘：实现scripts/action_language.py与build_action_dataset.py，独立新数据目录生成train12828/dev1593/test1627及Sorted1，合计24381事件56411论元。正式生成tmux退出0；25项新回归+7项presence回归全部通过。全量直接对照data/SciEvents原始gold、输入/输出hash及Sorted1行内容验证通过。seed42均匀抽15条，主代理逐条看gold/action，独立审查15/15一致（19事件52论元，4 None），报告在build-20261012/random15-review.md。实际完整序列max2686、assistant目标max671，2条超过旧319输出上限。保留非连续组成、重复实例/role与原始gold，不对标注重裁决；旧数据、模型与评分器不改，不启动训练/API。

## 2026-10-12 Action SFT 与基础 GRPO 代码实现

用户授权：增加输出上限，编写SFT/RL代码。仅实现与验证，不启动完整Qwen训练/教师API。上限1024，上下文4096；新Action目录独立，不修改历史实验配置与产物。

规格：SFT从已完成seed42 CoT adapter启动，fresh optimizer/scheduler，Sorted1首轮sorted后seed+epoch shuffle、micro1/gradacc16、最多4轮/3208步，沿用warmup97与dev plateau；dev用用户确认的文本存在匹配TC→TI，完全同分留较早。统一存raw actions/compact forest/解析/截断/定位诊断，不丢预测；仅assistant全token(含EOS)监督。GRPO从选出的Action SFT启动，冻结同SFT LoRA reference共享基座，G4，temp1/topP1/topK0/no mask/num_iterations1、lr1e-5/clip.2/beta.01、原始train一次seed42shuffle保留None；全completion token，原始采样不重分词，group std归一、同分优势0，loss按序列长度归一。奖励沿用户最新口径0.6presenceTC+0.3presenceTI+0.1TypeCount；论元奖励延期。dev/test隔离，test通过独立评测入口，仅选中checkpoint。完整基座/数据/代码/adapter hash，记录token预算/行为概率/ref校验与RL诊断。

设计校验与置信度：现有torch2.4.1/transformers4.52.3/peft0.19.1可用、TRL未安装；直接torch实现可显式满足reference与采样概率约束，无新增依赖。已查阅PEFT官方多adapter文档和GRPO官方loss文档，沿用现有tokenizer/collator/sorted sampler与完整模型锁。原方案与本次执行指令覆盖设计授权；95%。两个实现子任务按文件所有权分派，主代理负责共享评分/配置/锁与最终集成。

- [x] 共享配置/来源锁/Action评价与reward：先写边界回归并确认失败，再实现。
- [x] SFT入口、dev控制和真实tiny模型更新/checkpoint测试。
- [x] GRPO数学/采样/reference与入口，真实tiny LoRA冻结/更新测试。
- [x] 集成CPU端到端测试、入口help/preflight，独立交叉审查。
- [x] 完整运行文档、风险边界与验证结果归档。

结果复盘：SFT、GRPO、独立评测与preflight入口已实现；默认生成上限1024、上下文4096，保留完整Action目标。最终CPU回归69项通过（2.74秒），入口help和编译检查通过，最终完整来源锁/数据预算preflight通过。独立交叉审查未发现生产阻断。通过tmux在真实Qwen3-8B上分别完成SFT与G4 GRPO各1次更新：损失有限、初始policy/reference teacher概率完全一致、reference与原CoT adapter不变，短步峰值17.63 GiB。该GPU验证仅生成12 tokens并使用受控奖励验证优化器路径，不代表正式抽取指标；完整1024-token生成显存、全量dev与正式长训尚未验证，未启动正式训练。

实现复核修正：显式关闭Transformers模型默认生成参数回填，防止Qwen默认温度覆盖temp1及greedy dev；直接保存实际generate scores中的行为概率，保留原始token，不以teacher重算替代（真实bf16采样/teacher最大logprob差0.11698，作为诊断记录）。选中adapter的权重和配置均校验SHA。验证报告归档于tasks/action-experiment/verification-20261012/report.md；运行说明见tasks/action-experiment/README.md。

## 2026-10-12 正式启动 Action SFT → GRPO

用户授权：重新验证代码，无阻断后用tmux启动正式训练。先检查历史教训、来源锁、CPU回归及真实1024-token长输出显存；正式SFT seed42完成后，核查completed与best adapter来源再串行启动GRPO。失败即停止队列并保留日志，不自动忽略错误；不并行占用同一GPU，不覆盖旧run。

计划校验：GPU空闲155MiB，磁盘余量281GB，现有conda环境和tmux可用。现有69项回归与短步结果仅作历史依据，启动前重新执行。

- [x] 重新跑CPU回归、完整SFT preflight与只读交叉审查。
- [x] tmux执行真实1024-token长采样/更新/dev显存验证。
- [x] 冻结串行启动脚本与新run路径，启动正式SFT→GRPO队列。
- [x] 核查正式进程、初始loss/状态/日志，归档会话和监控方法。

启动复盘：fresh CPU 69项通过，完整preflight通过，独立生产入口/队列审查无阻断。真实强制1024-token G4采样+GRPO更新及batch4 greedy生成通过，峰值18.46 GiB，reference与旧CoT不变。压力检查的min_new_tokens抑制EOS，行为概率/teacher差值不可与生产默认差值混淆，正式代码没有强制最短长度。

2026-10-12 21:19:51 +08:00启动tmux ee-action-train-20261012，队列脚本launch-20261012/run.sh。正式SFT已完成至少2次optimizer更新，loss 0.3413、0.4800均有限，GPU利用率99%。SFT run为action-sft-42-20261012，GRPO run为action-grpo-42-20261012（尚未开始）；SFT完成并通过GRPO preflight后自动串行继续。此处完成的是验证与启动任务，正式训练仍在运行，未声称训练完成或最终指标通过。状态/日志/监控说明归档launch-20261012/README.md。

## 2026-10-12 暂停训练、三项全量审计、v2 prompt与缩短上限

最新用户指令覆盖继续训练授权：停止SFT/RL队列，保留已产出数据/日志，不自动重启。要求全量原Gold→Action→独立Tree解析一致；全部不连续/非单调片段与重复文本原offset保持及诊断定位限制；真实tokenizer逐条监督/生成长度与截断边界。v2与冻结模板逐字节相同，原v1文件路径缺失，为保留旧来源锁从冻结template恢复相同历史v1文件，不覆盖v2或用户copy文件。新生成上限768（671最大目标+97余量），所有SFT/RL/dev/test一致，上下文4096不变。现有暂停run仍记录旧1024，不篡改历史配置；后续新run使用新上限，不自动恢复。

设计已核对：从1024→768覆盖全gold，512会截断最长样本，319已有2条超长；此配置变更和独立审计已明确授权，置信度95%。无新依赖，复用现有tokenizer/parser/评分契约；token审计加独立模拟768边界检查，不调用模型推理。

- [x] 确认SFT及launcher退出，RL未启动，保留中断记录。
- [x] 全量raw Gold→Action→独立Tree一致性审计。
- [x] 不连续片段/重复文本全量offset审计与歧义案例。
- [x] 缩短统一上限768，重新token审计与回归/preflight。
- [x] v2逐字节核对、RL prompt说明与结果归档；保持不运行。

结果复盘：三份独立脚本全量审计通过，Gold/Action/独立Tree覆盖16048条、2508原文档、24381事件及56411论元，无实例丢失。offset按原顺序验证，2492非连续、819非单调论元保留；文本不能唯一回填重复坐标，诊断不参与删预测。真实tokenizer和assistant mask/截取768解码全量通过，train/dev/test最长295/671/520，768零Gold截断，512有2条超长。69项回归与768完整preflight通过。

期间用户继续修改v2，已重新逐条构造新版输入测长（train平均1831→996）；原稿删除现loader标题/Roles及关键约束，另提供补全兼容建议稿，不改用户文件、不替换正式数据。新稿schema角色集合/顺序验证通过。RL与SFT共用学生prompt，奖励在代码中计算。

停止状态：旧run287步中断，4600条training_step进入记录严格等于Sorted1前缀；无dev、无Actioncheckpoint、无RL。SIGSTOP未保持停止原因未知；实际SIGINT退出，launcher报告exit0不等于completed，已用interruption.json明确标记用户中断。GPU已空闲155MiB，不自动重启。完整复盘与脚本链接见audit-20261012/README.md。

## 2026-10-12 v2短prompt/768正式重启与git提交

用户重新授权tmux启动SFT→RL并git提交，撤销此前暂停要求。采用已交付补全兼容短prompt（保留用户v2原稿），冻结为prompts/action-v2-checked.txt，另建action-v2-goldorder-20261012数据，Action/Tree/gold/Sorted1不变。SFT/RL共用同一新数据、上限768；原287步中断run保留不续训，fresh optimizer新run从原CoT启动。先新数据三项全量审计/CPU回归/预检，再启动。运行源码冻结，不改训练期间代码。git只提交本次代码/必要依赖/prompt/可复现实验说明与小型证据，不提交大数据/权重/历史日志或其他dirty改动，不推送。

- [x] 冻结兼容v2 prompt并生成新数据，验证答案与旧数据相同。
- [x] 新数据三项独立全量审计、回归和完整来源preflight。
- [ ] 新tmux串行队列启动，检查真实首批loss。
- [ ] 核对git提交范围，提交代码/依赖/配置与说明，记录commit。

启动前复盘：新短prompt数据完整生成，逐条仅prompt变化；三项全量审计通过，70项CPU回归通过，完整preflight包含sampling.py来源锁并再次逐文件hash确认一致。2026-10-12 23:24:54 +08:00已启动tmux ee-action-v2-20261012，SFT fresh CoT接续，完成后完整GRPO preflight再继续RL。新run路径action-sft-v2-42-20261012/action-grpo-v2-42-20261012；检查首批更新后记录最终启动状态。git采用显式范围，只存代码/必要依赖/prompt与小型证据，其他dirty内容和大数据保留工作区。
