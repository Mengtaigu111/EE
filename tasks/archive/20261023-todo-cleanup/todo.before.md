# EXCEEDS 当前任务

## 2026-10-09 启动六cp补测→修复词序noRE42/66

最新授权：seed58暂不跑；原六个RE checkpoint补测保留，接修复gold-order noRE42、66，各保留CoT一轮。从已CPU验证实现收缩执行清单，旧三seed计划保留归档，新执行计划仅两seed。review后先独立tmux GPU smoke，验证CoT→树→checkpoint reload及来源/有限损失；通过后新tmux串行补测六cp→noRE42→66，失败停止，确认首批GPU推理/状态。不提交git、不改历史产物。

- [x] 两seed清单回归先红后绿、冻结新计划，独立review。
- [x] tmux独立GPU smoke完整通过，日志/退出码/源码hash验收。
- [x] tmux启动正式串行流程，核实六cp补测首批进度及58没有启动。
- [ ] 六cp和两个noRE完整结束后验收结果。


## 2026-10-09 六checkpoint补测与修复词序noRE三seed（仅写代码）

规格：RE42 cp2406/1804/2005；RE66 cp1604/1804/2406，全部生成完整test并用scripts/score_tree_presence.py评价，保留现有dev-best和历史产物。独立新目录、来源锁、完整覆盖/有限结果/每次日志及失败状态，不按补测test重新选模。修复gold-order noRE：seed42/66/58，各原始Qwen3-8B/新LoRA、一轮CoT802更新/cosine warmup25，树修复数据Sorted1、最多3208/13dev反馈、原官方offset devAC选模并附自定义指标；与旧noRE原配方一致只更正树监督及seed，作为对应RE对照。三个seed依次跑，预备入口默认只检查/dry-run，显式执行才启动；本轮不调用tmux、不启动GPU、不训练/推理。

架构与计划校验：六cp补测复用predict_appendix_fixed.py与score_tree_presence.py。noRE独立复制已完成seed52训练实现，重建修复数据来源锁，参数化seed42/66/58；不改旧源码/manifest/run。CPU测试先失败再实现，验证cp清单/checkpoint存在/初始化/数据/预算/种子/命令与失败停止；独立review交叉审查。无新依赖，置信度95%。

- [x] 六cp独立补测计划与执行/验收入口，CPU测试。
- [x] 独立修复gold-order noRE三seed配置及训练/验证入口，CPU测试。
- [x] 仅dry-run/source/data/checkpoint验收、独立review及文档，确认没有启动tmux/GPU。


## 2026-10-09 自定义评价代码统一重评

用户要求按我们的评价代码比较test并列出加RE所有dev cp、标记每列最佳及平均。统一scripts/score_tree_presence.py，复用保存预测，不训练、不重新生成、不改历史选模。四个dev-best test（无RE42/52、RE42/66），RE dev42共13点、66早停共9点；按相同步数均值，42后四点不冒充双seed均值。完整P/R/F1存JSON/CSV，展示六项F1；最佳限表内不称文献SOTA。验证来源hash、覆盖、test与已有presence结果一致、独立交叉核验均值/最大值。

- [x] 用当前同一评分代码重评四组test与全部22个dev预测。
- [x] 计算同cp双seed均值、逐列最佳，保存完整结果与报告。
- [x] 覆盖/来源/既有结果一致及独立交叉核验，展示结果。


## 2026-10-08 追加 seed66 队列

用户授权在seed42后用tmux串行追加seed66；同数据/预算/评测，新模型新LoRA，独立源码/manifest/run，不改运行中的seed42源码或配置。计划校验：原代码将42写死在config与最终验收，故复制到同层独立seed66目录，仅改种子及展示文字；验证两manifest除种子/实验名外等价。队列等待seed42退出码0、完成状态及原tmux结束后启动；上游失败或会话丢失则停止并留日志，不争用GPU。复用已通过GPU smoke的相同训练逻辑，运行CPU回归及队列成功/失败条件测试，再启动独立tmux并确认处于等待。

- [x] 建独立seed66代码和manifest，核验差异与来源hash。
- [x] CPU回归、队列条件测试与独立交叉审阅。
- [x] 启动tmux等待队列，保存状态/日志/退出码入口。
- [ ] seed66正式训练与评测完成后归档结果。

排队复盘：18项CPU测试通过，独立review无阻塞；manifest仅种子/实验名变化，seed42冻结源码hash未变。tmux `ee-cot-repos-66-after42-20261008`已运行，status.json=waiting_for_seed42，尚未加载seed66模型。证据goal-26/seed66/prelaunch-verification.json、cpu-tests.log、queue.log与status.json。

## 2026-10-08 CoT + 正关系RE 试验

规格与计划已核对：[goal-26](../goal-26/plan.md)。用户授权先跑不含NONE的一版；独立seed42、新LoRA、CoT12828/RE7991各一轮不重复，按任务进度交错802/500次更新，接修复后树Sorted1最多3208次，现有官方offset dev AC选模后test。旧数据/run不变，不调用API。

- [x] 构建正关系RE SFT，验证候选/答案同步和全量正关系保留。
- [x] 实现独立混合训练与树衔接，CPU测试/独立review。
- [x] tmux GPU smoke全流程通过；恢复训练状态/覆盖核查通过，但不逐位复现，详见goal-26/review.md。
- [x] tmux正式训练及评测，记录日志/退出码/结果复盘。结果：goal-26/main-result.json与run/report.md；exit0，全量验收通过。

启动核验：tmux `ee-cot-repos-main-42-20261008` 正在运行，CoT/RE首批更新loss与梯度有限；详见goal-26/startup-verification.json。全量训练与最终评测尚未完成。

## 2026-10-08 论元内部词序修复

规格：只修树监督构造，RE/CoT/评分器/旧run不改。保留原gold有序offsets及token文本；事件/论元列表仍按位置和schema排序，位置排序使用min/max而不改片段内部顺序。补充独立原gold对照，拒绝标注text/tokens/有序offset不一致。新版本保存句级train/dev/test及沿用旧ID顺序的Sorted1，保留旧冻结数据和配置。不开训练。

计划校验：根因已复现为span_text及canonical_events排序去重；现有roundtrip仅自洽，需新增原gold比较。沿用Python/unittest，无新增依赖，复用已有树构造器及固定课程ID序列；实现置信度95%。

- [x] 回归测试先失败：非单调及重复offset、真实two knowledge graph datasets、原gold校验。
- [x] 修正共享linearize与句级构造验收，保留列表排序规则。
- [x] 构建独立修复数据及Sorted1版本，逐字段核对只改变错误completion。
- [x] 运行相关回归、全量gold文本/offset核验及来源hash检查，独立review。
- [x] 记录新入口、修复计数与旧数据保留证据，归档结果复盘。

验证标准：658/66/95个论元恢复gold词序；558/58/85行completion变化；train/dev/test行数12828/1593/1627、prompt/ID/metadata/课程顺序不变；原始数据与旧冻结产物hash不变。错误构造必须被独立gold验收捕获。

结果复盘：[修复与新入口](stage2-event-tree/argument-order-repair-20261008/README.md)。核心回归35项及独立构建/验收9项通过；新增回归先失败后通过。tmux完整构建exit0，全量逐字段及原gold签名检查通过，50项来源/旧产物/输出hash一致；独立review无阻塞问题。新数据位于`data/sentence-c-appendix-no-copy-gold-order-20261008`，Sorted1位于`curriculum/sorted-first-gold-order-20261008`（均相对stage2-event-tree）。旧数据、模型、RE和冻结训练配置保留，没有启动训练。

更新：2026-10-07。这里是唯一任务状态入口；[主要结果](../findings.md) · [操作规则](lessons.md) · [目录导航](README.md)。
历史记录见[整理前原文](archive/20261007-doc-cleanup/files/tasks/todo.md)。下列实验状态来自整理前最后记录，本轮仅整理文档，未重新检查后台进程。

## 当前待办

- [ ] **RE 全量标注收尾**：等待三个 API 分片结束，合并旧成功与新结果；核验 event/pair 全量覆盖、互斥性、源绑定、输出协议和失败清单。旧成功 1905 events，剩余 11181 分成三份各 3727；总量 13086 events / 34531 pairs。每项首次请求后最多重试 5 次，鉴权/余额错误立即停止。不得把结构验收当作语义准确率，未启动 Qwen 训练。入口：[分片目录](scievents-re-20261006/runs/full-v6_1-three-api-20261007/)。
- [ ] **Seed52 补充 test 验收**：核验 cp2205 → 1804 → 3208 各 1627 句 / 250 摘要、六项官方 P/R/F1、complex 与四种评分口径、源码及来源锁，并与 dev-best2406 并列。保留原 dev 选模，不按 test 改 best。入口：[补测目录](stage2-event-tree/analysis/cot1-seed52-extra-test-20261007-155617/)，tmux `ee-cot1-seed52-test-20261007-155617`。

## 暂停与历史遗留

这些项目不自动排队；恢复时先核对对应规格与实际产物。

| 项目 | 保留边界与入口 |
| --- | --- |
| Stage-3 refinement | D2 OOM 后重跑已停止，无有效完成结果；R1/R0/R01、reward scorer 与完整验收仍待处理。[说明](stage3-tree-refinement/README.md) |
| Nugget 结构 E1 → E2 | E1 已完成，原节点高覆盖假设未成立；E2 未启动。[复盘](nugget-structure-diagnostic/runs/e1-v1/review/report.md) |
| AMR anchor-local P2 | P1 静态组过闸，上下文化组未过；停止于 P1，不启动 P2。[报告](stage1-syntax-amr-local/report.md) |
| 旧项目总览 / Type-Skeleton | 旧清单存在未关闭项；恢复前核对是否已被后续 warm-up/Type-CoT 实验覆盖。[原清单](archive/20261007-doc-cleanup/files/tasks/todo.md) |
| 旧同 seed 提速重跑 | 已由 seed52 原生配置复现替代；benchmark 留存，不再待执行。[计划](type-cot-experiment/type-cot-fast-20261006/plan.md) |
| 提醒尝试 | 原 10 月 3 日提醒未创建，已过期，不作为当前待办。 |

## 主要结论

- 主线为 Qwen3-8B 句级 event forest：完整摘要 + 目标句，逐句独立生成后合并；train/dev/test = 12828/1593/1627。
- Sorted1 与在线 dev 尚未证明稳定收益；CoT→树 seed42 的收益未在 seed52 复现。完整数值与证据见[结果索引](../findings.md)。
- F.2 权威定义见[精确参考](../task/scievents_appendix_f2_exact_reference_v2.md)。任务分解讨论与课程难度讨论分开；不自动增加 subevent 阶段。

## 本轮精简计划与复盘

- [x] 检查入口与教训，归档六份原文并记录 SHA-256。
- [x] 校验范围：只精简管理文档，保留当前待办、关键约束、结果及历史入口。
- [x] 合并重复任务与教训，移出逐次轮询；旧计划/进度改为导航。
- [x] 核对本地链接、归档 hash、关键事项及 diff。

整理原则：后续待办仅记录下一动作及结果入口；详细过程写入实验目录，完成后归入结果索引，避免逐轮追加到本文件。

复盘：todo 1366→39 行，lessons 167→34 行，findings 122→40 行；根计划/进度各压为 5 行导航。六份原文 SHA-256 一致，48 个本地链接均有效，当前待办与停止边界检查通过，文档 diff 无空白错误。代码、数据、日志和实验报告未改动。


### Git archive completed — 2026-10-07

- [x] Scoped local commit `75c397b`: online-dev training, CoT prompts, seed 42/52 code, tests, experiment metadata and results.
- [x] Independent scope review; 91 CPU tests and 21 subtests passed; staged whitespace check passed.
- [x] Verified commit created and staging empty. Unrelated workspace changes preserved; no push. Model weights, raw datasets and live mutable evaluation outputs excluded.

## Gold 转换损失与 argument 顺序审计（2026-10-07）

规格：只做 CPU gold 诊断，复用当前固定 parser/aligner/merge 与官方评分；完整 test，输出新目录，保留旧审计和真实预测。分别核查 argument 列表顺序与片段内部 token/offset 顺序，比较去重开关，不修模型或评分器。

- [x] 核对当前链路及已有审计；独立复核 argument 顺序实现。
- [x] 校验计划：完整覆盖/来源 hash/原文不变，差异逐条落盘；gold 自评分与转换后评分分开。
- [x] tmux 重跑完整 gold 审计，并核查 argument 顺序敏感性与默认去重。
- [x] 核对产物及差异原因，记录复盘与简明结论。

复盘：[完整审计](stage2-event-tree/analysis/gold-roundtrip-order-20261007/README.md) exit0、输入/源码hash不变；默认链路gold AC95.5305%，原gold自评98.6356%。173论元offset序列失配（69仅顺序/重复、104位置不同），3trigger失配。argument列表逆序不改变回填内容，AC因重复span覆盖变化0.0091pp；未改任何真实预测或训练。

## 独立树评分入口（2026-10-07）

规格/计划校验：新增 scripts/score_event_trees.py，显式接收 gold JSON、句级数据 JSONL、原始树预测 JSONL；同时计算官方 raw-token 与官方 offset 六项P/R/F1。复用已核对的原版utils、固定解析/合并及raw-token转换，默认去重策略与正式链路一致。拒绝重复/缺失ID和来源错配，不改历史文件；JSON输出来源hash、解析/对齐诊断及分数。原版重复span覆盖行为保留，不引入新指标或声称补偿损失。已有内部函数功能齐全，所缺是通用CLI；复用避免重写公式。实现置信度95%：本地官方源码/真实调用链已核对，现有依赖足够。

- [x] 核对原版逻辑与既有内部实现，明确CLI输入/输出和验证范围。
- [x] 先建立缺失/重复ID、类型/角色计分和文本/位置区分的失败测试。
- [x] 实现独立脚本，通过测试。
- [x] tmux对完整seed42 cp1804评分，与保存18项offset及raw-token指标逐项一致；补充使用方式和复盘。

复盘：scripts/score_event_trees.py独立入口已完成；7项测试通过、独立代码审阅通过；tmux完整test exit0，两种口径共36项指标与cp1804历史记录一致，最终结果来源hash全部核对一致。[使用说明与结果](stage2-event-tree/analysis/standalone-score-20261007/README.md)。未改官方公式或旧实验。

实际预测案例复核：[三个token/offset案例](stage2-event-tree/analysis/token-examples-20261007/README.md)。文本同文异位置、训练目标词序改变、重复show覆盖均有实际cp1804证据；两show子集offsetTC100%/tokenTC50%新复算通过，输入hash不变。未修改评分器或预测。

Token低分核查：[cp1804全量诊断](stage2-event-tree/analysis/token-examples-20261007/cp1804-token-diagnostics.json)已复算，hash不变；官方AC41.9066%、实例AC42.8843%，训练表示一致诊断42.9533%。最后一种改变比较目标，仅作诊断；训练构造问题按用户要求后续处理。

## 树文本一对一评分实现（2026-10-07）

已确认规格：预测原始树与原gold分别取type/trigger文本/argument文本/role，保留重复实例，不回填offset、不改gold词序或历史预测。TI按trigger多重集；TC按type+trigger多重集；同TC组内事件一对一最大匹配（优先AC，再AI，再EC_C/EC_I）；论元按文本/role多重集计分。EC沿用论元文本同时作为事件trigger的定义，不声称恢复位置或真实边实例ID。输出P/R/F1、TP/FP/FN、事件配对和来源hash。复用已有经过验证的instance_scorer.py，新增独立token-only CLI，不再通过官方字典计算主分数。

置信检查95%：已有一对一实现可复用、原版公式及数据结构已核对、无需新依赖；验证需覆盖同文本不同type、重复event/argument不能复用、顺序不敏感及完美gold自评。遵循已使用的brainstorming/Confidence Check/TDD流程，用户已确认设计并授权实现。

- [x] 确认一对一评分规格与边界，登记执行/验证计划。
- [x] 新入口测试先失败，覆盖重复实例与顺序。
- [x] 实现token-only脚本，测试通过并独立review。
- [x] tmux完整test复算并核对计数/来源，输出新旧结果对比与复盘。

复盘：[树文本一对一评分](stage2-event-tree/analysis/tree-token-matching-20261007/README.md)新增token-only CLI，9项测试及全量核验通过；4CP tmux exit0，gold自评18项100%，72项与独立实例记录一致；输入/源码hash和顺序不变性通过。1804 AC42.8843%、2205 AC43.1852%；原gold/历史预测/训练均保留。

## 用户指定层级文本存在匹配（2026-10-07）

最新规格覆盖一对一主评分要求：TI匹配trigger；TC匹配trigger+type；AI匹配trigger+type+argument；AC再加role。按同文档对侧是否存在完整组合计分，允许多个同文实例命中同一表达。召回按gold侧命中比例，精确率按预测侧命中比例，额外错误预测仍计FP；不做字典覆盖，不把新规则伪称所有baseline的既有实现。EC保持论元文本也是trigger的文本连接定义。

- [x] 确认新规则，保留前一版代码和结果。
- [x] TDD覆盖允许复用、逐级条件、预测误报和重复文本不同类型。
- [x] 新独立脚本及复核通过。
- [x] tmux重算四CP，输出完整指标与复盘。

复盘：[层级文本存在匹配](stage2-event-tree/analysis/tree-presence-matching-20261007/README.md)完成，取消一对一消耗，7tests/review通过；四CP及普通baseline完整test重算exit0，gold自评18项100%，双侧计数公式/来源hash/独立嵌套搜索及顺序不变性通过。1804 TI76.3600/TC64.3758/AC43.3750；2205AC43.5900，baselineAC42.9252。同规则比较，旧评测结果留存。

## 2026-10-07 SSE中断重试修复与续跑

偏差及重新规划：三路均因HTTP200 SSE内upstream_stream_read_error误触发fatal停止；实际是可重试上游传输中断。已成功1905+649+3190+1512=7256，三组累计失败15，其中3个fatal中断，尚未处理5815。不重跑成功，不重置6次请求预算。隔离新runner，仅精确匹配结构化error.code=upstream_stream_read_error允许重试，未知/账户错误仍停；测试覆盖第6次恢复、6次耗尽、非瞬态仍停、迁移已用次数保留。旧目录完整保留，新目录复制归档与states，仅将可证实的3个旧fatal改为非终止error并移除对应失败state，保留原始raw/status200及迁移说明。冻结新配置并开3个tmux继续原分片，最后合并与覆盖验证。

- [x] 修复瞬态SSE错误分类并通过回归测试。
- [x] 保留旧结果、迁移归档与请求计数，验证成功记录一致。
- [x] 启动并核验三路续跑。
- [x] 最终合并验证（全量结束，失败单独保留）。

续跑结果：58tests通过，3条真实fatal原始响应均重现并精确识别upstream_stream_read_error。原目录不改、迁移7256成功记录保持源绑定且字节一致；3条中断已用attempt保留并重新开放，其余12条已耗尽失败保留。新目录runs/full-v6_1-three-api-streamfix-20261007，新tmux ee-re-api1/2/3-resume-20261007均存活，三个原中断事件后续请求均已归档，其中api2已成功。全量仍未完成，后续自动合并核验。

2026-10-08最终核验：三个续跑tmux均正常exit0，每片3727已处理并结构验证通过。原1905成功与三片成功合并为13008events/34269pairs；78失败单独保存。再次核对合并ID唯一且与全部13086源事件精确一致；覆盖/源绑定/响应结构已验证，语义准确率未测。结果runs/full-v6_1-three-api-streamfix-20261007/merged，不超预算自动重试78失败，未启动训练。

## 2026-10-08 78个失败event重跑

计划校验：用户明确要求再次重跑失败，授权独立新请求预算首次+最多5次重试。仅选最终78失败ID，三份26event分别原三个API/2workers，同v6.1无类型模板及已修复SSE重试runner。13008成功记录原样保存且不请求；新目录不改历史结果。先验证源manifest、冻结版本与互斥完整分组，启动3tmux核验请求与状态；结束后复用已验证finalizer单片检查及合并新成功/仍失败，全13086覆盖。

- [x] 校验78失败范围、26/26/26分组、保留13008成功并冻结配置。
- [x] 启动3tmux并验证请求。
- [ ] 结束后验证及合并（自动执行）。

78失败补跑启动记录：三个tmux ee-re-failed1/2/3-20261008均存活且真实请求已归档，结果runs/failed-rerun-20261008；等待运行结束及自动合并，不宣称全量失败已解决。

## 2026-10-08 RE-CoT训练prompt草稿

计划校验：参考当前Type-CoT user-final-20261005模板的任务、语义定义、简短解释、最终答案结构，另写英文RE训练模板。保持现行12标签/方向、无nugget type输入、无confidence/status、候选完整性和offset规则。新增可监督的逐对证据/端点/方向解释，最终JSON与现有RE结果兼容；模板≤5000字符。不覆盖标注prompt，不改数据，不自动生成CoT或训练。

- [x] 核对实际Type-CoT配置/模板与RE定义。
- [x] 写独立英文RE-CoT模板。
- [x] 核验标签/方向/占位符/长度及训练兼容性说明。

RE-CoT草稿复盘：独立模板4691字符，12标签及3占位符检查通过，先解释逐对证据/端点/方向，再输出JSON。说明既有纯JSON标注需补解释方可用于CoT训练；未改标注模板/数据或接入训练。文件prompts/re_cot_training_en_v1.txt及同名notes/check。

## 2026-10-08 RE训练prompt去解释输出

用户纠正：目标是学习关系识别、标签和方向，直接监督现有关系JSON，不生成reasoning。保持12个关系定义/方向/例子逐字不变，删除解释生成任务、Reasoning/Final Answer包装及RE-CoT提示；保留文本证据/端点检查作为判断要求。规范名称re_training_en_v1.txt，旧草稿路径同步内容并记录历史。检查长度、占位符、定义未变、无解释输出及JSON格式；不启动训练或改标注数据。

- [x] 修改prompt及配套说明。
- [x] 核验定义不变、纯JSON输出与长度。

结果：re_training_en_v1.txt共4313字符，关系定义区逐字不变、12标签/3占位符及纯JSON输出检查通过；旧草稿完整归档，旧路径同步纯JSON。仅修改prompt文档，不生成reasoning、不改标注、不启动训练。

## 2026-10-08 编写新RE训练说明

计划校验：用户指定task/1008 新训练说明.md，记录RE目标、真实候选/标注/训练prompt路径、当前完整数据与补跑状态、训练样本构建、生成式SFT及验收步骤。说明旧导出仍带类型/offset，不把历史小样本SFT或计划产物写成当前全量数据。只写说明及证据，不修改运行/标注/训练策略。

- [x] 核对数据统计、prompt与导出代码现状。
- [x] 写具体流程、路径与格式实例。
- [x] 检查链接/数据计数/最新约束，确认文档落盘。

说明交付：task/1008 新训练说明.md已写，全部本地链接存在，完整基线13008/34269及补跑新增56/187和真实示例逐项复核。明确新版4291字符prompt、纯JSON/无type/无offset输出、旧导出不兼容、全量SFT待构建及训练调度尚未冻结；没有改标注数据或启动训练。

## 2026-10-08 补充生成树评测说明

计划校验：在用户指定的新训练说明中补充当前文本存在匹配与官方token入口、输入格式、可执行命令、结果读取及两者区别；只改文档，保留现有RE说明。

- [x] 核对实际入口、参数与结果字段。
- [x] 补充生成树评测流程和命令。
- [x] 核验文档路径、命令与交叉审阅。

结果复盘：保留原RE说明，新增第9节，区分最近使用的presence脚本与官方token入口；写明输入、流程、五项指标、完整命令、JSON读取和覆盖限制。21个本地链接及命令输入路径存在，两个CLI的help核验通过，读取结果的Python语法通过；只读交叉审阅后补充官方重复论元顺序敏感及EC仅检查trigger文本集合的说明。未改评分代码、启动训练或生成新实验结果。


### 2026-10-08 最近 CoT→事件树训练说明

- [x] 核对已有文档、两个 seed 的配置、阶段衔接、课程采样与反馈代码。
- [x] 保留 RE 及评分章节，新增已完成训练的详细说明与绝对路径入口。
- [x] 核验路径和关键参数，独立复核新增章节。

复盘：在 `task/1008 新训练说明.md` 新增第10节，保留原RE与评测章节；核对27个绝对路径链接、两run实际产物、预算/选模/停止原因。独立只读复核通过，修正衔接证据文件名为 `handoff_evidence.json`。本次仅修改说明及任务记录，不启动训练或修改历史run。

## 2026-10-08 RE矛盾监督核查

用户提供重复曝光/冲突统计，要求审阅。以原完整13008event/34269pair成功文件固定范围，独立复算NONE、source occurrence重复、同context标签/方向及最新无type/无offset全输入冲突，区分对称关系端点换序与真方向冲突。保存可复核case及口径；不自动裁决或修改标注。

- [x] 复算统计并核查相同输入案例。
- [x] 区分监督矛盾与合理重复，给出训练前处理建议。

独立复算确认用户全部数字：NONE21530/62.83%、不限定context重复额外412、同context标签/方向冲突68（对称正规化后仍68）、完全相同无类型/无offset输入冲突6组12events。冲突涉及107events，12包含其中；建议清洁训练版先整event隔离/统一裁决，不自动多数投票或只删答案。证据runs/supervision-conflict-audit-20261008/audit.json与report.md，原数据未修改。

## 2026-10-08 107冲突event复核与清洁数据

计划校验：用户授权处理107events/68冲突pair。固定原13008成功事件，不覆盖源。三组独立AI语义复核68对，依据当前RE定义和上下文/精确端点，不把源nugget/event type当答案；主代理交叉核查裁决及方向。确定的冲突在所有对应event中一致更新；不能可靠裁决的整event隔离。保存逐对依据、旧新标签和source定位，清洁版验源绑定/候选全集/同输入矛盾清零，并核对未涉及记录不变。AI复核不称人工gold，不修改运行中的API脚本或自动训练。

- [x] 逐对复核68冲突及方向，交叉核查裁决。
- [x] 独立生成107事件处理结果、清洁全量及隔离队列/变更记录。
- [x] 验证冲突清零、结构和非目标记录不变，更新训练说明入口。

107事件处理完成：68pair三组独立AI审阅并交叉核查，47裁决/21争议；64event保留、43整event隔离，实际生效42条关系修改。清洁标签池12965events/33914pairs；same-occurrence/context冲突68→0、同全输入冲突6→0。63tests通过及独立重新统计passed；原hash未变、12901非目标事件深度逐字段不变。输出data/clean-annotations-107-v1-20261008；报告runs/conflict-resolution-107-20261008/report.md。训练说明第11节及顶部入口已更新，原数据/隔离原标签保留，未改prompt、未启动训练，新增失败补跑标签未合并。

2026-10-09自定义重评复盘：26评分、11均值及统一评分源码hash全部通过；独立复核计数换算P/R/F1、原dev-best、dev13+9覆盖及5表逐列加粗通过。输出tasks/type-cot-experiment/analysis/presence-re-comparison-20261009/report.md、all-metrics.csv与verification.json。RE−noRE平均test AC+0.0211pp、EC−2.0917pp；种子组不同且gold监督修复，非严格RE因果对照。无新训练/推理、未改变选模。

2026-10-09预备代码复盘：sixcp runner8、串行计划3、noRE22，共33项CPU测试通过，独立双路复核通过。最终冻结计划dry-run只检查且四个新运行目录均不存在，GPU仍155MiB/0%；未创建tmux、未推理/训练。修正review指出原训练DATA/GOLD来源绑定，新增回归先失败后通过。入口planned-goldorder-noRE-20261009/README.md，manifest/日志/验收证据在该目录；后续授权启动前仍需独立GPU smoke。

2026-10-09启动复盘：最新两seed计划11项回归通过，独立review无58；GPU smoke全流程exit0、finite loss/阶段重置/14句重载输出完全一致、sourcehash通过。正式tmux ee-retest-nore42-66-20261009运行，六cp第一项42cp2406已生成首批test，后续自动noRE42/66。seed58没有run，完整训练/评测仍待结束；日志execution-42-66/及补测analysis/re-extra-test-20261009/。预计北京时间10月11日09:00–12:30。

2026-10-10用户撤回noRE66启动授权：只等待当前seed42完成。先SIGSTOP仅队列调度PID2394103，seed42/树训练不发信号，确保66不会启动；取消记录execution-42-66/cancel-pending-seed66.json。待42训练/评测/验收结束后仅清理调度进程并归档结果；58继续不跑。

2026-10-10最终状态核验：六RE checkpoint补测已完成；修复词序noRE42已于北京时间17:22完成，cot802/tree3208、devbest2807、test250/1627，训练与最终验收文件存在且通过；修订范围scope-exit0。noRE66/58均无run。项目presence F1：TI76.5847/TC64.4041/AI44.9135/AC42.5109/EC48.0137。用户取消待启动66后仅清理队列调度，raw dispatcher main.exit可能为-9，不代表当前seed42训练失败；最终状态以scope-exit/seed42-only-result和run为准。

## 2026-10-10 修复词序noRE42全部dev自定义评价

计划：复用13个已保存dev预测，用同一score_tree_presence.py重评251摘要/1593句，核查来源hash/覆盖/有限值并从计数复算P/R/F1，展示TI/TC/AI/AC/EC（EC_C）与逐列最佳；不推理/训练、不改原官方dev-best2807。
- [x] 完成13个CPU评分、验证和结果表归档。13点评价覆盖251/1593，所有来源hash及按计数换算PRF通过；report/summary/all-metrics.csv在analysis/noRE42-goldorder-dev-presence-20261010。TC/AC/EC最佳2807，TI1604，AI1804；原dev-best未更改。

## 2026-10-10 noRE42 cp1604/1804补测test

用户授权修复gold-order noRE42 cp1604/1804全量test，自定义presence TI/TC/AI/AC/EC；不改变dev-best2807。复用已验证补测入口独立配置，先check原训练hash/权重/步数/命令，再tmux串行生成+验收+评分，记录退出码。
- [x] preflight验证并启动tmux首批预测。tmux ee-nore42-test1604-1804-20261010运行，cp1604首批已生成；预计北京时间19:40–19:50完成两cp。
- [x] 两个cp完整test结束、hash/覆盖/评分验收与结果展示。北京时间19:38结束exit0，均250/1627、来源hash复核通过；cp1604 AC42.9888/EC47.9653，cp1804 AC43.1041/EC49.4406（presence F1%）。原devbest2807保留。


## 2026-10-12 面向 Pro 的最佳方法介绍

规格：输出独立简体中文 Markdown，可直接交给 Pro 讨论方法改进。围绕10月7日周报的最好主线，不罗列历史失败变体；解释实际输入/输出、Type-CoT教师与学生区别、树生成、课程与dev；附3–4个正式训练数据真实CoT案例及来源。结果保留seed/checkpoint与评分口径，不能把test补测点写成dev-best，不把未证实RE写成已获收益。

计划校验：只读实验和训练数据，新增介绍文档，不训练、不推理、不修改实验。两项独立只读探索已分派；交叉核对原始JSON后归档。

- [x] 核对周报对应run、参数、课程公式与评分结果。
- [x] 从正式Type-CoT数据选取可追溯案例。
- [x] 撰写task/1012_给Pro的方法介绍.md。
- [x] 核查文档数值、例子原文、链接、需求覆盖，并记录复盘。

结果复盘：已生成 [给Pro的方法介绍](../task/1012_给Pro的方法介绍.md)，围绕周报主线，包含真实训练CoT四例（单事件、多类型、同类重复、None）。新鲜脚本核验四条completion逐字一致、学生prompt无gold区块、周报五项数值及原始指标精度、全部本地链接、真实PRP子树和Markdown代码围栏通过；独立事实复核无阻塞问题。明确1804为补测点、2205为原dev-best、自定义文本存在匹配与论文报告口径有区别。未修改代码/实验产物，未启动训练、推理或API。


## 2026-10-12 当前主线代码 review

用户选择：当前CoT→事件树训练、数据构造和评分主线，包括未提交代码。范围以HEAD 75c397b为追踪基点，审查当前工作区实现；不是只看git diff。规格依据主线config/manifest、task/1008 新训练说明.md第10节与词序修复规格。代码只读，不修复、不训练、不启动推理、不改冻结实验。

- [x] 固定范围与规格，平行审查规范/方法实现。
- [x] 审查数据构造、树解析/对齐和当前文本评分，核查真实数据边界。
- [x] 运行相关CPU测试与最小复现，排除误报。
- [x] 输出可定位的问题、优先级、影响与验证结果，归档复盘。

结果复盘：[审查报告](../task/1012_当前主线代码审查.md)。Standards确认1项P2：模型来源锁漏5个实际权重分片与tokenizer/生成配置；Spec未发现确认缺陷。三个独立只读审查轴交叉核查完成，主代理直接CPU测试92项+27子测试通过，独立spec额外plan测试通过。另从实际RE42 cp2205预测复现正确Aim因非单调gold offsets被对齐丢弃，列为已记录的转换局限而非新bug。未修代码、未修改冻结实验、未GPU训练/推理。


## 2026-10-12 修复模型完整来源锁并Git提交

规格与设计：用户授权修复审查P2并commit。修改当前type-cot-goldorder/config.py，按safetensors index解析全部实际分片；锁config/tokenizer_config/index与实际tokenizer.json/vocab/merges/特殊token/生成配置，不把旧manifest补写成历史已经锁过。新manifest完整校验，旧弱锁manifest由新版runner拒绝，历史run/source_snapshot/manifest均保留，后续需重新生成独立manifest。SHA256流式读取，不缓存文件内容hash，避免替换绕过检查。不训练/推理/运行旧队列，不提交其他已有改动和实验数据。

计划校验与置信度：根因已独立证实；复用现有sha/verify_locks，标准库json/pathlib，无新依赖；完整源文件清单源于本地transformers/AutoTokenizer加载器及权重index，未找到已有完整锁实现；95%。已有review与本次用户授权覆盖方案，无需再次确认。

- [x] 使用小型本地模型文件fixture写权重/tokenizer/生成配置替换失败回归，确认先红。
- [x] 实现完整模型文件枚举与严格锁校验，补文档和教训。
- [x] 跑CPU回归、实际5分片完整hash覆盖验证，独立review。
- [x] 核查暂存范围，仅提交此次修复相关文件，记录commit及复盘。

验证复盘：新增12项回归在旧实现全部失败，修复后12项通过；完整gold-order CPU套件34项通过（117.26秒），独立review无阻塞。真实Qwen枚举12项输入，覆盖全部5个分片（总16,397,431,693字节）；完整回归构造及验证实际模型manifest通过。此前记录的5份历史manifest/计划/源码快照SHA256复核未变。仅当前config、新回归、迁移说明及本次任务/教训追加进入提交范围，其他已有dirty代码与实验产物保留。

提交复盘：Git commit `77dd0ac`，5个文件；仅追加本次任务/教训记录，既有其他dirty变更未提交。暂存区已清空。

## 2026-10-12 精简复制主线到 EXCEEDS1 并审查 Action/GRPO 方案

规格：保留原项目；复制当前主线代码、原始与正式训练数据、prompt、已完成 seed42 CoT adapter 及来源证据。保持相对目录以复用代码；共享 /opt/models/Qwen3-8B 和既有 conda 环境。省略历史 runs、缓存、日志、.git、API 凭据。两份新方案只审查，不启动其训练。历史来源文件原样归档；新副本运行时 correction manifest 只迁移输出路径，不修改训练数据或旧证据。

计划校验：目标目录为空；两项独立只读探索已完成。使用 SHA256 核对复制前后，验证新目录输入审计与 CPU 评分测试，说明未验证的 GPU 训练及历史回归依赖。

- [ ] 复制代码、数据、prompt、seed42 adapter 与来源证据。
- [ ] 迁移副本运行时输入锁路径，生成复制清单与入口 README。
- [ ] 写方案审查文档，列出评分、对齐、reference、对照验收问题。
- [ ] 执行哈希核对、输入审计和 CPU 回归，归档结果。

2026-10-12 用户撤回迁移：停止后续迁移操作，保留已复制内容。原目录未移动或删除；此前已复制163项约708MB，后台CPU验证只读，不属于训练。迁移任务取消，未将其标记完成。

## 2026-10-12 仅审查 Action SFT / GRPO 方案

范围：按最新指令，仅审查 task/101201_codex_action_rl_plan.md 与 task/101202_action_prompt_v1.txt。复核实际 scorer、aligner 与 LoRA reference 契约，不更改两份原稿，不启动训练。

- [x] 对照当前实现与新方案文法、数据、奖励、训练和选模约束。
- [x] 独立审查并复现重复触发评分反例，核查未定位预测丢弃路径。
- [x] 归档 task/1012_Action_GRPO方案审查.md，明确需要先补充的验收。

结果复盘：主要风险为官方重复触发覆盖导致奖励失真、aligner丢未定位预测、奖励不包含论元、默认PEFT reference不等于SFT LoRA模型。补充公平选模/预算、prompt路径和解析边界建议。本次只审查，没有修改训练代码或启动实验。

## 2026-10-12 复核文本存在匹配与 LoRA reference 解释

规格：按用户指定检查 score_tree_presence.py 能否解决同触发不同类型 TC 覆盖问题，运行已有测试和三组对照反例；核对旧 CoT adapter 已保存事实。论元奖励暂不扩展。不启动训练。

- [ ] 核查 presence 签名及重复匹配语义，运行测试和反例。
- [ ] 核查 adapter 保存记录，澄清 base+LoRA 的 reference 冻结方式。
- [ ] 更新审查结论与教训，报告范围和限制。

验证复盘：上述三项完成。presence现有7项测试通过；直接官方/presence对照复现同触发异类型TC从0.5变1，重复同签名增减TC仍1符合存在匹配契约。核查正式seed42 CoT adapter存在及802步完成记录；追加审查结论，论元奖励按用户指令延期。未修改评分代码、原方案/prompt或启动训练。

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
- [x] 新tmux串行队列启动，检查真实首批loss。
- [x] 核对git提交范围，提交代码/依赖/配置与说明，记录commit。

启动前复盘：新短prompt数据完整生成，逐条仅prompt变化；三项全量审计通过，70项CPU回归通过，完整preflight包含sampling.py来源锁并再次逐文件hash确认一致。2026-10-12 23:24:54 +08:00已启动tmux ee-action-v2-20261012，SFT fresh CoT接续，完成后完整GRPO preflight再继续RL。新run路径action-sft-v2-42-20261012/action-grpo-v2-42-20261012；检查首批更新后记录最终启动状态。git采用显式范围，只存代码/必要依赖/prompt与小型证据，其他dirty内容和大数据保留工作区。

完成复盘：新tmux实际更新已验证，首批loss [0.3986, 0.5901, 0.4467, 0.3527] 均有限；正式SFT仍运行，RL排队未开始。Git提交234a36e，38个文件，范围复核与cached whitespace检查通过；工作区既有其他dirty内容保留，暂存区为空。没有推送，也未声称完整训练完成。源码已冻结，后续仅读取日志与记录运行状态。

## 2026-10-13 SFT结束即停，RL工程与诊断改进

用户撤销自动RL启动：保留当前SFT完成/早停，取消后续GRPO派发，不自动启动新版RL。launcher收到SIGTERM（仅父shell），按现有TERM trap在前台SFT结束后退出；模型进程不发信号。记录stop-after-sft.json，验证SFT日志继续而无RL进程。当前约2441/3208步，best1604 TC0.628921，bad_count4，LR5e-5，若后续无改善2807早停。

实施规格：在rl_v2独立子目录开发，不改SFT来源锁覆盖的主目录py/tests；旧baseline可复现。G4默认一次batch generate，可明确配置microbatch2而保持G4，不静默降G或mask动作；按每行EOS/PAD独立mask，保留原采样token和真实scores行为概率；teacher/reference与backward逐样本执行控制显存。schema已LRU缓存、presence evaluator是全局单例，startup一次初始化并记录缓存次数。奖励仍0.6TC+0.3TI+0.1Count，Event仅作逐事件诊断，包含precision/recall/F1、类型/触发错、missing gold，不写入训练loss。保留SFT reference与dev/test隔离。增加生成/更新时间、响应长度/PAD、活跃组、奖励分量/截断诊断。

设计校验：用户给出具体实现规格覆盖本次设计授权；独立目录满足冻结约束，复用现有loss/模型/评分，无新依赖，置信度95%。不声称batch比串行快，GPU正在跑SFT，只能先CPU真实tiny验证，Qwen batch4/768显存和吞吐需SFT完成后的单独验证，正式RL保持不启动。

- [x] 查询SFT状态，取消后续派发，保留SFT继续。
- [x] 独立G4批量采样模块与EOS/PAD/概率回归。
- [x] 事件级诊断与缓存复核、错误类型反例验证。
- [x] 新runner/来源快照/训练诊断集成，CPU真实tiny G4更新、reference与奖励一致性校验。
- [x] 独立review、文档与当前SFT状态归档；不启动RL。

本次复盘：新版独立入口rl_v2/train.py已实现，CPU全套88项通过；真实tiny双LoRA优化验证和PAD裁剪loss/梯度对照通过，独立review无阻断问题。SFT源码锁66项无变更，当前约2490步继续更新；取消的是后续dispatcher，不中断SFT，未创建RL run。Qwen batch4/768显存与吞吐仍待GPU释放后验证，尚无速度提升倍数证据，不自动启动正式RL。新增文件未git提交。

2026-10-13后续安排：用户希望SFT结束、测试后继续新版RL。当前2807完整dev已开始，188/1593；最近三次dev耗时27.5/30.3/29.8分钟，预计本次余25–30分钟。test1627行相近，best-adapter完整test预计30–40分钟（含模型加载与验收）。用户反馈早停6次过长，下一轮应缩短并保留降LR恢复窗口；当前源码不改。后续顺序为dev结束/SFT完成→所选best的test（只报告，不用于调参）→Qwen新版batch4显存与更新验证→通过后继续RL。

SFT已完成：2807步早停，bad_count6，best1604 TC0.628921/TI0.758855，最终devTC0.619392，run status completed。launcher exit143是取消自动RL的预期退出，非SFT错误。已新建tmux ee-action-test-v2-20261013启动best checkpoint完整test，输出test-best-20261013，日志launch-test-v2-20261013/test.log。后续RL仍需真实Qwen批量显存/优化更新验证后衔接。

## 2026-10-13 test完成后衔接新版RL

规格：test只报告、不用来调reward或选模；init采用SFT dev选择的1604 best。沿用原GRPO算法奖励，新入口默认G4 batch4，逐事件只诊断。先tmux真实Qwen最长train prompt强制768token batch采样和一次受控奖励更新验证EOS/PAD/行为概率/有限梯度/reference不变，不能将该smoke作为正式结果。通过后启动正式rl_v2/train.py并检查真实初始化/初始dev，无静默OOM回退。若实测OOM则重新规划显式microbatch2验证，保持G4。

- [x] test exit0，1627条，TC0.631536/TI0.768093/AC0.427381，耗时1850秒。
- [x] 真实Qwen batch4/768显存与更新验证，原SFT权重/reference不变。
- [ ] 验证通过后tmux正式RL启动，确认日志和来源冻结，归档。

用户随后要求“等一下”、先展示全部test结果：正式RL启动暂缓。真实Qwen smoke已独立完成exit0：最长train prompt1527tokens，G4一次生成强制768token，生成34.46秒，peak20,143,027,712 bytes，受控奖励一次更新有限/reference及输入SFT权重不变；该验证不是正式训练且不证明实际EOS/PAD混合采样（CPU已有相关回归）。尚未启动正式RL，等待用户后续继续指令。

2026-10-13用户明确授权开RL：已启动tmux ee-action-rl-batch-20261013调度器，init1604 best，正式入口rl_v2/train.py，G4一次batch，gradacc4，dev每200更新，奖励原公式/Event仅诊断。当前另用户T5进程占约9GB显存，排队等待GPU0空闲显存>=22000MiB才加载模型，不终止其他进程、不降低batch配置。正式训练是否进入rl阶段以launch-rl-batch-20261013/status.log为准，不能将排队描述为已开始更新。

## 2026-10-13 RL排队显存竞争修复

发现第一次正式加载exit1：单次nvidia-smi显存空闲采样发生在其他用户实验间隙，随后T5加载9.16GiB，与Qwen同时占用导致base.to(cuda) OOM；尚无group/update，非batch4算法显存验证失败。计划：保留原partial run与日志；新dispatcher要求连续180秒free>=22000MiB，缩小启动间隙风险；仍可能外部竞争，若仅模型加载阶段OOM且无dev/group/update则保留独立attempt并重新等候；训练已开始后不自动重跑以免重复样本。验证shell语法、tmux存活与waiting阶段，不声称已更新；不修改冻结训练代码、不终止他人进程。

用户催促直接挂tmux：已挂ee-action-rl-batch-retry-20261013，shell语法验证及session存活通过。连续180秒free>=22000MiB后启动，每次尝试独立output/log；仅模型加载OOM且无dev/groups/updates时重新等待，训练开始后的错误不会静默重跑。现阶段waiting_gpu，未宣称训练更新已开始。

## 2026-10-13 按1013codex_rl_only_final_for_codex实施R0

旧RL已SIGTERM停止并确认PID消失/GPU释放，不恢复旧optimizer。用户附件是本轮唯一规格：独立action_rewards.py；修改主train_grpo.py接线/概率诊断/R0日志；原eval、presence、action_language、grpo_core、action_common、SFT/prompt/data保持原样。回到原主GRPO串行G4采样，不强制延续独立batch提速实验；loss/clip/KL/advantage/dev频率不改。SFT→RL来源校验全部保留，新run默认源码扫描纳入奖励与诊断py/tests。

R0设计：严格解析，事件按(type,ordered trigger token tuple)分组；同key矩形最大权指派确保min(pred,gold)事件配对数，权重为父事件内部(role,ordered argument token tuple)Counter交集数量；实例/重复不去重、不跨父事件借论元。FE/FL边界严格按附件，R0=.5FE+.5FL。仅日志记录结构差异数与EOS/truncation，不额外惩罚。复用已安装SciPy1.10.1 linear_sum_assignment(maximize=True)，官方rectangular接口已核实，confidence95%；实现按附件已授权设计与TDD，不再另求设计确认。

- [x] 停止旧RL，保存原run，读取规格/历史教训并冻结不可变文件hash。
- [x] R0先红后绿单元测试（重复、最优配对、父归属、空/非法/STOP无EOS）。
- [x] 主GRPO接线及R0/概率/结构差异日志，原评测器固定raw回归完全一致。
- [x] CPU集成、来源校验/新文件快照、真实GPU G4/R0全部概率路径与更新验证。
- [x] 独立review、通过后新目录tmux正式RL启动；若概率异常不放行。

R0实施验证：127 CPU passed；16048 Gold全部R0=1；原1627raw eval指标/计数完全一致；完整handoff/source preflight passed，新action_rewards/grpo_diagnostics均锁定。GPU真实16样本/4组累积一次更新 passed/exit0，current-old mean.003440/P95.016081/P99.098537/max.121199，actual ratio越界0%，eval/train及policy/reference差0，refhash不变/policy变化，norm.290468。已直接启动tmux ee-action-r0-20261013，无180秒等待，输出action-grpo-r0-42-20261013；先核对provenance和实际模型加载，再标启动核验完成。

用户新增规格：dev间隔400 optimizer steps，其他GRPO参数不变。首R0 run尚在初始dev48条、0更新时停止，保留记录；新tmux ee-action-r0-dev400-20261013启动主train_grpo.py --dev-steps400，新输出action-grpo-r0-dev400-42-20261013。不改冻结代码/已有configuration或provenance，全部来源锁仍开启；原127CPU/GPU概率验证适用于相同训练路径。dev0+400..3200+最终3207，共10次full dev，相较18次减少8次。

最终启动核验：正式dev400配置已保存且与要求一致，PID2274085在GPU占约18.3GiB，action_rewards/grpo_diagnostics都在input_sha256和source_snapshot且hash相同；6个受保护文件hash仍一致。正式模型已加载，先执行初始dev；训练/最终test尚未完成，不标完成。交付verification-r0-20261013/report.md、code.diff、source-hashes及全部验证证据。

## 2026-10-13 R0代码提交并推送GitHub

范围：本轮action_rewards/grpo_diagnostics/train_grpo及新tests、执行规格、R0验证脚本/小型证据/报告。保留其他dirty文件、独立rl_v2实验、运行日志与大数据/权重，不改正在运行的锁定py。先重跑CPU并核对staged内容/whitespace，再本地commit，push已有review/action-sft-grpo-v2-20261012分支，读取远端head确认与commit一致。无需新PR/合并。

- [x] CPU复核、复制已存在GPU/正式配置来源证据，显式范围暂存与审查。
- [x] 提交并push已有review分支，远端SHA核验。

提交复盘：1d51440f9d26fbbfcd745ee7b33cedcbc6ce876b，23文件，主线102 CPU/工作区127通过，staged whitespace通过，push review/action-sft-grpo-v2-20261012成功，ls-remote SHA与本地一致。运行RL72项Python来源锁仍一致；其他dirty保留，暂存区为空。

2026-10-14状态检查发现正式R0在group14 sample3（3次optimizer更新后）概率guard退出exit1。60条检查/3841有效tokens，单个token29 delta=.239485 ratio1.270594，使57-token响应clip外率1/57=1.754%越过默认1%。eval/traincheckpoint全零，cached/no-cache与current/old差相同；定位范围为cached生成与全序列teacher数值路径，尚未证明BF16为根因。保留formal-probability-stop.json及所有raw概率；不放宽阈值、不声称仍在训练。下一步需真实重复采样/精度与attention/cache对照，再决定修复与重新验证。

## 2026-10-14 概率保护机制修复与同参数异常重放

用户新review覆盖旧逐响应1%硬阈值：保持停止，先定位异常token/是否EOS，同参数重放区分cached/no-cache/attention/BF16；保留原token/逐token日志，改为多响应汇总，非有限、严重偏差、持续异常仍硬停。R0、Prompt、eval、grpo_core、loss clip.2、G4串行、dev400均不变。不只是把loss clipping1.2放大，不加入batch优化。

设计：旧同参数3更新状态未保存optimizer，利用原rollouts原token/old概率/advantage严格重放前12组（3次更新）重建参数并核对异常teacher与原record；异常token29 ID12526，记录字符串/EOS与数值。固定参数强制原token的cached逐步forward和full teacher对比，再math SDPA及FP32 LM-head诊断（仅probe，不生产变更）。保护默认单token abs logdiff>1或非有限硬停；保留优化clip[.8,1.2]诊断，独立保护ratio[.5,2]。按跨16回答且>=512有效tokens构建不重叠窗口，P99>.15、clip外>1%或保护ratio外>1%连续3窗则硬停；短回答单个1.27只报警。生产不改变分布/精度/attention后端。门槛与累计警告显式写配置和日志。TDD单token不误停/严重非有限立即停/连续异常多窗停/正常窗重置/掩码/窗口汇总测试。

- [x] 相同参数异常重放，定位token并记录attention/precision对照与局限。
- [x] 多响应guard先红后绿测试，主入口接入和逐token证据/配置日志。
- [x] 全量旧eval/R0回归、真实短程GPU ratio/clip/KL/gradient/reference验证及独立review。
- [x] 通过后直接tmux启动新目录dev400正式RL，无180秒空闲等待。

2026-10-14验证复盘：主线119/工作区144 CPU通过，旧R0/codec/scorer/core/prompt/common6项hash不变。历史48sample重建第一次更新前误差0、更新后不精确，明确不用于resume；live固定参数digest前后相同，token29非EOS，BF16 full/cache差.11984，FP32输出层差.00310（仅诊断，生产BF16/SDPA不变）。64sample/4update GPU passed/exit0，3675有效tokens，current-old mean.003275/P99.092955/max.24156，优化clip外2/3675=.05442%，reference不变且真实policy更新；无持续bad window。新tmux ee-action-r0-window-20261014直接启动，新输出action-grpo-r0-window-dev400-42-20261014，dev400/G4串行/lossclip.2保持。旧run只有step0checkpoint，没有可精确恢复的step3 optimizer，因此正式从原SFT最佳起点fresh optimizer启动，不能声称精确恢复旧3更新；旧日志保留。

启动核验：PID2325934实际占GPU约19GB，正式configuration确认lossclip.2、wide保护.5–2、16回答/512token/连续3窗、R0版本与dev400；新run Python來源锁全部一致，正在step0 dev。验证和交付报告verification-probability-20261014/report.md。

## 2026-10-15 按用户要求评估中断RL的dev最佳checkpoint

当前R0 run在optimizer2195/group8781触发严重logprob差1.01388退出，保持停止。完整dev至2000，按TC→TI最佳2000（TC.642928）；用户明确要求先test，覆盖之前仅最终训练结束test的时间约束，但test仍不用于调参或选模。

- [x] 核对完整dev history与best-adapter/对应checkpoint权重和配置一致，保留停止状态。
- [x] 独立只读阶段性评估入口：完整来源锁/数据/模型/选模权重校验，不伪造completed、不修改原evaluate_adapter/action_evaluation/scorer。
- [x] tmux完整test1627条，确认真实GPU进程/输出/selection-source注明stopped run；预计30–40分钟。

启动复盘：tmux ee-action-rl-test-20261015，PID3965307实际占GPU约19GB，已用step2000开始生成test。模型/数据/全部来源锁通过，best与step2000权重/配置SHA相同；独立CPU反例证明同分选较早、best权重被改会拒绝。只新建阶段性评估script，不修改冻结评测或伪造completed；最终test尚未完成。

## 2026-10-15 补测step1600：空卡立即启动

用户明确要求step1600完整test，当前GPU被占。先在tmux做完整来源/指定checkpoint核验和CPU模型准备，随后每1秒看GPU0显存，free>=21000MiB即转CUDA开始评估，不设任何连续空闲窗口或180秒等待。保留step2000 dev-selected best不变，1600标记用户指定的补充test，不据test自动重选模型。不修改原冻结脚本/评测器/训练provenance。

- [x] 新独立入口按显式step选择，核验dev记录/权重/配置及全部来源锁。
- [x] tmux先CPU准备，再1秒检查空卡立即加载GPU；确认进程、日志与等待方式。

补测启动核验：tmux ee-action-test1600-20261015，PID659008；来源校验和CPU模型加载已完成，日志CPU_READY，正在每1秒查询显存，无180秒条件。实际选择step1600、不存在1700被拒绝；best-adapter step2000不修改。开始GPU后预计30–40分钟完整test。

## 2026-10-15 补测step800并接在1600后

用户补充要求800 checkpoint test。tmux ee-action-test800-20261015先独立核验checkpoint/source并预加载CPU，依赖1600 launcher exit.code后才尝试GPU；1秒轮询，无连续空闲/180秒等待。1600先运行，800结束前不抢1600显存；CPU预加载完成后1600一结束、free>=21000MiB立即转CUDA。输出test-step800-stopped-20261015，旧1600运行脚本完全不改，保持已捕获source hash；best-adapter2000不改，两个补测不用于自动选模。

2026-10-15用户纠正串行含义：已终止并删除原1600/800两个tmux及PID659008/661264，改为单一tmux ee-action-tests-1600-800-20261015。脚本先前台调用1600，成功退出后才前台调用800，无800预加载/等待进程。当前stage1600，仍1秒显存轮询，无180秒等待。新日志launch-test-sequential-20261015/step1600.log与step800.log。

## 2026-10-15 跨服务器迁移准备

按最新用户决定：不搬CoT final/SFT best/RL checkpoint；新机用已有Conda重建，不搬旧环境。代码GitHub，数据/CoT生成结果/来源记录精确rsync；保留基座校验及原路径约束。详细计划与结果见 tasks/migration-20261015/task_plan.md。

- [x] 源码/数据/模型/历史来源依赖和环境导出盘点。
- [x] 无训练权重传输清单与SHA校验、119项CPU回归、独立ZIP打包及CRC校验。本地已提交；GitHub push因代理及直连超时尚未完成。
- [ ] 新服务器传输和GPU验证（待SSH地址）。

2026-10-15用户要求重新挂：检查原单队列1600已完成1627条且selection-source存在，800在144条后exit143，当前无旧评测进程。保留800中断目录及input JSON为interrupted时间戳归档，仅新挂tmux ee-action-test800-restart-20261015重新跑800。1600完整结果不重复；仍CPU准备+1秒显存轮询立即启动，无180秒连续空闲等待。

## 2026-10-15 归档seed42 CoT+Action SFT+R0 RL全过程

- [x] 核对真实CoT与Action SFT/RL配置/来源/prompt/课程与训练截止状态。
- [x] 汇总完整dev、SFT及RL800/1600/2000 test；区分dev选模与补测。
- [x] task/result中文实验记录，列明公式/实现/路径/结果与局限，逐项数值和链接验证。

复盘：实验记录已写入task/result/20261015_CoT_ActionSFT_R0GRPO_seed42_实验记录.md。原始JSON逐格校验SFT11次dev、当前R0 RL6次dev和4组完整test（105项F1）；64个绝对路径链接存在，4组预测各1627条；独立RL实现交叉复核通过。区分CoT随机顺序/SFT Sorted1、dev选中2000/指定补测800和1600及2195概率硬停状态。未启动或修改训练/评测。

## 2026-10-15 R0-v2 四字段奖励与选择性策略更新

规格：task/1015codex_r0_v2_four_field_mask.md，用户已授权实现；最新追加重跑Action SFT。保留旧run及既有修改，不切分支。

- [x] 读取规格、历史教训、本地入口及配置。
- [x] 校验计划：训练奖励/trace与token mask独立模块；原评测保持；新SFT最佳权重供A/B共同初始化。
- [x] 先写失败测试，再实现R0-v2语法/匹配/四项评分和共享trace。
- [x] 四字段真实IDs定位、边界保守规则、A正负零、明确回退。
- [x] loss只mask策略分子，KL/有效分母不变；CLI显式版本与短程预算及日志。
- [x] 数值/oracle/梯度/正式评测回归/真实tokenizer与人工样例验收。
- [x] 独立review修复及真实G4 GPU smoke。
- [x] 验收后写tmux：新目录原配置重跑Action SFT，继承原CoT final；完成后A/B各从新SFT best独立启动。
- [ ] A/B各800 optimizer steps、dev400、梯度累积4/G4；同预算和来源，不按test选模。
- [ ] 归档准确状态与未完成项。

设计：旧默认保留；新增r0_v2+off/field_v2。正式配置dev400，A/B显式覆盖CLI默认200；两间隔预算800更新=3200组，不重复数据。SFT保持原seed42、Sorted1、最多4轮、原dev调度与patience规则。

复盘：184项CPU回归通过；16048条Gold自比全部1；7项启动器测试通过。独立review修复合法STOP非规范BPE误回退、等待GPU后来源复核及启动异常状态记录。真实GPU smoke16回答/1次更新通过，正10负6，reference不变/policy变化，定位回退1/16。已启动tmux ee-r0v2-sft-ab-20261015，单一串行队列新SFT→新SFT smoke→A800→B800；后续阶段尚未开始，结果未完成。证据目录 tasks/action-experiment/verification-r0-v2-20261015。

R0-v2队列启动验收补充：新SFT PID1103593实际GPU22758MiB、已完成2次更新；CoT权重精确交接/全新optimizer与scheduler通过；15个关键SFT配置字段与原run相同。正式训练未完成，A/B仍待串行执行。

## 2026-10-15 用户纠正A/B预算

- [x] 检查SFT仍运行，旧A/B未开始；仅SIGSTOP旧controller PID1103589，未向SFT PID1103593发信号。
- [ ] 新独立串行续接队列，等待当前SFT完成，之后smoke/A/B；A/B各3207更新=12828条×G4，dev400及最终dev3207。
- [ ] 检查/review/test并冻结新启动入口；启动和核实无800步派发，旧来源不改。

### 预算纠正后的再规划

用户进一步澄清关注早停耐心，不是固定满轮。full-budget新脚本未启动；旧dispatcher保持T，原SFT仍运行。已询问3次/4次无有效提升即停，待回答后实现SFT和RL一致可配置的早停，保留TC→TI选模与最大预算的区别。不得按此前推断直接派发。

## 2026-10-15 用户确认连续3次早停

- [x] 用户明确3次；停止原SFT PID1103593及已暂停旧dispatcher1103589，保留旧run，不伪造resume/completed。
- [ ] SFT可配置patience默认保留6，本轮显式3；RL默认关闭earlystop，本轮显式3；TC有效改善阈值沿用.001，独立TC/TI选模。
- [ ] RL结束审计按真实曝光前缀、实际None数及实际步数，不假报完整一轮；max3207为上限。
- [ ] 测试、独立review、真实新CLI smoke后，新目录原CoT fresh重跑SFT→新SFT smoke→A/B串行。

### 2026-10-15 最终TC patience3方案启动

用户已确认TC主指标，TI仅平分选模；TC相对anchor至少+.001才重置，连续3次即停。正式SFT新目录action-sft-v2-patience3-42-20261015；A/B上限3207而非固定800，也不要求跑满。

- [x] SFT/RL可配置早停，默认历史兼容，显式3生效。
- [x] 202CPU回归、10launcher测试；独立review修复prefix记录/源码预校验/防覆盖，再审无未解决项。
- [x] 启动tmux ee-r0v2-patience3-20261015，先真实SFT4步/GRPO2步CLI smoke，通过后正式SFT→新SFT smoke→A→B。
- [ ] CLI smoke完成与正式SFT真实GPU更新确认。
- [ ] 正式训练与A/B完整dev结果。

早先800队列及full-budget草稿已作废，旧SFT与调度器已停止；新目录独立冻结，不伪造旧run来源。

TC patience3启动复核：真实SFT CLI smoke4步/GRPO CLI smoke2步(8组32回答)全部completed，patience3/reference冻结/实际prefix证据通过。单tmux已进入正式SFT，controller PID1146381；后续smoke/A/B尚未开始。三次plateau规则单测覆盖，未声称真实smoke已触发。

## 2026-10-16 用户要求因SFT分数偏低暂停

已SIGSTOP当前A模型及其专属pipeline父进程，确认均T，B未派发；保留GPU内存中的optimizer/RNG和既有权重/日志，不终止模型、不改来源锁。暂停证据 verification-r0-v2-patience3-20261015/user-pause-20261016.json。待用户决定后续。

## 2026-10-16 用户指定本次SFT2.00epoch完整test

检查发现本次step1604权重未独立保存，被1804覆盖；现存best1804/last2205，1604仅有完整dev预测。未启动test、未终止暂停A。已向用户说明需重跑才能获得新的2epoch权重，待选择；禁止拿上次1604或本次1804冒充指定checkpoint。

## 2026-10-16 上次Action SFT 2.00epoch作为A/B初始化

用户明确重新授权开tmux：选 action-sft-v2-42-20261012/best-adapter（step1604，约2epochs），不使用本次重跑1804，也不使用旧树SFT1804。

- [x] 核验原SFT dev最佳step1604、adapter/config hash及来源交接；当前已验证训练代码不改。
- [x] 终止已暂停、初始化不同的旧A及旧dispatcher；保留旧run，禁止其B派发。
- [x] 独立新目录A/B队列，仅A→B，无SFT或额外test；同SFT独立加载，r0_v2/off对field_v2，patience3、TC anchor+.001、dev400、max3207。
- [ ] 启动器测试/review/source freeze；单tmux启动并核实真实GPU及初始dev。
- [ ] 记录当前阶段与后续结果，完成选模后自动汇总dev，无test选模。

启动进度：11项launcher测试与独立review通过，--check及来源冻结通过。tmux ee-r0v2-sft1604-ab-20261016已启动，A PID1808060，controller1808052，B尚未创建；GPU加载/初始dev待确认。复用同源码/基座/数据/相同init的已通过真实GRPO CLI smoke，所有input_sha256已核对一致。

SFT1604新A/B启动确认：A PID1808060实际GPU18738MiB，正确原SFT1604及patience3/dev400/max3207/r0_v2/off，初始dev已落盘16/1593；B未创建。后续A成功后单队列独立原SFT初始化启动B。

## 2026-10-17 A 概率一致性异常排查

- [x] 核验A失败退出、B未启动及原始traceback；保留冻结源码与checkpoint。
- [x] 从失败前逐token日志构造CPU保护重放，定位cached/no-cache分歧；模型数值根因仍需GPU复现。
- [ ] 依据可复现证据确认根因；如需改动先加入回归测试，不放宽保护阈值掩盖异常。
- [ ] 验证修复与checkpoint恢复条件，记录是否具备optimizer/RNG完整恢复能力；禁止把重新初始化称为续训。

范围：先CPU排查与复现；当前GPU资源不足以直接加载原训练，原A/B队列保持失败状态。

排查记录：[概率异常证据](action-experiment/diagnosis-probability-20261017/report.md)。已重现保护触发，未确认模型根因；未修改保护阈值/正式源码、未重启A/B。最近保存step800无optimizer/RNG，不能从1097精确续训。

## 2026-10-17 用户指定 A cp800 完整test

设计与执行：复用显式checkpoint评估器，独立launcher与test目录；GPU0无其他计算进程且空闲显存>=21000MiB连续60秒才加载，中途占用或查询失败清零；只测test，不恢复A/B。

- [x] 核查step800真实权重、dev记录与既有评估入口。
- [x] 连续60秒与占用/查询失败重置单测2项通过，脚本编译和bash语法通过。
- [x] 输入hash校验通过并记录*-inputs.json；单tmux ee-test-A800-wait60-20261017，PID3287366，CPU加载完成，wait-state确认waiting/idle=false/计时0，未开始GPU测试。
- [ ] 完成后保留完整test指标与预测，独立于训练选模。

cp800 test启动复盘：复用显式step选择校验，保留原run/训练来源；GPU空闲定义为无其他计算进程且free>=21000MiB，每秒检查、连续60秒后加载，OOM竞争重置。日志 action-experiment/launch-test-A800-20261017/test.log，退出码exit.code；结果目录原A/test-step800-20261017。

## 2026-10-17 cp800等待窗口调整为120秒

- [x] 确认旧评估器仍waiting后终止，PID3287366消失，退出143，保留旧日志。
- [x] 独立wait120目录、输出目录与计时测试；只测试同一A cp800。
- [x] tmux ee-test-A800-wait120-20261017已启动，PID3290864，CPU_READY与wait-state确认120秒/当前waiting计时0；两项门控测试通过。首次脚本路径替换错误已修正，错误日志单独保留，未加载错误模型。

## 2026-10-18 cp800 空闲立即启动

需求：取消连续120秒窗口；GPU0无其他计算进程且free>=21000MiB时立即加载，每秒检查，仍只评估同一A step800。独立新launcher保留旧输入来源和日志。

- [x] 核查旧等待器、来源路径与GPU门控；计划与本次授权一致。
- [x] 创建立即启动launcher，核验门控与脚本语法。
- [x] 停止旧等待器并在tmux启动新等待器，核验唯一进程和真实状态。
- [x] 记录验证与结果复盘。

复盘：新launcher为action-experiment/launch-test-A800-immediate-20261018，IdleGate(0)，每秒查询，保留21000MiB及无其他计算进程条件和上卡前复查/OOM重试。旧PID3290864已退出143；新tmux ee-test-A800-immediate-20261018，PID168095，CPU_READY日志确认start immediately，wait-state持续更新（当前其他进程占卡，free6020MiB）。原两项测试、立即门控/忙碌及查询失败断言、Python与Bash语法均通过。仍只测A step800；测试结果尚未产生。

## 2026-10-18 cp800 test等待窗口改5秒

- [x] 核实旧immediate仍waiting，终止PID168095并确认消失。
- [x] 独立wait5启动目录与输出目录，仅门控改IdleGate(5)，保持原A cp800/test评分。
- [x] 两项5秒与重置测试通过，bash语法通过；tmux ee-test-A800-wait5-20261018/PID178441 CPU_READY，wait-state确认waiting/idle=false/0秒。

## 2026-10-18 SFT/RL周报（10.12—10.18）

范围：输出/home/myluo/周报/26.10.18.md，重点SFT训练及重跑、旧R0完整test、新R0-v2 A/B方法与阶段性dev、故障与当前cp800等待test。不修改训练或启动新实验。

- [x] 阅读指定实验归档、既有周报风格和历史教训；分流历史事实独立复核。
- [x] 四份历史test原始metrics均验证1627条/250文档与五项F1；核对重跑SFT、新A dev和实时wait5状态。
- [x] 写入/home/myluo/周报/26.10.18.md，以SFT/RL为主线，包含两代奖励、完整test、SFT重跑及当前dev/故障/排队状态。
- [x] 独立审查通过事实边界，按反馈明确“双方均无论元”奖励规则并修正公式转义；原始test表、全部本地引用和控制字符检查通过。

周报交付复盘：文件/home/myluo/周报/26.10.18.md，覆盖本周SFT/RL方法、两次SFT、旧R0正式与补测、新四字段A/B及资源/概率异常。未改动训练或等卡任务，当前cp800仍waiting，B未启动；后续结果需单独更新。

## 2026-10-18 周报精简修订

- [x] 对照此前716/966字符周报，将原5777字符长稿归档至tasks/archive/weekly-report-20261018，正文压缩为约一页。
- [x] 保留SFT/旧R0 test表、新四字段RL阶段结果及当前状态，移除细参数和证据清单；保留选模/补测与新旧cp800区别。

## 2026-10-18 周报方法说明修订

- [x] SFT改为NEW/TRG/ARG/END/STOP动作表与真实ITT监督例子。
- [x] 重写两版RL奖励公式、匹配规则及改进动机，说明组内优势和A/B字段更新；保留原结果表。
- [x] 对照原实验奖励定义与四字段规格检查匹配/空值/mask边界；真实train首条核验例子，LaTeX转义与控制字符检查通过。

## 2026-10-18 用户稿周报润色

- [x] 保留用户表格和SFT动作例子，核对旧reward及GRPO/mask实际实现。
- [x] 重写答案级优势动机、两版奖励与选择更新例子；服务器掉线保留为用户观察，散热标为待查原因。
- [x] 奖励公式、GRPO组内优势与field mask规则对照实现一致，公式转义/控制字符检查通过，用户编辑批注已清除。

## 2026-10-18 RL逻辑澄清

- [x] 阅读用户附件作为方法背景，明确其中V1属于旧方案。
- [x] 仅重写周报RL两节，补奖励→优势→策略更新链路，给出同组奖励与只错类型的贯穿例子。
- [x] 验证0.9/0.7/0.5/0.3的优势符号与0.75四字段示例；公式/控制字符检查通过，SFT/结果表/当前状态与用户稿保持逐字一致。

## 2026-10-18 copy周报精简

- [x] 读取用户copy版本，保留两条奖励公式和字段mask规则表，删重复解释及数值算例。
- [x] 保留SFT动作例子，收束解析状态改进方向与G4维护请求；仅编辑指定copy，公式/控制字符检查通过。

## 2026-10-19 cp800 test取消5秒等待

- [x] 确认旧PID178441仍waiting后终止并核实消失，保留旧产物。
- [x] 独立immediate-20261019目录与输出，门控改为IdleGate(0)，同一A cp800/test评估条件。
- [x] 两项即时门控测试与bash语法通过；tmux ee-test-A800-immediate-20261019，PID1489898，CPU_READY且waiting，当前free850MiB，资源满足即加载。

## 2026-10-19 cp800 渐进显存预留

规格：用户要求有空闲即先占用。独立launcher在同一评估进程用无计算的uint8块逐步预留，目标21000MiB（沿用现评估预算）、256MiB全卡余量、每秒重查；不再要求其他PID退出。凑够后释放块引用交给本进程PyTorch缓存供模型复用，立即加载原cp800，竞争OOM回CPU重试；不改指标、权重、batch或训练。旧队列日志和输入锁保留。

- [x] 核查当前等待PID1489898、来源锁、评估预算；无既有预留器；采用当前PyTorch分配器，查阅本地官方memory.py实现及文档，根因是整卡门控且未预留（实现信心>=90%）。
- [x] 单元测试覆盖逐步累积/预算上限/余量/OOM保留/释放缓存，再实现并做小显存CUDA验证。
- [x] 停止唯一旧等待器，在tmux启动新队列并核查真实显存与状态。
- [x] 复盘归档；正式测试完成不属于本次启动成功声明。

复盘：5项单元测试、Python/Bash语法、小显存CUDA真实分配→保留缓存→模型复用且CPU/CUDA输出一致均通过。独立代理审核无阻断问题：21000MiB为启动预算/占位上限，不是评估阶段硬上限，生成阶段竞争OOM仍可能发生。旧PID1489898退出143；新tmux ee-test-A800-reserve-20261019，唯一评估PID1494607，状态reserving，真实PyTorch reserved764MiB、全卡free254MiB，已实际预留。测试尚未开始，保留原cp800、batch4及全部输入hash校验，新输出test-step800-reserve-20261019。

## 2026-10-19 可恢复RL A/B（goal-27）

- [x] 独立新版trainer支持A800 weights-only接续，原SFT reference、历史dev/prefix3200与global3207预算已核对。
- [x] 原子完整状态/异常回滚/日志与dev恢复实现，21项CPU测试通过（含tinyQwen3+PEFT真实adapter恢复下一update逐位一致）。
- [x] 同进程分块预留预算21000MiB、系统余量256MiB，无稳定窗口，串行smoke-A→smoke-B→A800→B，已观察PID1517828持有1148MiB。
- [x] CPU回归与独立review通过；tmux ee-r0v2-recover-A800-B-20261019已启动，test原reserve进程已退出143。
- [ ] 等资源后完成真实CLI smoke、核实正式A实际更新及后续B；目前CPU来源校验/准备中，未声称smoke或正式训练已开始。

## GPU 显存占用脚本（2026-10-19）

需求：在 EE/EXCEEDS 新增独立 Python 脚本，默认 GPU 0、8 GiB 常驻张量及 4096×4096 持续矩阵乘法；使用已有 conda 环境与具名 tmux，日志记录启动，支持 Ctrl+C 释放。

计划校验：用户已明确给出算法及启动授权；GPU 0 检查时有 15191 MiB 空闲，默认分配可容纳。保留现有训练及文件。

- [x] 检查历史教训、重复实现、GPU 空闲显存及 conda 环境。
- [ ] 写入脚本，检查语法、CLI 和输入校验。
- [ ] 在 tmux 启动默认配置，核验日志、进程及实际 GPU 显存。
- [ ] 归档运行结果及停止命令。

用户修正：改为远程 Bash 直接运行，指定终端 pts/12；停止本次 tmux 进程，核查指定终端后重新启动并复验，不沿用默认 tmux 偏好。

结果复盘：gpu_occupy.py 已写入；语法、--help、非法size参数校验完成。曾在tmux成功分配8GiB并计算，nvidia-smi核实PID1839893占8800MiB。按用户修正已用Ctrl+C停止，日志确认Memory released且tmux会话消失。pts/12已有python train_dpo.py（PID1392449），保留该进程。尝试在独立Bash直接启动的工具请求被用户拒绝，未重新启动；当前本脚本未运行。直接启动命令：/home/myluo/miniconda3/envs/exceeds/bin/python -u /home/myluo/EE/EXCEEDS/gpu_occupy.py --gpu 0 --size_gb 8。

## 2026-10-19 recovery_v3 代码审查与无窗口等待核验

范围：HEAD f720b99 为已跟踪基准；未跟踪 recovery_v3 全目录与 launch-recovery-A800-B-20261019 实际源码快照完整纳入审查。规格为当前用户消息：A800后继续RL A，异常保存可恢复checkpoint，逐步预留空闲显存，不添加稳定等待窗口；B串行。

计划校验：只读核对进程及来源锁，规范/需求独立审查，运行CPU恢复测试与故障注入，交叉验证后报告。当前请求首先是review，审查阶段不改运行中冻结代码、不重启训练。

- [x] 核对tmux、命令、显存与源码快照。
- [x] 并行审查规范与需求并定位缺陷。
- [x] 运行恢复/资源测试和必要故障注入。
- [x] 交叉核实并记录结果复盘。

执行记录：系统无python别名，文档更新改用python3；实验测试使用既有conda绝对路径。

## 2026-10-19 G52 独立终端终止观察

规格：只启动和终止本次创建的GPU占用进程，不操作既有PID1392449或pts/12；SSH分配PTY，输出真实tty/PID，以8GiB配置模拟；另一个SSH连接发送默认SIGTERM，原Bash记录退出码并观察终端输出；核验新进程消失及显存回落。两端输出用于区分终止命令终端和被终止进程所在终端。

- [ ] 创建独立远程PTY并核实新PID/TTY及分配成功。
- [ ] 从独立连接验证身份并发送SIGTERM，读取两端输出。
- [ ] 确认进程退出、原任务保留，归档观察结果。

审查结果复盘：21项CPU测试通过；另三项故障复现确认smoke initial保留策略阻断正式A（P1）、optimizer dirty提前清零导致提交窗口中断错配（P1）、A800继承初始化失败后无latest且被队列拒绝（P2）。规范/可靠性2项、需求2项，去重为3处根因；详见 action-experiment/review-recovery-20261019/review.md 与可重跑证据。本轮按review范围未改冻结训练代码、未重启或停止现有进程。来源锁/快照一致；当前smoke-A800仍reserving，尚未正式A更新；无空闲稳定等待窗口，资源不足每秒轮询。

执行偏差与重新规划：远程pts/19已创建，测试PID1850530在CUDA初始化时报busy/unavailable，自行退出1，未发送kill。复查显存仅87MiB空闲（原先15191MiB），不争抢/终止其他GPU进程。当前任务目的为终端信号显示，改用独立交互Bash中的等待Python进程观察SIGTERM；明确此阶段不声称GPU分配成功。使用systematic-debugging检查环境/compute_mode/显存，未修改远程环境。

验证复盘：交互Bash测试终端/dev/pts/20，等待Python PID1851372；独立发送终端/dev/pts/22，先核对/proc cmdline含独有TERMINAL_KILL_PROBE_PID及time.sleep(600)，再os.kill(pid,SIGTERM)，等价默认kill PID。发送端仅显示脚本主动打印的Sent SIGTERM，无系统自动提示；接收端实际显示“已终止”，随后记录退出码143并返回bash提示符。kill -0确认测试进程不存在，退出测试Bash。既有PID1392449仍在pts/12运行。本次GPU试运行因资源变化失败，不与CPU信号观察混淆。

- [x] 已创建并记录独立终端/PID；8GiB GPU尝试失败已明确记录。
- [x] 独立连接验证目标身份并发送SIGTERM，采集两端实际输出。
- [x] 测试进程退出，原任务仍在；记录退出码与终端提示并关闭测试终端。

## 2026-10-19 recovery v4 修复并重启（已获用户授权）

规格/方案：保留v3源码、原来源锁及日志；独立recovery_v4与launch-recovery-A800-B-v4-20261019，沿用原tmux名称重新启动单一串行队列。首次policy参数摘要保存于每个完整checkpoint payload，resume继承而不重算；smoke通过初始/最终参数摘要验证，不依赖保留目录，并在跳过completed时仍验证。optimizer dirty标记覆盖safe_payload/RNG提交全过程。新建目录写明确初始化标记；无latest且无已发布快照时可归档重建，拒绝丢失latest但保有发布快照的目录。A800来源、reference、奖励、mask、早停、预算、显存预留/轮询维持原配置。

方案比较：永久保留initial会耦合训练快照与校验且增磁盘；提高keep数仅推迟问题；采用持久初始参数摘要可跨任意恢复。初始化不删除旧产物，归档重建。计划已按用户“改一下再重新跑”授权校验，无新增确认步骤。

信心检查：根因3项均有确定CPU故障复现；沿用torch/PEFT与既有原子save_checkpoint、不引依赖或重复训练器运行；实现前读取本地torch optimizer/state_dict及Python os.replace作为实际版本API依据。目标信心>=90%。

- [x] 创建隔离修复目录与失败回归测试（覆盖正常/恢复smoke、边界中断、初始化失败/拒绝不明训练产物）。
- [x] 实施三处最小修复，CPU全套与故障测试通过。
- [x] 独立复核修复、语法、来源锁与实际启动命令。
- [x] 只停止旧队列所属进程，保留日志和来源，冻结v4来源并在原tmux重启。
- [x] 核查唯一新队列、GPU预留及日志，归档真实状态；GPU不足时不声称正式训练已启动。

v4实现摘要：新增12项pipeline/初始化故障回归与2项提交边界/payload回归；两批红→绿，原21项连同新增共35项CPU全套通过，真实PEFT恢复下一更新逐位一致。smoke首次digest不依赖任何initial目录，completed也复验；初始化失败保留原目录归档、拒绝丢latest的已发布快照；dirty提交窗口全程保护。v3源码锁与快照未改，GPU资源逻辑及训练配置保持原值。

完成复盘：独立trainer与runtime/queue复核均无新增P1/P2；最终35项CPU测试通过，v4源码12文件与launcher冻结。原tmux ee-r0v2-recover-A800-B-20261019 内旧模型1845231退出（队列exit241源于SIGTERM），单队列已切换v4 controller1856872/model1856873。最新实测新模型持有15100MiB，resource-state为reserving（allocator14716MiB、全卡free约88MiB），初始source/CPU准备完成，尚无正式A/B更新；资源足够立即smoke，之后按原队列串行A800→B。v3源码与快照锁未改，原产物保留；未终止其他用户进程。证据 action-experiment/verification-recovery-v4-20261019/{cpu-tests-final.log,restart.json,startup-check.json,report.md}。当前主机GPU实测RTX4090，A800指原A的step800 checkpoint。GPU smoke/正式训练尚未通过，不作为代码修复/重启成功的已验证结果。

## 2026-10-19 切换为满足GPU条件立即加载（recovery v5）

最新用户选择覆盖此前预留策略：GPU无其他计算进程（忽略本进程CUDA上下文）、空闲显存>=21000MiB即立即加载；不设稳定窗口、不在等待期间预留；资源不足每秒轮询、查询失败不放行。保留v4三项恢复修复、训练参数与原tmux，独立v5目录及来源锁。

实施前计划校验：用户已明确选择第一项；无需再确认。采用独立无分配gate而非修改冻结v4；nvidia-smi查询沿用既有immediate评估器命令，CPU模型准备结束empty_cache释放适配器加载的临时缓存，再入gate；OOM回CPU释放后重新检查。

- [x] 实现无预留gate和CPU门控测试，验证忙碌/不足/查询失败不放行、满足即加载、OOM重查。
- [x] 接入v5 trainer和队列，完整恢复回归与独立审查通过。
- [x] 停止旧预留队列，原tmux启动v5，验证显存释放与waiting状态。
- [x] 归档实际进程/资源状态与验证复盘。

v5完成复盘：51项CPU测试/3.722s通过，独立review无阻断，15个新源码/launcher已冻结，v3/v4锁保持不变。原tmux已从旧1856873切为controller1869826/model1869827；旧15GiB预留已释放。实际resource-state waiting，free14803MiB，foreign PID1392449存在；新进程只保有CUDA上下文，无预留线程/张量、尚未上GPU训练。符合“无其他计算PID且free>=21000MiB即加载”，无稳定窗口。完整证据action-experiment/verification-recovery-v5-20261019/startup-check.json及report.md；A/B正式目录尚未创建。

## 2026-10-20 A结束后test800→test1200→B串行队列

用户授权：A结束后不仅测试step800，再测试step1200，两份test完毕后才进行B；seed保持42，800/1200为checkpoint step。已核实两份真实policy权重各174655536字节存在。test单独报告，不参与dev选模/早停。

方案/计划校验：运行中v5源码与来源锁不修改；只SIGSTOP旧controller1869826阻止旧B派发，不向A模型2439990或进程组发信号。独立CPU后继tmux等待当前A正常完成并退出，然后清理已停止旧controller，逐个前台派发test800、test1200、B。任一步失败停止，保留原产物，支持重启跳过已验证完成阶段；CPU等待器不会加载第二个模型。评估器复用已有显式checkpoint选择核验及统一完整test评分，保持无其他计算PID且free>=21000MiB立即上卡、等待无预留。

- [x] 实现后继队列/显式评估器与串行、失败阻断、跳过完成测试。
- [x] CPU回归、实际800/1200来源核验、脚本语法及独立审查通过。
- [x] 暂停旧controller并启动后继等待tmux，核对A模型未中断且B未创建。
- [x] 归档新队列顺序与状态；未完成test不声称已测试。

最新规格覆盖：用户明确不管step2000 dev是否提升，A完成该dev后保存并暂停，先test800→test1200→B；A余下预算之后可补跑。未改运行中冻结trainer。独立后继controller监控A的dev-step2000完整快照，再向A模型单独发SIGTERM触发已有异常原子保存，读取最终checkpoint确认step2000，模型退出后才派发test。若step2000自然早停，则沿用完整完成快照。旧dispatcher只SIGSTOP阻止B抢先。后继评估器允许显式暂停的正式run，核验pause记录与两份完成dev权重；不假称A自然完成。SIGTERM若遇partial rollout由既有逻辑回滚至step2000。A后续恢复原v5 run，不重置optimizer/RNG/早停。

后继队列安装复盘：最新授权为A完整dev2000后无论提升否保存暂停→test800→test1200→B。24项CPU测试通过（monitor13、evaluator7、queue4），真实800/1200来源锁核验通过，独立review无阻断；新源码+原v5锁冻结，运行中训练源码未改。部署只SIGSTOP旧controller1869826且实测T态；A2439990保持Rl+、GPU19732MiB，未收到信号。新tmux ee-r0v2-A2000-test800-1200-B-20261020 状态waiting_A_dev target2000，仅CPU监控；A当时checkpoint1725，test/B未启动。原控制器不得自行恢复，后继monitor在A退出后清理。A完整恢复点与dev结果保留，后续补跑需明确恢复原A run；CPU验证时默认python缺torch已改用qwen3meld-run绝对路径。证据launch-A-test800-test1200-B-20261020/{deployment.json,verification/final-tests.log,verification/startup.json,README.md}。

## 2026-10-21 A1775故障后先test800→test1200→B

最新用户授权覆盖原2000等待：A已在1775因概率一致性保护停止且完整异常快照校验通过，不修阈值、不恢复A；先完整test800、test1200，再串行B。保留原2000队列和v5来源锁，独立新launch。新的停止证据如实记录saved_failure/step1775，不伪造2000 dev完成。新评估器核验latest完整manifest、failure/recovery-state与停止记录一致，评估source状态为interrupted_at_step1775。旧冻结controller须在A确已退出后单独TERM+CONT清理，禁止它再派发B。

- [x] 编写停止证据/评估接受1775已保存故障的回归测试，再实现新入口。
- [x] CPU全套、两份checkpoint来源及恢复快照核验、独立review通过。
- [x] 清理已停止旧controller，新tmux启动test800→test1200→B并核查实际加载/推理。
- [x] 保留A1775恢复点并归档新队列/运行状态。

追加授权：用户要求tmux启动后每2小时巡检一次。已搜索automation_update及全部可用工具，当前会话未暴露原生automation工具；将使用本地Codex CLI queue（已核对--help实际接口）向同一chat排队巡检消息，独立tmux维护2小时间隔/单实例锁/下次时间/投递结果。每次要求核查队列、日志、GPU与输出证据，仅异常/阶段完成/需用户处理时通知，正常不重复打扰；不得绕过概率保护。部署后验证一次实际queue投递连通性并保存结果。

执行验证：新入口13项CPU测试通过，真实A1775快照11文件sha与800/1200来源/训练输入锁全通过。旧controller1869826已在确认A退出后TERM+CONT清理，未恢复A。新tmux ee-test800-test1200-B-A1775-20261021，queue2980505/evaluator2980539，test800已ready并实际GPU18772MiB/98%、predictions.jsonl生成。当前串行test1200/B未创建。Codex CLI queue实际向当前chat投递成功（message01a15a37-cad5-7251-93d3-72e38be22dea），两小时巡检scheduler在配置中。

最终验证：test800实际GPU推理且预测持续写入，source锁全匹配，A1775完整恢复点仍保留、B未启动。两小时巡检tmux ee-r0-test-B-health-2h-20261021 已运行，state next_check=created+7200且scheduler PID存活；6项监控CPU测试通过，实际codex queue投递连通性已成功验证。每次巡检仅新异常/阶段或队列完成/需用户操作时通知，正常不重复打扰；巡检只读，不恢复A或放宽保护。具体下次本地时间与PID见新launch/verification/startup.json及monitoring/state.json。

## 2026-10-21 A1775概率一致性故障只读诊断

目标：解释当前新A为何在step1775报错，区分直接触发条件与已证实根因，比较采样cache、no-cache eval、train三路同token概率，核验更新/adapter/截断对齐与历史相似故障。仅用CPU及已有证据；B继续原训练，不启动GPU重放、不改冻结trainer/阈值、不恢复A。

- [x] 提取最后失败响应/token记录，以真实ProbabilityGuard构建CPU确定性复现。
- [x] 分解三路差异、定位最坏token/上下文，核对是否更新过大或训练模式所致。
- [x] 独立代码审查及历史概率故障证据对照，必要时查本地官方实现/文档。
- [x] 交叉验证结论与局限，输出可重跑证据和中文诊断报告。

A1775诊断复盘：已真实CPU重放保护错误（两种采样比较max1.215576）及eval/train零差控制。全15610响应/847612tokens统计一致定位生成与整段评分路径，异常为completion第26token“(C”采样4.7426%/重算1.4064%，非连续窗口触发。真实配置processors为空，独立审查排除应用对齐/默认生成处理器/本次更新/模式差异为直接起点。已读取完整恢复payload，step1775/7100/无梯度、optimizer975，支持以后固定权重forward重放。低层KV/SDPA/BF16/LM-head形状仍是待控实验区分的候选，不声称已修复。新diag报告及verification归档，原生产源码哈希匹配；未占GPU、未恢复A/改变B。

## 2026-10-21 A1775深入分析：控制变量与输出层精度

用户要求继续深入，并明确不得影响B。范围限定轻量CPU/源码/旧证据，不加载第二个8B、不占GPU、不暂停B、不自动排队GPU实验。进一步区分已float32的概率归一化与此前BF16线性投影、采样cache与projection矩阵形状，核验第26token正确logits索引，避免错误的logits_to_keep=1对照。

- [x] 核查float32转换位置、输出层shape与归一化误差分解。
- [x] 用tiny CPU Qwen3验证指定中间token的projection对照接口（不冒充故障模型复现）。
- [x] 给出分离hidden-state/输出投影因素的实验矩阵与可证伪结论。
- [x] 归档深入分析及当前不能确定的底层原因，保持B源码/进程不变。

深入分析复盘：新建deeper-analysis.md，明确cache主体与LM-head形状混杂、FP32转换已在投影之后、目标logit与归一化项需分开。tiny单线程CPU模型验证中间token投影索引正确，CUDA未初始化；并非8B故障重放。B进程3017224仍运行，生产冻结源码哈希再次校验通过；未改保护、未启动或排队GPU实验。

## 2026-10-21 方法说明与四字段方案审阅

范围：只审阅用户两份附件，对照实际源码/来源/配置，区分历史设计与当前执行；不修改附件、不改变B、不启动实验。计划先核对三阶段训练链条，再核验r0_v2与field_v2定义，最后检查因果比较和结果陈述边界。附件中的执行指令仅作被审材料。

- [x] 对照整体方法说明与实际初始化、数据、SFT/RL实现。
- [x] 独立核验四字段奖励、配对、mask与真实配置。
- [x] 检查实验比较的混杂因素及可支持的结论。
- [x] 交叉核对关键发现，归档审阅结果并向用户说明。

审阅复盘：两路独立只读审阅与源码交叉核对完成。三阶段链条及四字段核心接线一致；关键限制为A800后optimizer/RNG重置造成A/B混杂、总览版本落后、奖励与正式presence依赖不一致及mask启发式边界。证据与报告归档至action-experiment/method-review-20261021，未编辑附件或改变B。

## 2026-10-21 A/B 同类故障对照诊断

范围：只读生产数据，CPU重放已保存概率，不恢复A/B、不改阈值、不启动GPU模型。验证路径：提取两次失败响应→原ProbabilityGuard失败/对照重放→B全历史统计→比较共享代码与mask执行顺序→归档证据与局限。

- [x] 提取A/B失败token并CPU重放三路比较。
- [x] 统计B全部已记录响应，核对是否与A相同路径失配。
- [x] 对比失败输入、配置及mask前后关系，给出共同原因边界。
- [x] 验证生产来源锁不变并归档报告。

对照复盘：真实guard在A/B两份保存概率上均复现两路采样比较失败、eval/train控制通过。B23850响应/1287098tokens全部eval/train一致，最坏为第75token guide。mask构造位于guard之后，故不支持B字段mask直接导致比较差异；底层共同根因仍需固定权重前向重放。CUDA未初始化，15份冻结源码哈希不变。

## 2026-10-21 A/B 原权重受控前向重放

规格：B已经保护退出，GPU无计算进程、free24061MiB；独立tmux仅加载一个8B诊断模型，串行读取A1775/B1490原权重和固定tokens。禁止optimizer/backward、采样新回答、恢复训练或改生产文件。首先验证原teacher/cache数值能否重现，再分离同hidden输出投影形状/FP32输出层/attention backend。若原数值不重现，明确不可据此归因。输出只写diagnosis-AB目录。启动前再次检查无GPU进程且free>=21000MiB，否则退出，不抢占。

计划校验：复用原initialize_dual_adapters/import_policy/model config；旧replay_cached_token包含类似前向对照但需要重建optimizer，不适用于已有完整checkpoint，本次不重建更新。安装源码确认logits_to_keep中间位置接口已有tiny CPU验证。保护触发机制已CPU复现；只对诊断脚本有足够把握，不在底层原因未确认前实施训练修复。

- [x] 独立脚本与CPU语法/索引验证，检查GPU空闲及来源hash后tmux启动。
- [x] 重放A/B固定token，检查与原记录吻合程度。
- [x] 分离输出形状、输出精度、attention backend与hidden误差。
- [x] 确认诊断模型退出/生产hash不变，归档结论和未解决项。

复盘：两次GPU诊断正常退出，A/B全部有效token的full/cached均与原teacher/sampling精确一致；同hidden投影shape无该token logp差，FP32head不能解决，MATHattention仅缓解，BF16低精度归约禁用不能通解。故障目标位置第1层已有差异并在深层总体放大，未定位具体首个算子，不宣称训练已修复。独立审查通过；15生产源码/两恢复点各11文件哈希不变，诊断进程退出、GPU无计算进程。详见diagnosis-AB-20261021/forward-report.md。

## 2026-10-21 持续定位A/B共同机制（goal-28）

- [x] 首层全前缀与同输入算子控制，区分基座BF16 GEMM/LoRA/attention差异。
- [x] 同原权重全主体FP32、固定MATH backend反事实，A/B大差缩至8.01e-5/1.545e-4。
- [x] 额外4条同step响应验证，CPU原guard完整窗口与生产/恢复点来源终验。
- [x] 独立审查、资源退出及报告归档。

复盘：共同根因定位为共享BF16执行路径的形状相关数值不一致，归一化及后续多层传播放大；不是仅head形状或直接reward/mask问题。6distinct/457tokens验证，不能外推全部训练稳定。未修改冻结trainer/阈值/数据，未恢复A/B；恢复点各11文件hash保留。详见action-experiment/diagnosis-AB-20261021/root-cause-report.md。

## 2026-10-21 AB高精度修复与B完整恢复（goal-29）

方案已获用户解决授权：保留BF16权重存储，统一FP32主体算术，attention固定MATH；分块checkpoint评分控制显存。不改原guard/old概率定义，不恢复A、不改旧v5/runs。验证后新fork从B1490继承完整state继续。

- [ ] TDD稳定算术及概率/梯度/签名CPU等价。
- [ ] 新fork完整恢复与历史/来源安全CPU验证。
- [ ] GPU失败样本/控制、真实backward/accum/optimizer/resume及最坏显存。
- [ ] 独立review、来源冻结和旧恢复点核验。
- [ ] tmux启动新B并验证真实进度，归档结果。

### goal-29后台交接复盘（2026-10-22）
- [x] 稳定算术、完整fork、串行handoff与监控CPU验证：130 tests、13 subtests通过；来源receipt已写。
- [x] 旧A/B恢复点与v5冻结来源哈希保留；后台handoff及单一两小时巡检进程存活。
- [ ] GPU验收及dev batch1/2/4测速：tmux等待无其他计算进程且free>=21000MiB，立即加载，不预留。
- [ ] 全部GPU验收通过后自动冻结来源并启动新B1490；正式训练尚未启动。
用户明确授权挂tmux后交接，前台不继续阻塞等待；任何失败停机留证，不恢复A/不自动重试。详见goal-29/background-deployment.json。

### 2026-10-22 用户疑似tmux掉线核验
- [x] 核对tmux/控制器/B进程/验收日志与exit：验收completed、exit0，原验收tmux正常退出。
- [x] B控制器及模型进程存活；实际CPU读取校验旧run/checkpoint，尚未进入GPU训练；dev batch4全部显存验收通过。
- [x] 未重启任何任务；新GPU外部作业存在，原资源条件保持。巡检摘要已保存新队列monitoring/inspection-state.json。

### 2026-10-22 后续GPU启动规则调整
- [x] 核验B已占GPU运行；用户明确选择保留当前训练，后续启动连续空闲60秒。
- [x] 将最新策略写入next-start-policy.json及lessons；不修改冻结v6数值源码/来源锁，不中断当前B，不新开模型。
- [ ] 下次实际启动/恢复前接入并验证60秒准入gate（旧v6 gate仍为即时加载，禁止原样复用）。

## 2026-10-22 用户要求B停止5分钟并释放显存
规格：正常SIGTERM保存边界1507/cursor6028，释放GPU；停止后至少300秒，再由tmux串行恢复同一run。每次GPU加载前连续60秒无foreign且free>=21000MiB，中断重置；无预留；仅这次显式恢复，不自动重试故障；数值/旧冻结源码/数据/guard保持。
- [x] 发送SIGTERM，确认完整恢复点1507已发布且旧B模型退出、GPU释放。
- [ ] CPU验证独立60秒gate与外部启动器，核验来源/恢复点。
- [ ] tmux定时恢复交接及进程唯一性核验；巡检摘要保存。

执行复盘：step1507/cursor6028完整恢复点11文件及原部署38文件来源锁核验通过，AdamW/RNG可恢复，保留best1200。外部60秒gate+启动器CPU 28tests/8subtests通过并独立审查，未改冻结v6数值源码。tmux ee-stable-B1507-resume60-20261022的PID115776已在暂停deadline等待；当前无本项目GPU模型，监控仅一个。GPU不足时继续后台等。实际后续训练尚待timer与60秒准入，不声称已恢复训练。

## 2026-10-22 B1600 dev OOM修复与显式恢复
需求/设计：用户授权先处理dev显存峰值，再tmux从step1600完整optimizer/RNG边界续跑。保持冻结v6数值、数据、guard、G4/accum4和最大dev batch4；独立评估包装器释放缓存/每批临时张量，仅CUDA OOM时按4→2→1重试同批，成功后才写预测，记录真实batch与fallback，不重试batch1失败。沿用连续空闲60秒+free>=21000MiB+无foreign，不预留，仅一次显式恢复。原run/provenance来源不覆写，单独冻结资源/评估扩展并记录；部分dev由原reconcile规则隔离；旧A/B保持。
- [ ] CPU重现评估OOM并红绿验证同批降batch、无重复缺失/异常保真/恢复training与padding。
- [ ] 完整checkpoint1600/source核验，独立审查新评估与入口接线。
- [ ] 新来源锁和run扩展协议记录，tmux单模型恢复；验收以完整dev1600输出为准，等待期间不声称修复GPU验收完成。

### 用户修正：先根因后修复（覆盖上方立即恢复方案）
候选adaptive评估未部署，未重启。计划改为：
- [ ] 精确失败batch/token预算、验证脚本与真实dev内存上下文对照。
- [ ] 保存1600原权重受控重放，采集allocator/live tensor证据，区分真实峰值/引用残留/碎片。
- [ ] 单变量修复对照+原完整dev验证，证明原因后才冻结、tmux恢复。

根因调查阶段进展：已量化2.318359GiB失败申请精确对应FP32 head权重，失败batch1340短于验收1406；优先怀疑native碎片，未定论。独立诊断tmux ee-B1600-dev-memory-diagnosis-20261022已运行；CPU helper3tests通过，1600 checkpoint11文件及冻结部署来源核验通过。保持原batch4/768，用原权重及首20条token校对、native/expandable串行进程+分配快照检验，不更新优化器。当前GPU被foreign作业占用，诊断仅CPU准备/等待；adaptive候选未部署，B未恢复。报告diagnosis-dev1600-oom-20261022/report.md。

## 2026-10-22 用户要求代码/方法及修复真实性审查
范围：冻结v5→v6数值与完整恢复、新GPU60秒包装器、未部署dev兜底及受控诊断；不审无关工作区改动，不修改生产代码/重启任务。
- [ ] 算法/数值标准审查与恢复/需求审查并行独立完成。
- [ ] 重跑适用CPU测试，核对真实A/B失败样本前向、v6训练到1600、dev OOM及当前诊断状态证据。
- [ ] 对每项可复现问题给定位/影响；区分已验证修复、未解决、仅候选，归档review报告。

审查复盘：已完成两名独立审查和本轮CPU重跑（v6+launch130/13subtests、pause28/8、未部署safe-dev11、probe3；独立reward/mask66有重叠）。原概率修复在1490→1600的1760条新协议响应上最大logp差1.049e-4、eval/train0，guard未放宽；1600完整恢复点11文件通过。未解决devOOM及验收历史覆盖；新增可复现控制器中断遗留子进程P2、未部署候选漏识别诊断P2。本次仅审查，生产源码23份SHA未变；GPU诊断仍等待，正式B未重启。详细报告action-experiment/review-stability-20261022/report.md。

### GPU空闲后根因验证进展
已使用既有单一tmux取得GPU，native1600原权重首24及连续128条dev通过；原20条逐token一致、权重未改、无optimizer更新。故未复现历史OOM，不能把cached碎片假说当定论；expandable对照正在串行运行。下一步补训练→评估分配历史对照，正式B仍停1600，不部署adaptive候选。证据diagnosis-dev1600-oom-20261022/cold-replay-summary.json。

### B1600显存矩阵结束后的继续诊断
- [x] Native/expandable精确1600权重及原batch4/768：两组前128条通过，原20回答token完全匹配，未复现原故障，未证明原因。
- [ ] 新history诊断：同一16条已存训练token6396..6399前向/反向分配、无optimizer更新，native/expandable × keep/drop refs独立进程，head快照及参数摘要保护。
- [ ] 仅在复现及单变量证据支持后实施修复、完整dev1600，再恢复B；adaptive候选仍未部署。

## 2026-10-22 用户改为先降低dev显存并恢复B1600
- [x] 停止无optimizer更新的history25诊断并确认PID退出，保留诊断证据。
- [ ] dev有效批量上限2、仅CUDA OOM降1；保留数据/768预算/原guard/训练G4×accum4，启用已测可降低预留的expandable_segments，不声称原OOM根因已完全验证。
- [ ] 完成恢复入口全局运行锁/中断转发与检查probe路径，CPU回归、来源冻结和checkpoint验证。
- [ ] tmux显式1600恢复并验证单模型/60秒GPU准入；完整dev1600结果后再恢复训练。

诊断协调/异常：发现同一GPU已有另一个授权历史探针（原目录history/probe_history.py），因此新saved-token矩阵未启动，复用既有worker。真实G4历史4组+前24dev全部通过，原20回答token一致，仍未复现OOM。随后既有25组worker(PID788772)在完成6375..6378后收到外部SIGTERM，退出143，没有OOM/Python异常，未到dev；其来源需用户确认，以免重启另一会话主动停止的任务。B1600 checkpoint11文件再验通过；CPUcheckpoint生命周期实测与Torch2.4.1原始源码证据已归档新history目录；原错误申请布局仍待复现。

### B1600低显存恢复交接（最新要求执行）
- [x] dev有效cap2，仅生成CUDA OOM退1；训练G4/accum4、768预算和数值/概率保护不变；expandable allocator已接入。
- [x] CPU14tests通过；冻结来源38文件及完整1600 checkpoint11文件核验，独立扩展来源锁已生成。
- [x] tmux ee-stable-B1600-lowmem-20261022已启动一次显式恢复；先pending dev1600，再继续训练；控制器中断转发、模型运行锁和60秒gate已接入。
- [ ] 后台完整dev1600 GPU验证与后续训练完成；当前不声称历史OOM根因已完全证实。
复盘：旧todo中的根因优先阻塞策略已由用户最新降低dev显存继续指令覆盖。诊断25组SIGTERM是主代理为释放GPU主动执行，并非需用户解释的未知操作。


## 2026-10-23 用户授权 B1950 磁盘故障后显式恢复
规格：同一 v6 B run 从完整 periodic1950 边界恢复 optimizer/RNG、next_group7800、best1200 和早停状态；原1952磁盘写失败记录保留。保持训练G4/accum4、概率保护、FP32协议、dev cap2→1/768和expandable allocator。独立新入口/request/来源锁，只启动一次tmux；无其他计算进程且free>=21000MiB连续60秒才加载，等待不预留；不启动A/旧B/第二个模型，不自动重试后续故障。
- [x] 核验磁盘空间、1950完整清单/载荷、冻结来源、当前无本项目模型，确认恢复日志与dev1600保留行为。
- [x] 复用已有恢复入口到独立1950目录；绑定当前failure和manifest摘要，保留旧部署与旧来源锁。
- [x] CPU回归60秒gate、恢复入口/评估接线；冻结新入口来源并独立审查。
- [x] tmux启动一次显式恢复，核验队列PID、60秒资源状态、GPU无预留及单模型/单监控。资源不够即后台等待，不等待训练完成。

复盘：1950完整恢复点11文件、205来源文件、38冻结部署文件通过；独立审查确认504项Adam step1950、RNG/best1200/早停1600状态完整。CPU入口/评估14tests及gate23tests/8subtests通过。独立1950扩展10来源已冻结；tmux ee-stable-B1950-resume60-20261023正在CPU准备/资源等待，仅一个项目trainer、一个巡检调度器。当前仅完成后台挂载，实际训练恢复与GPU60秒准入由后台继续；没有等待整轮训练结束。
