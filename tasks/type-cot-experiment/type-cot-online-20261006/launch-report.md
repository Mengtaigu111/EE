# 2026-10-06 CoT1 → Sorted1 dense-dev 启动与审阅记录

正式启动：2026-10-06T00:31:00.677888+08:00。tmux：`ee-cot1-dense-20261006-003100`。

Run：[run.json](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-42-20261006-003100/run.json)。训练log：[cot-train.log](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-42-20261006-003100/cot-train.log)。

配置：全新Qwen3-8B/LoRA，用户定稿CoT prompt及原答案，随机全train一轮802更新（cosine/warmup25）；仅本run最终CoT权重进入树阶段，新optimizer/scheduler/feedback，原AppendixNoCopy/Sorted1课程，树最多3208步，warmup97。dev401/802/1203后每约200步，官方offset AC控制3次降率/6次停止。test仅最后dev-best模型。

审阅：独立spec/data review通过；独立quality review发现最终test绑定/原始到合并一致性/hash与LR组验收缺口，已TDD修复并获reviewer关闭。最终CPU56+5=61tests及21subtests通过。原CoT旧测试与树旧测试有同名module冲突，分别进程测试通过（与实际pipeline进程隔离一致），不修改历史模块。

GPU冒烟：最终run `/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/runs/C-TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev-42-20261006-002540-smoke` exit0，CoT2步/树4步，最长32真实样本，14句完整dev两次，checkpoint/optimizer/feedback/score独立核验；独立重载14条raw输出完全一致。另有第一轮通过记录保留。

启动核验：2026-10-06T00:34:35.937650+08:00，CoT已完成12/802更新、实际观察197microbatches，loss有限，source hashes未变，真实GPU PID/显存：`2493221, /home/myluo/miniconda3/envs/qwen3meld-run/bin/python, 21694 MiB`。采样前缀独立核对seed42随机顺序一致。当前是正式CoT训练，不是烟测或排队。

时间估计：CoT当前约11秒/更新，约2.5小时；结合历史树训练/完整dev耗时，完整pipeline预计15–20小时（北京时间10月6日约15:30–20:30）。早停、生成长度和GPU速度会影响实际时间；此估计含更密集dev及终点诊断/test，后续以真实速度更新。

训练尚未完成，不宣称性能改善。仅本次CoT→树实验已启动，没有额外启动无CoT对照。历史prompt/gold/代码/run保留。
