# Seed52 CoT1→Sorted1复现实验

用户最新决定：跑seed52，替代同seed提速重跑及拟议seed43。加速benchmark保留，不将batch2或Liger带入跨seed复现。

冻结规格：原始Qwen3-8B、新LoRA，无历史adapter；原用户定稿CoT prompt/原答案随机一轮802更新，LR1e-4cosine/warmup25；原AppendixNoCopy树prompt/原data/gold，首轮全部按难度排序、之后随机，树最多3208更新，warmup97/plateau，13完整dev点401/802/1203/1403/1604/1804/2005/2205/2406/2606/2807/3007/3208，officialoffset AC控制，bad3减半/bad6停，其他18P/R/F1记录，test不控制。micro1/acc16/bf16/SDPA/原gradientcheckpointing，不使用Liger。仅训练/采样seed42→52，CoT独立dev诊断不执行以省GPU时间。新directory/run/tmux/source snapshots，无旧文件覆盖。

实现：复制原online runner/config/verify/train与安全checkpoint继承；seed所有初始化/Trainer/data/sampler/独立验证一致52；tree_reference仍锁原数据/规格seed42来源，实际training cfg显式seed52。新sampling独立审计firstsorted和seed52+epoch随机后续及epoch尾12，原loss归一化不改。generation元数据seed52、batch4/greedy319保持。

- [x] CPU seed52配置拒绝旧seed、完整/partial采样与epoch尾组测试通过。
- [x] 独立review比较原版，仅seed与独立诊断移除。
- [x] 新tmux两阶段GPU冒烟：最长32、CoT2updates→tree4updates/2dev，标准adapterreload逐条一致，完整安全state/采样/反馈/评分核验。
- [x] 新tmux正式802+3208预算开跑；真实配置/初始化/采样前缀/GPU进度证明启动，按原真实耗时与新进度给ETA。
- [x] 归档startup/tests/review/smoke/manifest/source hashes；正式结束自动独立复算完整dev与devbesttest。
