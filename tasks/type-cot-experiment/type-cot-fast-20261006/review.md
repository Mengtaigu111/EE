# 独立review

2026-10-06 /root/fast_independent_review 对新方案与原online版本只读交叉检查，未发现P1/P2阻塞。独立亲自执行CPU suite：25 passed in10.46s。

核验：原prompt/gold/course、有效batch16、13完整dev、officialoffset AC反馈、test仅devbest、本次CoTadapter交接/新optimizer及安全state保存保留；多样本batch记录与epoch尾12条审计完整。真实CPU tinyQwen3跨epoch训练证明不同答案长度及batch1/2的loss归一化和参数更新在浮点容差内一致。

benchmark核验：真实Trainer native num_items_in_batch、固定16samplegroups、2warmup/8measured与最长32压力、独立fresh子进程、baseline失败禁止启动、candidate>=5%收益/finite/显存余量。正式需benchmark和两阶段GPUsmoke完成后再开。

非阻塞建议：run.py原docstring提及已删CoTdiagnostic已修正；256MiB余量门槛较紧，需结合实际最长压力/smoke评估。不声称已验证GPU或正式启动。

## 原生 GPU benchmark 的独立复核

逐组读取 config/result：同一 phase 的 batch1/2 update_groups 和 lengths 完全相同。CoT batch1 完成10次更新，8次计时，finite loss/gradient 与最长32压力通过；batch2 因 CUDA OOM 失败。树 batch1/2 均完成10次更新与8次计时、finite 和最长压力通过；平均耗时9.3525167 vs10.4397146秒/更新，batch2 慢11.6247%，峰值余量仅162988032 bytes（约155.44MiB），同时未满足原256MiB门槛。因此两阶段回退microbatch1/accum16与实际证据一致。未启动GPU进程。
