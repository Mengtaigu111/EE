# Online-dev与Type-CoT实验Git归档

本次归档从2026-10-04交接方案到2026-10-07 seed52复现：原生在线dev控制/安全checkpoint、用户定稿Type-CoT prompt与数据重写器、seed42 CoT1→树和seed52同方案代码/测试/最终指标，以及batch/融合loss诊断。

- [x] 明确文件范围并生成SHA256清单。
- [x] 执行相关CPU回归，复核文件范围/大小/敏感文件；91tests+21subtests通过。
本地提交的身份与内容以git log/git show为准，执行结果记录在共享tasks/todo.md。

结果入口：`tasks/type-cot-experiment/type-cot-online-20261006/results.md`和`type-cot-seed52-20261006/results.md`。seed42 devbest2205/testAC42.4606%；seed52 devbest2406/testAC40.9853%，官方offset。seed52补测2205/1804/3208仍在运行，当前只有启动证据，不能当成完成结果。

保存源码、prompt、manifest、验收与结果摘要；模型/优化器张量、缓存、原始大型训练语料和预测文件不纳入本次提交。完整原始文件仍在本机manifest指定路径。新clone需恢复SciEvents及CoT标注原始文件，按manifest核对SHA256，再运行builder/runner；本提交不是包含模型与语料的自包含镜像。

提速诊断使用的可选Liger0.6.2通过`pip install --no-deps --target tasks/type-cot-experiment/type-cot-fast-20261006/vendor liger-kernel==0.6.2`恢复；第三方vendor源码不重复提交。该诊断未用于正式seed42/52模型。CPU训练用conda环境qwen3meld-run；torch2.4.1/transformers4.52.3/peft0.19.1，GPU4090 24GiB。

共享tasks/todo.md/lessons.md及其他项目的并行修改不纳入本提交；本归档保留本次提交的计划、证据与结果入口。未请求推送远端，本次只本地commit。
