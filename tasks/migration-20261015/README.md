# EXCEEDS 新服务器迁移（不搬训练checkpoint）

按最终要求：**所有 CoT/SFT/RL adapter 和 checkpoint 均不迁移；新机已有 Conda，重建环境。** 旧服务器文件和任务保持不变，本轮没有启动训练。

## 迁移内容

实际清单 `transfer-manifest.json`：533文件、17,922,499,744字节（约16.69GiB）。其中Qwen3-8B基座12文件、16,397,431,693字节；代码/数据/记录约1.42GiB。没有复制整个19GiB模型目录，也没有Conda环境或训练权重。

包含：SciEvents原始数据、sentence/document数据、NoRE Type-CoT数据、Action v2数据和完整审计、schema、Sorted1课程顺序、当前源码以及历史来源快照、配置/评测记录。CoT生成材料是 `task/prompt/3.10-1.md`、`run_prompt_3_10_full.py` 和 `runs/prompt-3.10-1-train-full-merged-20260926.jsonl`；当前CoT训练使用 user-final-20261005 模板和数据，详见 `findings-dependencies.md`。Action采用数据清单冻结的短v2 prompt，不用后来手工修改文本冒充历史prompt。

`dependencies.json` 是完整盘点，包含已排除的旧权重；**实际传输只用 files-from-root.txt**，不是完整盘点列表。`files-project-only.txt` 排除基座，仅当新机已有逐文件SHA一致的基座时使用。

## GitHub 和 rsync

代码分支：https://github.com/Mengtaigu111/EE/tree/review/action-sft-grpo-v2-20261012

先在新机检出该分支，再在旧机执行rsync。必须保留真实路径 `/home/myluo/EE/EXCEEDS` 和 `/opt/models/Qwen3-8B`；目标账号需要对这些目录有写权限。也可以先由管理员建立bind mount。普通软链接会被 `Path.resolve()` 穿透，不能保证来源锁仍匹配。目标路径若无法保留，应先设计显式映射和新provenance，不能关校验或直接改历史hash。

```bash
# 新机：目标目录必须可写，且没有需要保护的同名文件
 git clone --branch review/action-sft-grpo-v2-20261012 --single-branch \
   https://github.com/Mengtaigu111/EE.git /home/myluo/EE/EXCEEDS

# 旧机：把 user@new-host 换成真实SSH目标。先预览，再传输；没有 --delete。
 cd /home/myluo/EE/EXCEEDS
 rsync -anR --protect-args --files-from=tasks/migration-20261015/files-from-root.txt / user@new-host:/
 rsync -aR --partial --info=progress2 --protect-args \
   --files-from=tasks/migration-20261015/files-from-root.txt / user@new-host:/
```

rsync会覆盖清单内的同名文件，因此目标应是本次迁移的新检出目录。必要源文件中有历史实验尚未提交的工作区版本；精确清单会将其一并搬过去，GitHub单独clone不能保证与旧机完整工作区一致。不会复制 `.git`、SSH密钥、shell环境、缓存或整个历史run。无需额外制作18GB压缩包。

如果新机已有同版基座，可将两条rsync命令的filelist换成 `files-project-only.txt`；之后仍对完整清单校验，基座hash不匹配不能继续。

## 新机Conda环境

```bash
 cd /home/myluo/EE/EXCEEDS
 conda create -n qwen3meld-run python=3.10.20 pip -y
 conda activate qwen3meld-run
 python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
 python -m pip install -r tasks/migration-20261015/environment/requirements-core.txt
 python -m pip install -r tasks/migration-20261015/environment/requirements-validation.txt
 python tasks/migration-20261015/environment/verify_environment.py
 python tasks/migration-20261015/verify_assets.py
 python -m pytest -q tasks/action-experiment/tests
```

环境完整快照及已知非可迁移依赖见 `environment-report.md`。不要盲装完整pip freeze中的本地构建路径。CPU版本检查通过并不代表新机GPU通过；还应检查驱动、BF16能力和实际生成/概率一致性，再开tmux正式训练。

## 重新训练的边界

这份材料支持重建 **CoT → Action SFT → R0 RL**，不能直接继续旧RL或test旧CP。重新生成当前源码对应的新实验manifest，输出到全新run目录；保留旧run元数据以供比较，不能复用其completed状态。CoT和SFT新权重将有新的hash，后续handoff应指向新产物，不能冒充旧adapter。

训练入口：`tasks/type-cot-experiment/type-cot-goldorder-20261009/train_cot.py`、`tasks/action-experiment/train_sft.py`、`train_grpo.py`。CoT总pipeline `run.py` 还会训练旧tree阶段，不能把它当作仅CoT入口直接运行。新CoT路径需与Action初始化参数匹配，RL reference保留完整SFT LoRA。当前R0、v2 prompt、文本评测器和来源校验均保留。

旧RL在step2195概率严重偏差保护退出，并非完整训练结束；迁移不代表该问题已修复，新服务器正式RL前仍需重新验证概率路径。旧历史manifest对当前 `train_grpo.py` 和CoT `config.py` 有版本hash冲突，详见盘点记录。`verify_assets.py`只证明搬运字节一致，不能替代每个训练入口的来源/语义校验。

## 本机验证 / 尚未执行

- Action主线CPU回归：119通过。
- 当前环境CPU版本/导入验证通过；清单文件本机全量SHA验证结果见progress.md。
- 尚未提供目标SSH，未跨机传输、未在新机安装或验证GPU，也未启动新训练。
