# 新服务器环境盘点（2026-10-15）

当前环境是 `/home/myluo/miniconda3/envs/qwen3meld-run`，磁盘占用 **8.6 GiB**。已导出环境清单，没有安装或升级任何包，也没有启动模型或训练。

| 项目 | 当前值 |
|---|---|
| 系统 | Linux x86_64，glibc 2.35 |
| Python | 3.10.20 |
| PyTorch / CUDA runtime | 2.4.1 / 12.1 |
| cuDNN | 9.1（`torch.backends.cudnn.version() == 90100`） |
| Transformers / Tokenizers | 4.52.3 / 0.21.4 |
| PEFT / Accelerate | 0.19.1 / 1.13.0 |
| NumPy / SciPy | 1.24.3 / 1.10.1 |
| Safetensors | 0.7.0 |
| GPU / 驱动 | RTX 4090 24 GiB / 535.288.01 |
| 传输工具 | rsync、tar、zstd、gzip 均存在 |
| conda-pack | 环境和 base 均未安装，本轮未安装 |

## 已导出的文件

- `environment/conda-explicit.txt`：Conda 精确构建/下载 URL，**不包含 pip 包**。
- `environment/conda-environment.yml`：`conda env export --no-builds`，已移除机器相关 `prefix:`；完整历史环境快照。
- `environment/pip-freeze.txt`、`environment/pip-list.json`：完整 Python 包清单。
- `environment/requirements-core.txt`：当前核心训练/评测直接依赖和 tokenizer/序列化运行依赖的版本固定。
- `environment/requirements-validation.txt`：测试依赖。
- `environment/source-import-dependencies.json`：依赖对应源码样例，避免凭空补包。
- `environment/nonportable-references.txt`：本地路径和 URL 依赖，迁移时单独处理。
- `environment/runtime.json`：机器架构、运行库、版本、工具可用性。
- `environment/verify_environment.py`：新机可执行的 CPU 环境检查；不加载模型、不占用模型显存。

导出过程过滤了 URL userinfo、常见凭证 query 参数和 Conda token channel 路径。没有读取 `.env`、`.bashrc` 或 SSH 私钥。

## 最方便的迁移方式

如果新服务器同样是 Linux x86_64，且能保留 `/home/myluo/miniconda3/envs/qwen3meld-run` 这个绝对路径，可以使用 `rsync -a` 或 `tar` 复制整个环境。**普通环境目录复制不支持任意改路径**：bin 脚本 shebang 等可能包含旧绝对路径。目标驱动、glibc 和 GPU 能力仍需检查；目录复制也不包括显卡驱动。

如果目标路径不同，使用 conda-pack 打包和解包后执行 conda-unpack 更适合，但当前两边均未确认安装 conda-pack，本轮没有安装或生成 conda-pack 包。也可以按下面的版本固定重新建立环境。环境全量清单包含许多旧项目依赖，不宜无条件重装全部 pip freeze。

## 重建参考命令

以下命令只写在说明中，**没有执行，也没有在新服务器验证**：

```bash
conda create -n qwen3meld-run python=3.10.20 pip -y
conda activate qwen3meld-run
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r tasks/migration-20261015/environment/requirements-core.txt
python -m pip install -r tasks/migration-20261015/environment/requirements-validation.txt
python tasks/migration-20261015/environment/verify_environment.py
```

`requirements-core.txt` 使用当前实际安装版本；Torch 单独指定 cu121 wheel 源，避免仅根据 `torch==2.4.1` 重建出不同 CUDA 运行库。该核心清单服务于 CoT / Action SFT / R0 RL 及现有评测，未包含全部历史数据预处理工具。完整环境恢复可使用原环境整体复制或完整 YAML，再做验证。

注意以下已核实问题：

1. `pip-freeze.txt` 中 `packaging @ file:///home/task_.../conda-bld/.../work` 是构建机路径，在新服务器不存在。当前实际版本为 25.0，完整 YAML 中由 Conda 的 `packaging=25.0` 提供；不能把这条 file URL 原样作为可安装依赖。
2. `en_core_web_sm` 来自 spaCy 官方发布 wheel URL；不在核心训练最小清单中。需要历史 spaCy 流程时，另行安装该 URL 并验证网络可达。
3. 原环境并存 cu11/cu12 的历史 NVIDIA 依赖，且 `torchaudio==2.0.0` 与 `torch==2.4.1` 不配套。这些不是当前 Action 训练调用的依赖；没有为了迁移修改现有环境。
4. Transformers 4.52.3 下代码依赖 `use_model_defaults=False`，避免 Qwen 生成配置覆盖显式默认参数。不能只迁移权重后任意升级 Transformers/Tokenizers 再声称复现实验。
5. R0 用到 `scipy.optimize.linear_sum_assignment`；SciPy 必须保留。当前训练不是 bitsandbytes 量化训练，环境中未安装 bitsandbytes。

新机正式训练前还需要验证完整来源校验、CPU 测试以及真实 GPU 上的 BF16 生成/概率一致性。这里的环境导出不等于已经完成跨机器可运行性验证。
