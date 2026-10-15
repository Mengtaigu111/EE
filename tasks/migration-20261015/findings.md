# 迁移发现

完整依赖盘点见 dependencies.json（594文件，19,065,770,512字节，包含可选旧训练权重）。实际传输以 transfer-manifest.json 为准，用户已要求排除所有训练adapter/checkpoint。

环境核对与重建见 environment-report.md。当前环境CPU检查通过；新机尚未验证。源码/历史锁冲突、CoT数据生成来源见 findings-dependencies.md。

无目标SSH地址，尚未执行跨机传输；未启动新训练。Github提交不包含数据或模型，它们由rsync清单覆盖，包括未提交或与历史Git版本不同的必要源码。
