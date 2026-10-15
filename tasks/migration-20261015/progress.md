# 迁移进度

2026-10-15：完成两个独立盘点。用户两次缩小范围：不搬任何adapter/checkpoint；不复制Conda环境。采用GitHub + 精确rsync清单 + 固定版本Conda重建，目标机地址尚未提供。

验证：Action主线119 passed；533文件17,922,499,744字节本机SHA全量核验0失败；manifest与rsync列表一致，无训练adapter/checkpoint/cache；验证脚本正确/损坏/缺失三个用例通过；环境CPU导入和固定版本检查通过。基座12文件16,397,431,693字节，其余1,525,068,051字节。此次完整重hash，不再仅继承旧模型hash。Git暂存限定核心概率保护、CoT必要源码、只读补测入口、验证证据及迁移工具；未暂存其他dirty代码或旧任务文档。

用户确认G4已有Conda/Qwen8B，并希望ZIP上传。默认project-only，无基座/环境/adapter。GitHub push尝试连接代理10.70.33.27:7897及直连github/ssh443均超时，已终止挂起push；本地核心提交f8ca0ff存在，未声称远端更新。改为独立ZIP（含迁移工具，解压无需clone）并做CRC/SHA验证。

ZIP首次实测：544文件，168,096,686字节（约160MiB），压缩加全量CRC耗时8.36秒。不含基座/训练权重/旧环境。G4目标路径与模型版本未核验，尚未上传。最终ZIP及同名.sha256位于 /home/myluo/EE/，主线119 CPU测试通过。
