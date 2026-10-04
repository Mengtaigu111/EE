# 数据与评测

## 数据单位

原split：train 12,828句、dev 1,593句、test 1,627句（250摘要）。每条输入=完整摘要+指定目标句；输出仅包括trigger位于目标句的事件，argument允许来自摘要其他句子。训练时句子样本混合，不保留摘要内连续训练顺序。

Test同样逐目标句独立生成，前句预测不喂给后句；最后合并摘要级评分。基线与课程的数据字段/gold相同，Easy2第二阶段train与基线逐字节相同。

## 两种课程

难度代理：0.5×[(1−1/事件数)+(1−句内各事件角色填充率的类型内百分位均值)]。角色填充率=已出现合法role种类/schema合法role种类；不是模型实测难度，ROOT不打难度分。

- Easy2：B1 5,427 + 固定ROOT 1,077，共6,504；随机2轮=814步。全量阶段3,196步。best2406处B1累计约4轮、B2约2轮，基线各约3轮。
- Sorted1：全部12,828首轮按难度升序，同分seed42随机，ROOT每隔6–7条插入；802步。其余3,208步逐轮随机。真实消费64,140条，完整5轮验证通过。

## 官方评测口径

[utils.py](/home/myluo/EE/EXCEEDS/utils.py:237)支持offset/token。offset比较token位置序列；token比较文本/token序列精确一致，不是语义相似。AC在trigger+type匹配后计分。

现有选模与历史主指标为offset。Raw token从原始预测树重新解析合并，不调用aligner、不删未对齐预测、不做文本去重。两种口径不可混为同一分数。

官方按trigger表达建字典：重复trigger文本在token模式可能覆盖；offset模式同位置多事件也有覆盖限制。token不一定比offset高，差值不能全部当作对齐损失。基线52篇、Easy2 67篇预测摘要有重复trigger文本。

使用隔离的 [evaluation-v2-20261003](/home/myluo/EE/EXCEEDS/tasks/stage2-event-tree/evaluation-v2-20261003/report.md)，修复跨句trigger复用、畸形树解析、对齐前文本去重等问题，原评分器不改。基线旧offset AC41.34%，同预测修复重评后41.88%；比较用修复后的结果。
