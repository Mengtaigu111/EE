# Seed52指定checkpoint完整test

用户指定2205/1804/3208；复用原评测和独立artifact核验，只改seed52与历史devbest2406。新输出，不改原run/best/数据/prompt/gold。batch4/greedy319/bf16/原官方offset及全部其他口径分别记录。

- [x] checkpoint完整性/来源/代码/数据锁冻结。
- [x] CPU重算已测2406全部artifacts及错误batch负例、独立review。
- [x] 新tmux串行2205→1804→3208，实查首批真实输出与ETA。
- [ ] 每CP结束独立重解析/merge/18官方指标/7complex/四口径复算；完毕四CP并列表。
