# 用户定稿prompt与已有CoT答案兼容性

用户提供的最终英文prompt原样保存于同目录prompt_template.txt，未追加trigger限定、完整角色列表、固定类型排序或输出禁令。

全量审计：12828条train，10704条有事件、2124条无事件，19488个事件实例。所有最终inventory与gold一致；所有解释中的Type计数与gold一致（包括同类重复实例）；有事件样本均具备Reasoning/Event Family Reasoning/Event Reasoning结构。无事件样本不含Event Reasoning段，使用简短解释与None，符合用户无事件条款。旧答案的Type: <PRP></type>标签是类型表达的既有写法，inventory parser支持，无需改动答案。

独立语义抽检：十类各3个正例检查位及3个负例，共33个检查位、25条不同样本。解释确实以论元文本和角色关系支持类型，跨句论元使用完整摘要。任务与输出形式兼容，可更换输入prompt。

抽检发现的质量边界：部分Event Family描述与细分类别不严格对应；少数PUR/WKS解释明显由标注驱动、自然语言论证较弱；负例解释较模板化；正例常引用新prompt中未提供的Appendix F.2模板名。这些是已有解释质量问题，不是新prompt与输出格式冲突。本次不改gold或completion。全量结构检查与抽样审阅不能证明所有解释语义正确，也不能保证性能提高。

已生成新实验数据：
`tasks/type-cot-experiment/data/type-cot-recognition-user-final-20261005/`。
train/dev/test分别12828/1593/1627条，统一替换prompt，其余字段逐行原样保留。
历史data/type-cot-recognition、builder、历史run均保留。

可复现命令：

```bash
/home/myluo/miniconda3/envs/qwen3meld-run/bin/python tasks/type-cot-experiment/scripts/rewrite_prompt_dataset.py \
  --source tasks/type-cot-experiment/data/type-cot-recognition \
  --template tasks/type-cot-experiment/prompts/user-final-20261005/prompt_template.txt \
  --output <新的不存在的输出目录>
```

manifest.json保存prompt和全部split来源/输出hash；verification.json保存独立逐行字段、prompt回填、hash和训练token检查。本轮只完成prompt数据替换，尚未启动CoT训练；下一轮CoT训练应指定这个新data目录。

验证结果：5项CPU测试通过；全部16048条样本的非prompt字段完全一致，prompt回填及来源hash核验通过。使用真实Qwen tokenizer及训练chat template检查全12828条train，最长2163 tokens，小于4096，零超长、零截断，assistant prefix全部一致，最大答案长度887 tokens保持不变。

抽检document索引：A=febd573ada9568c635f6d8aeada27ec5，B=1a6285faf0918175c1ea9e0b7c8ea82e，C=6bab1cf097070e6d457c9c8fd0e74e57，D=71ff0f02bc14a28822f0cdf6c508aae2，E=636dd0c8ece0788d40d37b9f500026d8，F=bbe87249393bc09725f2b0dcfda04997，G=a0fd29c17984ed8d2e2b7f86831cb0a4；数字为sentence-index。
PUR=A3/A7/B4；ITT=A0/B0/C1；RWS=D1/E2/F1；RWF=A2/B1/D2；PRP=A3/A4/B2；WKS=A6/A7/B3；MDS=A3/A5/A6；FIN=A8/B4/B8；CMP=A8/C6/G3；FAC=A8/B4/B5；None=A1/A9/C0。
