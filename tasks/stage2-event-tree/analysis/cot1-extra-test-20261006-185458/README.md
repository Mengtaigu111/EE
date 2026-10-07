# CoT1 supplementary test checkpoints

User requested checkpoint1804/1604/3208 and faster inference. Original
dev-selected2205 and all historical outputs remain unchanged.

Benchmark48 examples (first32 plus16 longest prompts): batch4 48.9744s vs
batch8 45.3756s. Batch8 saved7.35% time but changed2 raw/parsed event trees,
so batch4 is retained with the original greedy319-token protocol. Benchmark
records and selected inputs are kept for review; no benchmark scores select
models or affect training.

Supplementary queue launched北京时间2026-10-06 18:59, tmux
`ee-cot1-extra-test-20261006-185915`, order1804→1604→3208, full1627 sentences
each. Estimated70–80min, around20:10–20:20. Actual status: state.json;
logs: console.log and checkpoint-eval/*.predict.log; process exit: exit.code.

Each completed checkpoint is independently reconstructed from raw outputs,
checked for exact IDs/metadata/input/adapter hashes, merged document coverage,
all18 official P/R/F1,7 complex scores and four scoring modes. Reference2205
full test validates the adapted helper (only tree/checkpoints path and batch
argument differ from the prior vetted verifier); wrong batch is rejected.
Source/input locks include original completed run, original sources and all
three checkpoint integrity files. New results are in this directory only.

On completion report.md compares all three checkpoints with2205. Historical
dev selection is not changed based on supplementary test outcomes.
