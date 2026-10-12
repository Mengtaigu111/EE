"""Full real-tokenizer cap audit; no generation or training, no gold truncation."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tasks/action-experiment'))
sys.path.insert(0,str(ROOT/'scripts'))
from action_common import DATA_DIR,MAX_NEW_TOKENS,MAX_LENGTH,MODEL_PATH,verify_data,sha256,load_module
from action_language import parse_actions,load_schema
from transformers import AutoTokenizer
OUT=Path(__file__).resolve().parent
manifest=verify_data()
tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
schema=load_schema(DATA_DIR/'prompt_template.txt',ROOT/'data/SciEvents/ontology.json')
utils=load_module('_cap_audit_utils',ROOT/'tasks/stage2-event-tree/scripts/model_utils.py')
inputs={str(DATA_DIR/f'{split}.jsonl'):sha256(DATA_DIR/f'{split}.jsonl') for split in ('train','dev','test')}
results={};longest=[];all_lengths=[]
for split in ('train','dev','test'):
 lengths=[];max_prompt=0;max_full=0
 for line in (DATA_DIR/f'{split}.jsonl').open():
  row=json.loads(line)
  prompt=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
  full=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']},{'role':'assistant','content':row['completion']}],tokenize=True,add_generation_prompt=False,enable_thinking=False)
  assert full[:len(prompt)]==prompt,row['id']
  answer=full[len(prompt):]
  encoded=utils.encode_training_row(row,tokenizer,MAX_LENGTH)
  assert len(encoded['input_ids'])==len(full),row['id']
  assert sum(t!=-100 for t in encoded['labels'])==len(answer),row['id']
  assert len(answer)<=MAX_NEW_TOKENS and len(full)<=MAX_LENGTH and len(prompt)+MAX_NEW_TOKENS<=MAX_LENGTH,row['id']
  recovered=tokenizer.decode(answer[:MAX_NEW_TOKENS],skip_special_tokens=True)
  assert parse_actions(recovered,schema)==parse_actions(row['completion'],schema),row['id']
  lengths.append(len(answer));all_lengths.append(len(answer))
  max_prompt=max(max_prompt,len(prompt));max_full=max(max_full,len(full))
  longest.append(dict(id=row['id'],split=split,assistant_tokens=len(answer),prompt_tokens=len(prompt),full_tokens=len(full)))
 ordered=sorted(lengths)
 results[split]=dict(rows=len(lengths),assistant_max=max(lengths),assistant_p50=ordered[len(ordered)//2],assistant_p95=ordered[int((len(ordered)-1)*.95)],prompt_max=max_prompt,full_max=max_full,above_cap=sum(x>MAX_NEW_TOKENS for x in lengths))
assert inputs=={p:sha256(p) for p in inputs}
verify_data()
v2=ROOT/'task/101202_action_prompt_v2.txt'
report=dict(status='passed',rows=len(all_lengths),generation_cap=MAX_NEW_TOKENS,context=MAX_LENGTH,assistant_prefix_mask_and_no_truncation_verified=True,decoded_capped_targets_parse_exactly=True,splits=results,counts_above_candidates={str(cap):sum(x>cap for x in all_lengths) for cap in (319,384,512,640,671,768,1024)},longest=sorted(longest,key=lambda x:x['assistant_tokens'],reverse=True)[:10],v2_equals_frozen_template=v2.read_bytes()==(DATA_DIR/'prompt_template.txt').read_bytes(),v2_sha256=sha256(v2),data_sha256=inputs,scope='Gold target token coverage; arbitrary sampled predictions may still reach cap and are recorded as truncated, not guaranteed complete')
(OUT/'token-cap-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False),flush=True)
