import json
from pathlib import Path
import sys
import hashlib
ROOT=Path('/home/myluo/EE/EXCEEDS')
sys.path.insert(0,str(ROOT/'tasks/stage2-event-tree/scripts'))
from model_utils import read_rows,audit_rows
from transformers import AutoTokenizer
manifest=ROOT/'tasks/stage2-event-tree/curriculum/user-v2-20261002/mean_curriculum_manifest.json'
m=json.loads(manifest.read_text())
tokenizer=AutoTokenizer.from_pretrained(m['base_model'],local_files_only=True)
tokenizer.padding_side='right'
result={'prompt_sha256':m['prompt_sha256'],'max_len':4096,'splits':{}}
for split in ('train','dev','test'):
    path=Path(m['data_dir'])/f'{split}.jsonl'
    audit=audit_rows(read_rows(path),tokenizer,4096)
    assert audit['overlength_count']==0
    result['splits'][split]={**audit,'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
with (manifest.parent/'token_audit_all_splits.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
