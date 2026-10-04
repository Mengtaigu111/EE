"""Score raw sentence forests with official token metric, without span alignment."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
STAGE=HERE.parent
sys.path.insert(0,str(STAGE/'evaluation-v2-20261003/scripts'))
from linearize import parse_forest
sys.path.insert(0,str(PROJECT))
import utils

KEYS=['TI_f1','TC_f1','AI_f1','AC_f1','EC_I_f1','EC_C_f1',
      'TI_p','TC_p','AI_p','AC_p','EC_I_p','EC_C_p',
      'TI_r','TC_r','AI_r','AC_r','EC_I_r','EC_C_r']

def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def convert_events(events):
    return [{'event_type':e['type'], 'trigger':{'tokens':e['trigger'].split()},
             'arguments':[{'argument_type':a['role'],'tokens':a['text'].split()} for a in e['args']]}
            for e in events]

def merge_predictions(rows,predictions):
    by_id={p['id']:p for p in predictions}
    if len(by_id)!=len(predictions) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Duplicate sentence ID')
    if set(by_id)!={r['id'] for r in rows}:
        raise ValueError('Prediction IDs do not exactly cover sentence gold')
    grouped={}
    for r in rows: grouped.setdefault(r['document_id'],[]).append(r)
    result={}; stats=Counter()
    for doc,group in grouped.items():
        ordered=sorted(group,key=lambda r:r['sentence_index'])
        if [r['sentence_index'] for r in ordered]!=list(range(len(group))):
            raise ValueError('Incomplete sentence indexes')
        result[doc]=[]
        for r in ordered:
            events,diag=parse_forest(by_id[r['id']]['raw_output'])
            stats.update({k:v for k,v in diag.items() if isinstance(v,int)})
            result[doc].extend(convert_events(events))
    stats['events']=sum(len(es) for es in result.values())
    stats['arguments']=sum(len(e['arguments']) for es in result.values() for e in es)
    return result,dict(stats)

def save(path,value):
    content=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text()!=content:
        raise FileExistsError(f'Refusing overwrite: {path}')
    path.write_text(content)

def main():
    sentence_path=STAGE/'data/sentence-c-appendix-no-copy-20261002/test.jsonl'
    gold_path=PROJECT/'data/SciEvents/test.json'
    rows=read(sentence_path)
    gold_docs=json.loads(gold_path.read_text())
    golds=[d['events'] for d in gold_docs]
    sources={'gold':gold_path,'sentence_data':sentence_path,'scorer':PROJECT/'utils.py',
             'parser':STAGE/'evaluation-v2-20261003/scripts/linearize.py'}
    variants={'baseline':STAGE/'runs/C-AppendixNoCopy-42-20261002/checkpoint-eval/test-checkpoint-2406.predictions.jsonl',
              'curriculum':STAGE/'runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003/checkpoint-eval/test-checkpoint-2406.predictions.jsonl'}
    sources.update(variants)
    before={str(p):sha(p) for p in sources.values()}
    metrics={}
    for name,path in variants.items():
        docs,diag=merge_predictions(rows,read(path))
        if set(docs)!={d['doc_id'] for d in gold_docs}:
            raise ValueError('Document IDs mismatch official gold')
        predictions=[{'doc_id':d['doc_id'],'events':docs[d['doc_id']]} for d in gold_docs]
        save(HERE/f'{name}.token.predictions.json',predictions)
        values=utils.calculate_F1(golds,[d['events'] for d in predictions],'token')
        metrics[name]={'official':dict(zip(KEYS,values)), 'diagnostics':diag,
                       'sentence_predictions':len(rows),'documents':len(predictions),'checkpoint':2406}
    after={str(p):sha(p) for p in sources.values()}
    if after!=before: raise ValueError('Source changed during scoring')
    result={'mode':'official token, raw parser output, no alignment or deduplication',
            'source_hashes':before,'variants':metrics,
            'limitation':'Official duplicate trigger text dictionary collisions retained; token/offset deltas are not pure alignment errors.'}
    save(HERE/'results.json',result)
    print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
