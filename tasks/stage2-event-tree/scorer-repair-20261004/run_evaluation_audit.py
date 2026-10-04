"""Reproduce official results and compare instance-safe metrics without training."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
PROJECT=STAGE.parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(STAGE/'curriculum'))
import utils
from verify_sorted_first_run import fixed_evaluation_modules

KEYS=['TI_f1','TC_f1','AI_f1','AC_f1','EC_I_f1','EC_C_f1','TI_p','TC_p','AI_p','AC_p','EC_I_p','EC_C_p','TI_r','TC_r','AI_r','AC_r','EC_I_r','EC_C_r']
VARIANTS={'baseline':STAGE/'runs/C-AppendixNoCopy-42-20261002',
          'Easy2':STAGE/'runs/C-CL-Mean-AppendixNoCopy-Easy2-42-20261003',
          'Sorted1':STAGE/'runs/C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003'}

def read(path):return json.loads(Path(path).read_text())
def jsonl(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    value=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if path.exists() and path.read_text()!=value:raise FileExistsError(f'Refusing conflicting result: {path}')
    path.write_text(value)

def order_documents(gold,documents):
    ids=[x['doc_id'] for x in documents];expected=[x['doc_id'] for x in gold]
    if len(set(ids))!=len(ids) or set(ids)!=set(expected) or len(expected)!=len(set(expected)):
        raise ValueError('Prediction/gold document IDs mismatch')
    indexed={x['doc_id']:x['events'] for x in documents}
    return [indexed[x] for x in expected]

def verify_hash_sets(*sets):
    for expected in sets:
        for file, digest in expected.items():
            if sha(file) != digest: raise ValueError('Source modified: ' + file)

def assert_grounded_mentions(documents, event_lists):
    if len(documents) != len(event_lists): raise ValueError('Document count mismatch')
    for doc, events in zip(documents, event_lists):
        tokens = doc['document']
        for event in events:
            for mention in [event['trigger'], *event['arguments']]:
                offsets = mention['offsets']
                if not offsets or any(type(i) is not int or i < 0 or i >= len(tokens) for i in offsets):
                    raise ValueError('Invalid mention offsets')
                if mention['tokens'] != [tokens[i] for i in offsets]:
                    raise ValueError('Mention text and source positions disagree')

def residual_text_matches(gold,pred):
    """Diagnostic only: after exact position 1:1 pairing, show extra text matches."""
    by_offset=defaultdict(list)
    for i,e in enumerate(pred):by_offset[tuple(e['trigger']['offsets'])].append(i)
    remaining_gold=[];used=set()
    for gi,g in enumerate(gold):
        candidates=by_offset[tuple(g['trigger']['offsets'])]
        if candidates:used.add(candidates.pop(0))
        else:remaining_gold.append(gi)
    by_text=defaultdict(list)
    for pi,p in enumerate(pred):
        if pi not in used:by_text[tuple(p['trigger']['tokens'])].append(pi)
    matches=[]
    for gi in remaining_gold:
        g=gold[gi];candidates=by_text[tuple(g['trigger']['tokens'])]
        if not candidates:continue
        pi=min(candidates,key=lambda i:abs(pred[i]['trigger']['offsets'][0]-g['trigger']['offsets'][0]));candidates.remove(pi)
        matches.append({'gold_index':gi,'prediction_index':pi,'gold_offsets':g['trigger']['offsets'],
                        'prediction_offsets':pred[pi]['trigger']['offsets'],'text':' '.join(g['trigger']['tokens']),
                        'gold_type':g['event_type'],'prediction_type':pred[pi]['event_type']})
    return matches

def raw_documents(rows,predictions,parser):
    ids=[p['id'] for p in predictions]
    if len(set(ids))!=len(ids) or set(ids)!={r['id'] for r in rows}:raise ValueError('Sentence prediction IDs mismatch')
    by_id={p['id']:p for p in predictions};grouped=defaultdict(list)
    for r in rows:grouped[r['document_id']].append(r)
    result=[]
    for doc,group in grouped.items():
        events=[]
        for row in sorted(group,key=lambda r:r['sentence_index']):
            parsed,_=parser.parse_forest(by_id[row['id']]['raw_output'])
            events.extend({'event_type':e['type'],'trigger':{'tokens':e['trigger'].split()},
                           'arguments':[{'tokens':a['text'].split(),'argument_type':a['role']} for a in e['args']]}
                          for e in parsed)
        result.append({'doc_id':doc,'events':events})
    return result

def official(g,p,mode):return dict(zip(KEYS,map(float,utils.calculate_F1(g,p,mode))))
def compact(score):return {k:v for k,v in score.items() if k!='matching'}
def assert_scores(a,b,label):
    if a.keys()!=b.keys() or any(abs(a[k]-b[k])>1e-12 for k in a):raise ValueError('Score mismatch: '+label)

def main():
    from instance_scorer import score_documents
    historical=read(HERE/'historical-source-locks.json')
    verify_hash_sets(historical)
    if Path(utils.__file__).resolve()!=PROJECT/'utils.py':raise ValueError('Unexpected official scorer')
    _,merge=fixed_evaluation_modules();parser=sys.modules['linearize']
    variants={};locks={str(p):sha(p) for p in [PROJECT/'utils.py',HERE/'instance_scorer.py',Path(__file__).resolve()]};examples=[];trace={};selections={}
    for name,run in VARIANTS.items():
        state=read(run/'run.json');locks[str(run/'run.json')]=sha(run/'run.json')
        variants[name]={'dev':{},'test':{}}
        for split,steps in [('dev',sorted(x['step'] for x in state['dev_checkpoints'])),('test',[2406])]:
            gold_path=PROJECT/'data/SciEvents'/f'{split}.json';gold=read(gold_path);g=[x['events'] for x in gold]
            sentence_path=STAGE/'data/sentence-c-appendix-no-copy-20261002'/f'{split}.jsonl';rows=jsonl(sentence_path)
            locks.update({str(p):sha(p) for p in [gold_path,sentence_path]})
            for step in steps:
                pred_path=run/'checkpoint-eval'/f'{split}-checkpoint-{step}.predictions.jsonl'
                metric_path=(STAGE/'evaluation-v2-20261003/results/baseline'/f'{split}-checkpoint-{step}.metrics.json') if name=='baseline' else run/'checkpoint-eval'/f'{split}-checkpoint-{step}.metrics.json'
                saved=read(metric_path);predictions=jsonl(pred_path)
                # Same repaired merge as historical official pipeline, including existing deduplication.
                merged=merge.merge_sentence_rows(rows,merge.reparse_raw_predictions(predictions))
                p=order_documents(gold,[{'doc_id':x['document_id'],'events':x['aligned_events']} for x in merged])
                assert_grounded_mentions(gold,g);assert_grounded_mentions(gold,p)
                raw=order_documents(gold,raw_documents(rows,predictions,parser))
                old_offset=official(g,p,'offset');assert_scores(old_offset,saved['official'],f'{name}/{split}/{step}')
                new_offset=score_documents(g,p,'offset');new_token=score_documents(g,p,'token')
                raw_token=score_documents(g,raw,'token')
                variants[name][split][str(step)]={'official_offset':old_offset,'official_aligned_token':official(g,p,'token'),
                    'official_raw_token':official(g,raw,'token'),'instance_offset':compact(new_offset),
                    'instance_aligned_token':compact(new_token),'instance_raw_token':compact(raw_token),
                    'gold_events':sum(map(len,g)),'aligned_events':sum(map(len,p)),'raw_events':sum(map(len,raw))}
                locks.update({str(q):sha(q) for q in [pred_path,metric_path]})
                if split=='test':
                    matches=[];affected=0
                    for doc,gg,pp in zip(gold,g,p):
                        extra=residual_text_matches(gg,pp)
                        if extra:affected+=1
                        bounds=[];start=0
                        for sentence in doc['sentences']:
                            end=start+len(sentence.split());bounds.append((start,end));start=end
                        def sent(position):return next(i for i,(lo,hi) in enumerate(bounds) if lo<=position<hi)
                        for e in extra:
                            e.update(doc_id=doc['doc_id'],gold_sentence=sent(e['gold_offsets'][0]),prediction_sentence=sent(e['prediction_offsets'][0]))
                            matches.append(e)
                            if len([x for x in examples if x['variant']==name])<5:
                                examples.append({'variant':name,'kind':'extra_token_TI_match_at_different_position',
                                    'match':e,'title':doc['title'],'abstract':doc['abstract'],
                                    'gold_event':gg[e['gold_index']],'prediction_event':pp[e['prediction_index']]})
                    off_tp=new_offset['counts']['TI']['tp'];tok_tp=new_token['counts']['TI']['tp']
                    if tok_tp-off_tp!=len(matches):raise ValueError('Residual TI decomposition mismatch')
                    trace[name]={'same_prediction_set_instance_TI_offset_tp':off_tp,'token_tp':tok_tp,
                        'extra_same_text_different_position_matches':len(matches),'affected_documents':affected,
                        'cross_sentence_matches':sum(e['gold_sentence']!=e['prediction_sentence'] for e in matches),
                        'matches':matches}
                    write(HERE/f'{name}.aligned.predictions.json',[{'doc_id':d['doc_id'],'events':es} for d,es in zip(gold,p)])
                    write(HERE/f'{name}.raw-token.predictions.json',[{'doc_id':d['doc_id'],'events':es} for d,es in zip(gold,raw)])
        for field in ['official_offset','instance_offset','instance_aligned_token','instance_raw_token']:
            dev=variants[name]['dev']
            best=max(dev,key=lambda step:((dev[step][field]['metrics'] if field.startswith('instance') else dev[step][field])['AC_f1'],int(step)))
            selections.setdefault(name,{})[field]=int(best)
        print(name,'complete',flush=True)
    verify_hash_sets(historical,locks)
    write(HERE/'evaluation-results.json',{'status':'completed','source_hashes':locks,'historical_sources_unchanged':True,
        'variants':variants,'TI_mode_decomposition':trace,'dev_selections':selections,
        'policy':'instance one-to-one scorer; alternate diagnostic metric, not official benchmark',
        'limitations':['Token exact text does not identify mention position.','No gold used to realign actual predictions.','Historical dev-best test checkpoints remain2406; new dev choices recorded separately.']})
    write(HERE/'actual-examples.json',examples)

if __name__=='__main__':main()
