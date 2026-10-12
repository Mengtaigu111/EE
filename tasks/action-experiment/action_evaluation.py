"""Action parsing, presence reward and full-document dev/test evaluation.

All parsed predictions survive scoring even when no source occurrence is found.
Malformed outputs get an unmatched invalid event in corpus metrics; they never
become a falsely valid empty prediction. Reward parsing failures always score 0.
"""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import math
import sys
import time

from action_common import (ROOT,DATA_DIR,MAX_NEW_TOKENS,append_jsonl,load_module,
                           read_json,read_rows,write_json,generation_arguments)
sys.path.insert(0,str(ROOT/'scripts'))
from action_language import load_schema,parse_actions,render_tree

presence=load_module('_action_presence',ROOT/'scripts/score_tree_presence.py')


@lru_cache(maxsize=8)
def schema_from(path=str(DATA_DIR/'prompt_template.txt')):
    return load_schema(path,ROOT/'data/SciEvents/ontology.json')


def to_scievents(events):
    return [dict(event_type=e['type'],trigger={'tokens':e['trigger'].split()},
                 arguments=[dict(argument_type=a['role'],tokens=a['text'].split()) for a in e['arguments']])
            for e in events]


def parse_prediction(raw_output,schema=None):
    try:
        events=parse_actions(raw_output,schema or schema_from())
        return events,None
    except (ValueError,TypeError) as error:
        return [],str(error)


def reward_action(raw_output,row,*,truncated=False):
    events,error=parse_prediction(raw_output)
    result=dict(reward=0.0,TC_f1=0.0,TI_f1=0.0,type_count_f1=0.0,
                parse_valid=error is None,parse_error=error,truncated=bool(truncated),predicted_events=len(events))
    if error is not None:return result
    gold=row['target_events'];pred=to_scievents(events)
    if not gold and not pred:
        result.update(reward=1.0,TC_f1=1.0,TI_f1=1.0,type_count_f1=1.0)
        return result
    score=presence.score_documents([gold],[pred])['metrics']
    gc=Counter(e['event_type'] for e in gold);pc=Counter(e['event_type'] for e in pred)
    denominator=sum(gc.values())+sum(pc.values())
    count=2*sum((gc&pc).values())/denominator if denominator else 1.0
    result.update(TC_f1=score['TC_f1'],TI_f1=score['TI_f1'],type_count_f1=count,
                  reward=math.fsum((.6*score['TC_f1'],.3*score['TI_f1'],.1*count)))
    return result


def ordered_match(text,tokens):
    """Conservative positional diagnostic only; never deletes or alters predictions."""
    needle=text.split()
    if not needle:return False
    index=0
    for token in tokens:
        if token==needle[index]:
            index+=1
            if index==len(needle):return True
    return False


def score_predictions(rows,predictions):
    ids=[r['id'] for r in rows];pids=[p['id'] for p in predictions]
    if not rows or len(set(ids))!=len(ids) or len(set(pids))!=len(pids) or set(ids)!=set(pids):
        raise ValueError('Predictions must cover unique row IDs exactly')
    by_id={p['id']:p for p in predictions}
    golds=defaultdict(list);preds=defaultdict(list)
    diagnostics=Counter();parsed=[]
    for row in rows:
        p=by_id[row['id']]
        events,error=parse_prediction(p['raw_output'])
        key=row['document_id'];golds[key].extend(row['target_events'])
        diagnostics['rows']+=1
        diagnostics['truncated']+=int(p.get('truncated',False))
        if error:
            diagnostics['parse_errors']+=1
            # Cannot overlap source tokens. It remains an explicit FP.
            preds[key].append(dict(event_type='__INVALID__',trigger={'tokens':['\x00invalid:'+row['id']]},arguments=[]))
        else:
            preds[key].extend(to_scievents(events))
            diagnostics['predicted_events']+=len(events)
            for e in events:
                diagnostics['triggers_without_ordered_target_match']+=int(not ordered_match(e['trigger'],row['target_sentence'].split()))
                for a in e['arguments']:
                    diagnostics['arguments_without_ordered_abstract_match']+=int(not ordered_match(a['text'],row['tokens']))
        parsed.append(dict(id=row['id'],events=events,compact_tree=render_tree(events) if error is None else None,parse_error=error))
    keys=list(golds)
    result=presence.score_documents([golds[k] for k in keys],[preds[k] for k in keys])
    for name in ('parse_errors','truncated','predicted_events','triggers_without_ordered_target_match','arguments_without_ordered_abstract_match'):
        diagnostics.setdefault(name,0)
    result.update(mode='hierarchical text presence; counterpart reuse allowed',
                  diagnostics=dict(diagnostics),parsed_predictions=parsed,documents=len(keys))
    return result


def evaluate_model(model,tokenizer,rows,output_dir,*,max_new_tokens=MAX_NEW_TOKENS,batch_size=4):
    import torch
    from transformers import GenerationConfig
    if max_new_tokens<1 or batch_size<1:raise ValueError('Positive generation limits required')
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();was_training=model.training;old_padding=tokenizer.padding_side
    tokenizer.padding_side='left';model.eval();predictions=[]
    eos=model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids={eos} if isinstance(eos,int) else set(eos)
    config=GenerationConfig(do_sample=False,max_new_tokens=max_new_tokens,
                            eos_token_id=eos,pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id,use_cache=True)
    try:
        for start in range(0,len(rows),batch_size):
            chunk=rows[start:start+batch_size]
            prompts=[tokenizer.apply_chat_template([{'role':'user','content':r['prompt']}],
                     tokenize=False,add_generation_prompt=True,enable_thinking=False) for r in chunk]
            batch=tokenizer(prompts,return_tensors='pt',padding=True,add_special_tokens=False,truncation=False)
            device=next(model.parameters()).device
            batch={key:value.to(device) for key,value in batch.items()}
            with torch.inference_mode():output=model.generate(**batch,**generation_arguments(config))
            for row,tokens in zip(chunk,output[:,batch['input_ids'].shape[1]:].tolist()):
                stop=next((i+1 for i,t in enumerate(tokens) if t in eos_ids),len(tokens))
                completion=tokens[:stop]
                record=dict(id=row['id'],document_id=row['document_id'],sentence_index=row['sentence_index'],
                            raw_output=tokenizer.decode(completion,skip_special_tokens=True),
                            completion_token_ids=completion,truncated=not any(t in eos_ids for t in completion))
                predictions.append(record);append_jsonl(output_dir/'predictions.jsonl',record)
    finally:
        tokenizer.padding_side=old_padding;model.train(was_training)
    result=score_predictions(rows,predictions)
    for p in result.pop('parsed_predictions'):append_jsonl(output_dir/'parsed.jsonl',p)
    result['wall_seconds']=time.monotonic()-started
    result['max_new_tokens']=max_new_tokens
    write_json(output_dir/'metrics.json',result)
    return result
