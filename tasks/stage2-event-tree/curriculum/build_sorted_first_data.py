"""Freeze an all-train easy-to-hard first epoch without changing any sample bytes."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

HERE=Path(__file__).resolve().parent
STAGE=HERE.parent
DEFAULT_OUTPUT=HERE/'sorted-first-20261003'
BASELINE=STAGE/'runs/C-AppendixNoCopy-42-20261002'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def order_rows(rows,scores,seed):
    ids=[r['id'] for r in rows]
    if len(set(ids))!=len(ids) or set(scores)!=set(ids):
        raise ValueError('Scores must cover unique original row IDs exactly')
    groups=defaultdict(list);roots=[]
    for r in rows:
        score=scores[r['id']]
        if r['n_events']==0:
            if score is not None:raise ValueError('ROOT must have no difficulty')
            roots.append(r)
        else:
            if score is None or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError('Invalid positive difficulty')
            groups[score].append(r)
    rng=random.Random(seed);positive=[]
    for score in sorted(groups):
        group=groups[score][:];rng.shuffle(group);positive.extend(group)
    random.Random(seed).shuffle(roots)
    # Equally spaced ROOT slots, leaving positive difficulty sequence unchanged.
    positions={math.floor((i+.5)*len(rows)/len(roots)) for i in range(len(roots))} if roots else set()
    if len(positions)!=len(roots):raise ValueError('ROOT placement collision')
    p=iter(positive);z=iter(roots)
    return [next(z) if i in positions else next(p) for i in range(len(rows))]

def save_new(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise FileExistsError(f'Refusing overwrite: {path}')
        return
    path.write_bytes(data)

def build(output=DEFAULT_OUTPUT):
    output=Path(output).resolve(); baseline=json.loads((BASELINE/'run.json').read_text())
    cfg=json.loads((BASELINE/'training/config.json').read_text())
    original=Path(baseline['data_dir'])/'train.jsonl'
    lines=original.read_bytes().splitlines(keepends=True);rows=[json.loads(l) for l in lines]
    if sha(original)!=baseline['audit']['data_sha256'] or len(rows)!=12828:
        raise ValueError('Frozen baseline data mismatch')
    assignment=STAGE/'difficulty-analysis/mean_percentile_difficulty_assignments.csv'
    with assignment.open(newline='') as f:
        selected=[x for x in csv.DictReader(f) if float(x['fraction'])==.5 and x['tie_policy']=='all_ties_easy']
    if len(selected)!=len(rows):raise ValueError('Assignment count mismatch')
    scores={r['sample_id']:None if r['score_mean']=='NA' else float(r['score_mean']) for r in selected}
    ordered=order_rows(rows,scores,42)
    line_by_id={r['id']:l for r,l in zip(rows,lines)}
    content=b''.join(line_by_id[r['id']] for r in ordered)
    output.mkdir(parents=True,exist_ok=True)
    data_path=output/'train-sorted.jsonl';ids_path=output/'ordered-ids.txt';prompt_path=output/'prompt.md'
    prompt=(HERE/'prompt_template.txt').read_bytes()
    if hashlib.sha256(prompt).hexdigest()!='2bcc66773a71904973ae4f85b865e60eddfacf4230515618576d3217e6776b41':
        raise ValueError('Original prompt differs')
    save_new(data_path,content)
    save_new(ids_path,''.join(r['id']+'\n' for r in ordered).encode())
    save_new(prompt_path,prompt)
    rebuilt=[json.loads(x) for x in data_path.read_bytes().splitlines()]
    if {r['id']:r for r in rebuilt}!={r['id']:r for r in rows}:raise ValueError('Any sample changed')
    if any(line_by_id[r['id']]!=line for r,line in zip(rebuilt,data_path.read_bytes().splitlines(keepends=True))):
        raise ValueError('Sample byte changes')
    eff=cfg['microbatch']*cfg['gradient_accumulation'];per=math.ceil(len(rows)/eff)
    total=json.loads((BASELINE/'training/trainer_state.json').read_text())['global_step']
    if (eff,per,total)!=(16,802,4010) or cfg['init_adapter'] is not None:
        raise ValueError('Unexpected baseline budget or initialization')
    source=[original,assignment,BASELINE/'run.json',BASELINE/'training/config.json',BASELINE/'training/trainer_state.json',HERE/'prompt_template.txt',Path(__file__).resolve()]
    source.extend(Path(baseline['data_dir'])/f'{s}.jsonl' for s in ['dev','test'])
    manifest={'experiment':'C-CL-Mean-AppendixNoCopy-Sorted1','seed':42,
        'baseline_run_file':str(BASELINE/'run.json'),'data_dir':baseline['data_dir'],
        'source_train_file':str(original),'sorted_train_file':str(data_path),'ordered_ids_file':str(ids_path),
        'prompt_template_file':str(prompt_path),'prompt_sha256':sha(prompt_path),
        'total_optimizer_steps':total,'effective_batch_size':eff,'steps_per_epoch':per,
        'checkpoint_steps':[802,1604,2406,3208,4010], 'train_count':len(rows),
        'positive_count':sum(r['n_events']>0 for r in rows),'root_count':sum(r['n_events']==0 for r in rows),
        'initialization':cfg['model_path'],'epochs':5,
        'sampling':'epoch0 frozen ascending positive difficulty with seeded ties/ROOT evenly interleaved; epochs1-4 seeded random permutations',
        'root_policy':'floor((i+0.5)*N/N_ROOT); ROOT IDs shuffled independently using seed42',
        'difficulty_formula':'0.5*((1-1/N_events)+(1-mean(type-local role-completeness percentile)))',
        'difficulty_is_proxy':True,'source_hashes':{str(p.resolve()):sha(p) for p in source},
        'output_hashes':{str(p):sha(p) for p in [data_path,ids_path,prompt_path]},
        'verification':{'all_original_rows_byte_identical':True,'full_id_coverage':True,'positive_score_monotonic':True,'ROOT_no_difficulty':True}}
    save_new(output/'manifest.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode())
    pos_scores=[scores[r['id']] for r in rebuilt if r['n_events']]
    assert pos_scores==sorted(pos_scores)
    gaps=Counter(b-a for a,b in zip([i for i,r in enumerate(rebuilt) if not r['n_events']],[i for i,r in enumerate(rebuilt) if not r['n_events']][1:]))
    audit={'status':'passed','train_count':len(rows),'ROOT_count':manifest['root_count'],
           'ROOT_slot_gaps':dict(gaps),'positive_monotonic':True,'all_samples_byte_identical':True,
           'first_epoch_updates':per,'later_epoch_updates':total-per,'prompt_sha256':sha(prompt_path)}
    save_new(output/'data-verification.json',(json.dumps(audit,indent=2)+'\n').encode())
    print(json.dumps(audit,indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    build(p.parse_args().output)
