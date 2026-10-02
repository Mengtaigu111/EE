#!/usr/bin/env python3
"""Train-only gold role completeness against Appendix F.2."""
import argparse, csv, json, math, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
TRAIN=ROOT/'tasks/stage2-event-tree/data/sentence-c/train.jsonl'
SCHEMA=ROOT/'task/scievents_appendix_f2_exact_reference_v2.md'
OUT=ROOT/'tasks/stage2-event-tree/difficulty-analysis'
TYPES=('PUR','PRP','WKS','ITT','RWS','RWF','MDS','FIN','CMP','FAC')

def read_schema(path):
    text=Path(path).read_text(encoding='utf8')
    data={}
    lines=text.splitlines()
    fence=chr(96)*3
    for i,line in enumerate(lines):
        m=re.match(r'^### ([A-Z]+) — ',line)
        if not m: continue
        typ=m.group(1)
        try:
            role_label=lines.index('Roles:',i+1)
            start=lines.index(fence+'text',role_label+1)
            end=lines.index(fence,start+1)
        except ValueError:
            continue
        data[typ]=tuple(x.strip() for x in lines[start+1:end] if x.strip())
    if set(data)!=set(TYPES): raise ValueError(f'Appendix F.2 schema mismatch: {data.keys()}')
    return {t:data[t] for t in TYPES}

def event_completeness(event, schemas):
    legal=set(schemas[event['event_type']])
    observed={a['argument_type'] for a in event.get('arguments',[])}
    invalid=sorted(observed-legal)
    valid=observed&legal
    return {'valid':valid,'invalid':invalid,'missing':legal-valid,'score':len(valid)/len(legal)}

def quantile(v,q):
    if not v:return None
    x=sorted(v); p=(len(x)-1)*q; lo=math.floor(p); hi=math.ceil(p)
    return x[lo]+(x[hi]-x[lo])*(p-lo)

def percentile(v,x): return (sum(a<x for a in v)+.5*sum(a==x for a in v))/len(v)

def sentence_summary(completenesses):
    if not completenesses:
        return {'event_count':0,'min_schema_completeness':None,
                'mean_schema_completeness':None,'max_schema_completeness':None}
    return {'event_count':len(completenesses),'min_schema_completeness':min(completenesses),
            'mean_schema_completeness':sum(completenesses)/len(completenesses),
            'max_schema_completeness':max(completenesses)}

def write_csv(path,rows,fields=None):
    rows=list(rows)
    with open(path,'w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or (list(rows[0]) if rows else []),extrasaction='ignore');w.writeheader();w.writerows(rows)

def stats(v):
    return {'mean':statistics.mean(v) if v else None,'median':statistics.median(v) if v else None,
      'std':statistics.pstdev(v) if v else None,'min':min(v) if v else None,'max':max(v) if v else None,
      'p10':quantile(v,.1),'p25':quantile(v,.25),'p50':quantile(v,.5),'p75':quantile(v,.75),'p90':quantile(v,.9)}

def analyze(train=TRAIN,schema=SCHEMA,outdir=OUT):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);schemas=read_schema(schema)
    samples=[json.loads(x) for x in Path(train).read_text(encoding='utf8').splitlines() if x.strip()]
    events=[];sentences=[];invalid=[];bytype=defaultdict(list);type_n=Counter();role_n=Counter();patterns=Counter()
    for s in samples:
        local=[]
        for i,e in enumerate(s.get('target_events',[])):
            t=e['event_type']
            if t not in schemas:raise ValueError(f'Unknown event type {t} in {s["id"]}')
            z=event_completeness(e,schemas); trig=e.get('trigger',{}).get('text','')
            if z['invalid']:invalid.append({'sample_id':s['id'],'abstract_id':s['document_id'],'target_sentence_id':s.get('sentence_index'),'event_index':i,'event_type':t,'trigger_text':trig,'invalid_roles':'|'.join(z['invalid'])})
            obs=[r for r in schemas[t] if r in z['valid']];miss=[r for r in schemas[t] if r in z['missing']]
            row={'sample_id':s['id'],'abstract_id':s['document_id'],'target_sentence_id':s.get('sentence_index'),'event_index':i,'event_type':t,'trigger_text':trig,'legal_roles':'|'.join(schemas[t]),'observed_roles':'|'.join(obs),'missing_roles':'|'.join(miss),'num_legal_roles':len(schemas[t]),'num_observed_roles':len(z['valid']),'schema_completeness':z['score']}
            events.append(row);local.append(row);bytype[t].append(z['score']);type_n[t]+=1;patterns[(t,tuple(obs))]+=1
            for r in z['valid']:role_n[(t,r)]+=1
        ss=sentence_summary([r['schema_completeness'] for r in local])
        sentences.append({'sample_id':s['id'],'abstract_id':s['document_id'],'target_sentence_id':s.get('sentence_index'),'event_count':len(local),
          'min_schema_completeness':ss['min_schema_completeness'],'mean_schema_completeness':ss['mean_schema_completeness'],'max_schema_completeness':ss['max_schema_completeness']})
    write_csv(outdir/'invalid_role_report.csv',invalid)
    if invalid:raise ValueError(f'{len(invalid)} invalid-role events; see invalid_role_report.csv')
    for r in events:r['within_type_completeness_percentile']=percentile(bytype[r['event_type']],r['schema_completeness'])
    write_csv(outdir/'schema_completeness_event_level.csv',events)
    write_csv(outdir/'schema_completeness_sentence_level.csv',sentences)
    role_rows=[];missing_rows=[]
    for t in TYPES:
      n=type_n[t]
      for role in schemas[t]:
        p=role_n[(t,role)];role_rows.append({'event_type':t,'role':role,'event_count_of_type':n,'role_present_count':p,'role_presence_rate':p/n})
        missing_rows.append({'event_type':t,'role':role,'missing_count':n-p,'total_events':n,'missing_rate':(n-p)/n})
    write_csv(outdir/'role_presence_by_event_type.csv',role_rows);write_csv(outdir/'missing_roles_by_event_type.csv',missing_rows)
    patrows=[]
    for t in TYPES:
      subset=sorted(((rs,n) for (tt,rs),n in patterns.items() if tt==t),key=lambda a:(-a[1],a[0]))
      for rank,(rs,n) in enumerate(subset[:10],1):
        patrows.append({'event_type':t,'rank':rank,'observed_role_set':'{'+', '.join(rs)+'}','events':n,'percent_within_type':n/type_n[t],'completeness':len(rs)/len(schemas[t])})
    write_csv(outdir/'role_set_patterns_by_event_type.csv',patrows)
    vals=[e['schema_completeness'] for e in events];pos=[s for s in sentences if s['event_count']>0];mins=[s['min_schema_completeness'] for s in pos]
    q1,q2=quantile(mins,1/3),quantile(mins,2/3);strata=defaultdict(list);ec=Counter()
    for s in sentences:ec['3+' if s['event_count']>=3 else str(s['event_count'])]+=1
    for s in pos:
      b='Low' if s['min_schema_completeness']<=q1 else ('Medium' if s['min_schema_completeness']<=q2 else 'High')
      c='3+' if s['event_count']>=3 else str(s['event_count']);strata[c,b].append(s)
    exact_by_count={}
    for c in ('1','2','3+'):
      grouped=Counter(s['min_schema_completeness'] for s in pos
          if ('3+' if s['event_count']>=3 else str(s['event_count']))==c)
      denominator=ec[c]
      exact_by_count[c]=[{'min_completeness':v,'sentences':n,'percent_within_group':n/denominator}
                         for v,n in sorted(grouped.items())]
    type_summary=[];percentiles_by_type=defaultdict(list)
    for row in events:percentiles_by_type[row['event_type']].append(row['within_type_completeness_percentile'])
    for t in TYPES:
      v=bytype[t];n=sum(x==1 for x in v);type_summary.append({'event_type':t,'events':len(v),**stats(v),'mean_within_type_percentile':sum(percentiles_by_type[t])/len(v),'full_schema_count':n,'full_schema_percent':n/len(v)})
    full=sum(x==1 for x in vals)
    summary={'train_samples':len(samples),'zero_event_sentences':ec['0'],'positive_sentences':len(pos),'total_gold_events':len(events),'per_event_type_counts':dict(type_n),'event_count_sentence_distribution':dict(ec),
      'overall':{**stats(vals),'total_events':len(vals),'full_schema_count':full,'full_schema_percent':full/len(vals)},
      'per_type':type_summary,'exact_completeness_distribution':[{'completeness':x,'events':n,'percent':n/len(vals)} for x,n in sorted(Counter(vals).items())],
      'sentence_min_distribution':[{'min_completeness':x,'sentences':n,'percent':n/len(mins)} for x,n in sorted(Counter(mins).items())],
      'sentence_min_distribution_by_event_count':exact_by_count,
      'quantile_cuts':{'q1':q1,'q2':q2},
      'event_count_x_min_completeness':{c:{b:{'count':len(strata[c,b]),'percent':len(strata[c,b])/ec[c]} for b in ('High','Medium','Low')} for c in ('1','2','3+')},
      'schemas':{t:list(schemas[t]) for t in TYPES}}
    assert sum(type_n.values())==len(events)
    assert all(0<=e['schema_completeness']<=1 and e['num_observed_roles']<=e['num_legal_roles'] for e in events)
    (outdir/'schema_completeness_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    plots(vals,bytype,sentences,outdir)
    report(outdir/'schema_completeness_train_report.md',summary,role_rows,missing_rows,patrows)
    return summary

def plots(vals,bytype,sentences,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from collections import Counter
    u=sorted(set(vals));c=Counter(vals)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([f'{x:.3f}' for x in u],[c[x] for x in u]);ax.set(xlabel='Schema completeness',ylabel='Gold events',title='Train gold event schema completeness');fig.tight_layout();fig.savefig(out/'overall_schema_completeness_distribution.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.boxplot([bytype[t] for t in TYPES],showfliers=False);ax.set_xticks(range(1,len(TYPES)+1),TYPES);ax.set_ylim(0,1.03);ax.set(ylabel='Schema completeness',title='Train schema completeness by event type');fig.tight_layout();fig.savefig(out/'schema_completeness_by_event_type.png',dpi=160);plt.close(fig)
    gs={'1':[],'2':[],'3+':[]}
    for s in sentences:
      n=s['event_count']
      if n:gs['3+' if n>=3 else str(n)].append(s['min_schema_completeness'])
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([gs[x] for x in ('1','2','3+')],showfliers=False);ax.set_xticks([1,2,3],['1','2','3+']);ax.set_ylim(0,1.03);ax.set(xlabel='Gold events per sentence',ylabel='Minimum completeness',title='Sentence min completeness by event count');fig.tight_layout();fig.savefig(out/'event_count_vs_min_completeness.png',dpi=160);plt.close(fig)

def pct(x):return f'{100*x:.2f}%'
def report(path,s,roles,missing,pats):
    o=s['overall'];L=['# Train Appendix F.2 Schema Completeness','',
    '## Q1. Overall gold event distribution','',
    f"Train has {s['train_samples']:,} target sentences and {o['total_events']:,} gold events. Mean {o['mean']:.4f}; median {o['median']:.4f}; SD {o['std']:.4f}; P25/P75 {o['p25']:.4f}/{o['p75']:.4f}; P10/P90 {o['p10']:.4f}/{o['p90']:.4f}. Full schema: {o['full_schema_count']:,} ({pct(o['full_schema_percent'])}).",'',
    '| Completeness | Events | Percent |','|---:|---:|---:|']
    L += [f"| {r['completeness']:.4f} | {r['events']} | {pct(r['percent'])} |" for r in s['exact_completeness_distribution']]
    L += ['', '## Q2. By event type','','| Type | Events | Mean | Median | SD | P25 | P75 | Full | Full % |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in s['per_type']:L.append(f"| {r['event_type']} | {r['events']} | {r['mean']:.4f} | {r['median']:.4f} | {r['std']:.4f} | {r['p25']:.4f} | {r['p75']:.4f} | {r['full_schema_count']} | {pct(r['full_schema_percent'])} |")
    L += ['', '## Q3. Most common role-set realizations (Top 10/type)','','| Type | Rank | Observed roles | Events | Within type | Completeness |','|---|---:|---|---:|---:|---:|']
    for r in pats:L.append(f"| {r['event_type']} | {r['rank']} | {r['observed_role_set']} | {r['events']} | {pct(r['percent_within_type'])} | {pct(r['completeness'])} |")
    L += ['', '## Q4. Role presence and missingness','','| Type | Role | Present | Present % | Missing | Missing % |','|---|---|---:|---:|---:|---:|']
    mm={(r['event_type'],r['role']):r for r in missing}
    for r in roles:
      m=mm[r['event_type'],r['role']];L.append(f"| {r['event_type']} | {r['role']} | {r['role_present_count']} | {pct(r['role_presence_rate'])} | {m['missing_count']} | {pct(m['missing_rate'])} |")
    L += ['', '## Q5. Sentence-level minimum completeness','','Zero-event sentences have NA completeness. The following gives the exact observed value distribution within each event-count group.','',
    '| Events | Min completeness | Sentences | Within-group % |','|---|---:|---:|---:|']
    for c in ('1','2','3+'):
      for r in s['sentence_min_distribution_by_event_count'][c]:
        L.append(f"| {c} | {r['min_completeness']:.4f} | {r['sentences']} | {pct(r['percent_within_group'])} |")
    L += ['', 'The next table groups the same values by train-positive-sentence terciles; these are descriptive groups, not curriculum thresholds.','',
    f"Cuts: Low ≤ {s['quantile_cuts']['q1']:.4f}; Medium > {s['quantile_cuts']['q1']:.4f} and ≤ {s['quantile_cuts']['q2']:.4f}; High > {s['quantile_cuts']['q2']:.4f}.",'',
    '| Events | High n (%) | Medium n (%) | Low n (%) | Total |','|---|---:|---:|---:|---:|']
    for c in ('1','2','3+'):
      d=s['event_count_x_min_completeness'][c];n=sum(x['count'] for x in d.values())
      L.append(f"| {c} | {d['High']['count']} ({pct(d['High']['percent'])}) | {d['Medium']['count']} ({pct(d['Medium']['percent'])}) | {d['Low']['count']} ({pct(d['Low']['percent'])}) | {n} |")
    L += ['', '## Q6. Raw completeness as curriculum difficulty','','Raw completeness is a role-coverage ratio, not a calibrated cross-type difficulty scale. Types with more legal roles have finer possible score increments; one absent role changes the score by 1/|schema|. A role that rarely appears in the training annotations can mechanically lower completeness for its type, without implying that the role is unimportant. The exact-value distribution shows how much scores concentrate on discrete fractions. Since type-specific baselines, schema sizes, and role presence differ, raw values are not automatically comparable between types. Within-type midrank percentiles place each type near the same 50th-percentile center by construction; this removes type baselines but also discards absolute role coverage. For illustration, compare each type’s raw mean and average within-type percentile below. The percentile is diagnostic only and is not used for curriculum labels.','',
    '| Type | Raw mean completeness | Mean within-type percentile |','|---|---:|---:|']
    for r in s['per_type']:L.append(f"| {r['event_type']} | {r['mean']:.4f} | {r['mean_within_type_percentile']:.4f} |")
    L += ['',
    '## Reproducibility and scope','','Uses only original sentence-C train gold plus the supplied Appendix F.2 reference. No dev/test data, predictions, model inference, role weights, or source-data edits. Quantile groups are descriptive, not proposed thresholds.']
    Path(path).write_text(chr(10).join(L)+chr(10),encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,default=TRAIN);p.add_argument('--schema',type=Path,default=SCHEMA);p.add_argument('--output-dir',type=Path,default=OUT);a=p.parse_args()
    print(json.dumps(analyze(a.train,a.schema,a.output_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
