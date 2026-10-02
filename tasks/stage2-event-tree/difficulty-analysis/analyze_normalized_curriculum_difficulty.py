#!/usr/bin/env python3
"""Analyze normalized train-only curriculum difficulty from existing CSVs."""
import argparse,csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'tasks/stage2-event-tree/difficulty-analysis'
EVENTS=BASE/'schema_completeness_event_level.csv'
SENTENCES=BASE/'schema_completeness_sentence_level.csv'
OLD=BASE/'continuous_difficulty_sentence_level.csv'
MANUAL=BASE/'curriculum_d1_d2_assignment.csv'
TYPES=('PUR','PRP','WKS','ITT','RWS','RWF','MDS','FIN','CMP','FAC')

def readcsv(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def writecsv(path,rows,fields):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def quantile(v,q):
    x=sorted(v);pos=(len(x)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return x[lo]+(x[hi]-x[lo])*(pos-lo)
def summary_stats(v):
    return {'count':len(v),'mean':statistics.mean(v),'std':statistics.pstdev(v),'min':min(v),'max':max(v),
      'p10':quantile(v,.1),'p25':quantile(v,.25),'p50':quantile(v,.5),'p75':quantile(v,.75),'p90':quantile(v,.9)}
def midrank(events):
    vals=defaultdict(list)
    for e in events:vals[e['event_type']].append(float(e['schema_completeness']))
    result={}
    for t,xs in vals.items():
        for e in events:
            if e['event_type']==t:
                x=float(e['schema_completeness'])
                result[(e['sample_id'],int(e['event_index']))]=(sum(v<x for v in xs)+.5*sum(v==x for v in xs))/len(xs)
    return result
def ranks(v):
    order=sorted(range(len(v)),key=lambda i:v[i]);r=[0.0]*len(v);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and v[order[j]]==v[order[i]]:j+=1
        rank=(i+1+j)/2
        for k in range(i,j):r[order[k]]=rank
        i=j
    return r
def spearman(x,y):
    a,b=ranks(x),ranks(y);ma=sum(a)/len(a);mb=sum(b)/len(b)
    num=sum((i-ma)*(j-mb) for i,j in zip(a,b));da=sum((i-ma)**2 for i in a);db=sum((j-mb)**2 for j in b)
    return num/math.sqrt(da*db) if da and db else None
def component_stats(v):
    return {'count':len(v),'mean':statistics.mean(v),'median':statistics.median(v),'p25':quantile(v,.25),'p75':quantile(v,.75)}
def bin_name(x):
    if x>=1:return '[0.9,1.0]'
    lo=math.floor(x*10+1e-12)/10
    return f'[{lo:.1f},{lo+.1:.1f})'

def analyze(event_path=EVENTS,sentence_path=SENTENCES,old_path=OLD,manual_path=MANUAL,outdir=BASE):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    events=readcsv(event_path);sentences=readcsv(sentence_path);old_rows=readcsv(old_path);manual_rows=readcsv(manual_path)
    pmap=midrank(events);by_sample=defaultdict(list)
    for e in events:
        key=(e['sample_id'],int(e['event_index']));p=pmap[key]
        existing=e.get('within_type_completeness_percentile','')
        if existing and abs(float(existing)-p)>1e-12:raise ValueError(f'Midrank mismatch {key}')
        if not 0<=p<=1:raise ValueError(f'Percentile out of range {key}')
        e['_p']=p;by_sample[e['sample_id']].append(e)
    old={r['sample_id']:r for r in old_rows};manual={r['sample_id']:r for r in manual_rows}
    if len({s['sample_id'] for s in sentences})!=len(sentences):raise ValueError('Duplicate sentence IDs')
    sentence_ids={s['sample_id'] for s in sentences}
    if set(by_sample)-sentence_ids:raise ValueError('Event sample missing from sentence data')
    records=[];positive=[];root=0;count_scores=defaultdict(list)
    for s in sentences:
        sid=s['sample_id'];n=int(s['event_count']);es=sorted(by_sample.get(sid,[]),key=lambda e:int(e['event_index']))
        if len(es)!=n:raise ValueError(f'Event count mismatch for {sid}')
        ps=[e['_p'] for e in es];raw=[float(e['schema_completeness']) for e in es]
        pmin=min(ps) if ps else None;dsch=1-pmin if ps else None
        devent=1-1/n if n else None
        score=(devent+dsch)/2 if n else None
        oldrow=old.get(sid)
        if oldrow is None:raise ValueError(f'Missing prior continuous row {sid}')
        oldscore=float(oldrow.get('difficulty_score_old',oldrow.get('difficulty_score',''))) if n else None
        oldbucket=oldrow.get('difficulty_bucket_old',oldrow.get('difficulty_bucket',''))
        if n and oldbucket not in ('B1','B2'):raise ValueError(f'Invalid prior bucket {sid}:{oldbucket}')
        if n:
            bucket='';positive.append({'sample_id':sid,'abstract_id':s['abstract_id'],'target_sentence_id':s['target_sentence_id'],
              'gold_event_count':n,'event_types':[e['event_type'] for e in es],'event_raw_completeness':raw,
              'event_within_type_percentiles':ps,'sentence_min_percentile':pmin,'schema_incompleteness':dsch,
              'normalized_event_multiplicity':devent,'difficulty_score_old':oldscore,
              'difficulty_score_normalized':score,'difficulty_bucket_old':oldbucket,
              'difficulty_bucket_normalized':''})
            if abs(oldscore-(n-pmin))>1e-10:raise ValueError(f'Old score formula mismatch for {sid}')
            count_scores[n].append(score)
        else:bucket='ROOT';root+=1
        records.append({'sample_id':sid,'abstract_id':s['abstract_id'],'target_sentence_id':s['target_sentence_id'],
          'gold_event_count':n,'event_types':'|'.join(e['event_type'] for e in es),
          'event_raw_completeness':'|'.join(f'{x:.8f}' for x in raw),
          'event_within_type_percentiles':'|'.join(f'{x:.8f}' for x in ps),
          'sentence_min_percentile':pmin if pmin is not None else 'NA',
          'schema_incompleteness':dsch if dsch is not None else 'NA',
          'normalized_event_multiplicity':devent if devent is not None else 'NA',
          'difficulty_score_old':oldscore if oldscore is not None else 'NA',
          'difficulty_score_normalized':score if score is not None else 'NA',
          'difficulty_bucket_old':oldbucket if n else 'ROOT','difficulty_bucket_normalized':bucket})
    positive.sort(key=lambda r:(r['difficulty_score_normalized'],r['sample_id']))
    half=len(positive)//2
    if len(positive)%2:raise ValueError('Odd positive count cannot be split equally')
    for i,r in enumerate(positive):r['difficulty_bucket_normalized']='B1' if i<half else 'B2'
    assigned={r['sample_id']:r['difficulty_bucket_normalized'] for r in positive}
    for r in records:
        if r['gold_event_count']>0:r['difficulty_bucket_normalized']=assigned[r['sample_id']]
    writecsv(out/'normalized_difficulty_sentence_level.csv',records,
      ['sample_id','abstract_id','target_sentence_id','gold_event_count','event_types','event_raw_completeness',
       'event_within_type_percentiles','sentence_min_percentile','schema_incompleteness',
       'normalized_event_multiplicity','difficulty_score_old','difficulty_score_normalized',
       'difficulty_bucket_old','difficulty_bucket_normalized'])
    assignment=[]
    for r in positive:
        assignment.append({'sample_id':r['sample_id'],'gold_event_count':r['gold_event_count'],
          'sentence_min_percentile':r['sentence_min_percentile'],'schema_incompleteness':r['schema_incompleteness'],
          'normalized_event_multiplicity':r['normalized_event_multiplicity'],
          'difficulty_score_old':r['difficulty_score_old'],'difficulty_score_normalized':r['difficulty_score_normalized'],
          'difficulty_bucket_old':r['difficulty_bucket_old'],'difficulty_bucket_normalized':r['difficulty_bucket_normalized'],
          'previous_D1_D2_label':manual.get(r['sample_id'],{}).get('difficulty_group','')})
    for r in records:
        if r['gold_event_count']==0:
            assignment.append({'sample_id':r['sample_id'],'gold_event_count':0,'sentence_min_percentile':'NA',
             'schema_incompleteness':'NA','normalized_event_multiplicity':'NA','difficulty_score_old':'NA',
             'difficulty_score_normalized':'NA','difficulty_bucket_old':'ROOT','difficulty_bucket_normalized':'ROOT',
             'previous_D1_D2_label':manual.get(r['sample_id'],{}).get('difficulty_group','')})
    writecsv(out/'normalized_difficulty_bucket_assignment.csv',assignment,
       ['sample_id','gold_event_count','sentence_min_percentile','schema_incompleteness','normalized_event_multiplicity',
        'difficulty_score_old','difficulty_score_normalized','difficulty_bucket_old','difficulty_bucket_normalized','previous_D1_D2_label'])
    scores=[r['difficulty_score_normalized'] for r in positive]
    histrows=[]
    for i in range(10):
        lo=i/10;hi=(i+1)/10
        c=sum(lo<=x<hi for x in scores) if i<9 else sum(lo<=x<=hi for x in scores)
        histrows.append({'bin':f'[{lo:.1f},{hi:.1f})' if i<9 else '[0.9,1.0]','count':c,'percent':c/len(scores)})
    writecsv(out/'normalized_difficulty_distribution.csv',histrows,['bin','count','percent'])
    # Assign equal buckets and component data.
    bucket_counts=Counter(r['difficulty_bucket_normalized'] for r in positive)
    eventcomp=[];type_events=Counter();type_pcts=defaultdict(list);type_bucket=Counter()
    for b in ('B1','B2'):
        subset=[r for r in positive if r['difficulty_bucket_normalized']==b]
        for cat in ('1','2','3','4+'):
            n=sum(('4+' if r['gold_event_count']>=4 else str(r['gold_event_count']))==cat for r in subset)
            eventcomp.append({'table':'event_count','group':b,'category':cat,'count':n,
              'percent_within_group':n/len(subset),'denominator':len(subset)})
    old_bucket_counts=Counter(r['difficulty_bucket_old'] for r in positive)
    for b in ('B1','B2'):
        subset=[r for r in positive if r['difficulty_bucket_old']==b]
        for cat in ('1','2','3','4+'):
            n=sum(('4+' if r['gold_event_count']>=4 else str(r['gold_event_count']))==cat for r in subset)
            eventcomp.append({'table':'event_count_old','group':b,'category':cat,'count':n,
              'percent_within_group':n/len(subset),'denominator':len(subset)})
    for r in positive:
        b=r['difficulty_bucket_normalized']
        for t,p in zip(r['event_types'],r['event_within_type_percentiles']):
            type_events[b,t]+=1;type_pcts[b].append(p);type_bucket[b,t]+=1
    totals={b:sum(type_events[b,t] for t in TYPES) for b in ('B1','B2')}
    for t in TYPES:
      for b in ('B1','B2'):
        n=type_events[b,t];typecomp={'table':'event_type_bucket_share','group':b,'category':t,'count':n,
          'percent_within_group':n/totals[b],'denominator':totals[b]}
        eventcomp.append(typecomp)
        type_rate={'table':'event_type_assignment_rate','group':b,'category':t,'count':n,
          'percent_within_group':n/(type_events['B1',t]+type_events['B2',t]),
          'denominator':type_events['B1',t]+type_events['B2',t]}
        eventcomp.append(type_rate)
    writecsv(out/'normalized_difficulty_bucket_composition.csv',eventcomp,
      ['table','group','category','count','percent_within_group','denominator'])
    # Type assignment skew old vs new.
    type_rates={}
    type_rate_counts={}
    for variant,groups in [('old',{r['sample_id']:r['difficulty_bucket_old'] for r in positive}),
                           ('normalized',assigned)]:
        bytype=Counter()
        for r in positive:
            for t in r['event_types']:bytype[t,groups[r['sample_id']]]+=1
        rates={t:bytype[t,'B1']/(bytype[t,'B1']+bytype[t,'B2']) for t in TYPES}
        type_rate_counts[variant]={t:{'B1':bytype[t,'B1'],'B2':bytype[t,'B2']} for t in TYPES}
        type_rates[variant]={'rates':rates,'max':max(rates.values()),'min':min(rates.values()),
          'range':max(rates.values())-min(rates.values()),'std':statistics.pstdev(list(rates.values()))}
    # Old/new and previous manual comparisons.
    oldconf=Counter();manualconf=Counter()
    for r in positive:
        oldconf[r['difficulty_bucket_old'],r['difficulty_bucket_normalized']]+=1
        label=manual.get(r['sample_id'],{}).get('difficulty_group')
        if label not in ('D1','D2'):raise ValueError(f'Missing manual label {r["sample_id"]}')
        manualconf[label,r['difficulty_bucket_normalized']]+=1
    old_agree=(oldconf['B1','B1']+oldconf['B2','B2'])/len(positive)
    manual_agree=(manualconf['D1','B1']+manualconf['D2','B2'])/len(positive)
    # Event count × score summaries, with 4+ grouped and raw counts preserved.
    group_stats={}
    for label,vals in [('1',[r['difficulty_score_normalized'] for r in positive if r['gold_event_count']==1]),
      ('2',[r['difficulty_score_normalized'] for r in positive if r['gold_event_count']==2]),
      ('3',[r['difficulty_score_normalized'] for r in positive if r['gold_event_count']==3]),
      ('4+',[r['difficulty_score_normalized'] for r in positive if r['gold_event_count']>=4])]:
        if vals:group_stats[label]={'samples':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals),
           'p25':quantile(vals,.25),'p75':quantile(vals,.75),'min':min(vals),'max':max(vals)}
    pair_overlap={}
    for a,b in ((1,2),(2,3)):
        va=count_scores[a];vb=count_scores[b]
        lo=max(min(va),min(vb));hi=min(max(va),max(vb))
        in_a=[x for x in va if lo<=x<=hi] if lo<=hi else []
        in_b=[x for x in vb if lo<=x<=hi] if lo<=hi else []
        shared=set(va)&set(vb)
        pair_overlap[f'{a}_vs_{b}']={'range_overlap':lo<=hi,'overlap_low':lo if lo<=hi else None,
          'overlap_high':hi if lo<=hi else None,'samples_a_in_intersection':len(in_a),
          'samples_b_in_intersection':len(in_b),'shared_exact_score_values':len(shared),
          'samples_a_at_shared_scores':sum(x in shared for x in va),
          'samples_b_at_shared_scores':sum(x in shared for x in vb),
          'a_min':min(va),'a_max':max(va),'b_min':min(vb),'b_max':max(vb)}
    # Illustrative cross-count comparisons: select largest-margin examples, deterministic IDs.
    examples={}
    singles=sorted([r for r in positive if r['gold_event_count']==1],key=lambda r:(-r['difficulty_score_normalized'],r['sample_id']))
    examples['one_event_high_difficulty']=[sample_case(r) for r in singles[:5]]
    examples['two_event_easier_than_one_event']=cross_examples(positive,2,1)
    examples['three_event_easier_than_two_event']=cross_examples(positive,3,2)
    mins=[r['sentence_min_percentile'] for r in positive]
    de=[1-1/r['gold_event_count'] for r in positive]
    ds=[r['schema_incompleteness'] for r in positive]
    old_scores=[r['difficulty_score_old'] for r in positive]
    rate_csv=[]
    for t in TYPES:
      for variant in ('old','normalized'):
        rr=type_rates[variant]['rates'][t]
        rate_csv.append({'comparison':'type_B1_assignment_rate','metric':variant,'group':t,'value':rr})
    oldnew_corr=spearman(old_scores,scores)
    summary={'train_counts':{'total':len(sentences),'root':sum(int(s['event_count'])==0 for s in sentences),
      'positive':len(positive),'b1':bucket_counts['B1'],'b2':bucket_counts['B2']},
      'normalized_difficulty':summary_stats(scores),'histogram':histrows,
      'event_count_stats':group_stats,'raw_event_count_counts':dict(Counter(int(s['event_count']) for s in sentences)),
      'bucket_counts':dict(bucket_counts),'boundary_score':positive[half]['difficulty_score_normalized'],
      'boundary_tie':boundary_summary(positive,half),
      'event_count_composition':composition_summary(eventcomp,'event_count'),
      'old_event_count_composition':composition_summary(eventcomp,'event_count_old'),
      'event_type_bucket_shares':composition_summary(eventcomp,'event_type_bucket_share'),
      'event_type_assignment_rates':{v:type_rates[v]['rates'] for v in ('old','normalized')},
      'type_rate_counts':type_rate_counts,
      'type_skew':{v:{k:type_rates[v][k] for k in ('max','min','range','std')} for v in ('old','normalized')},
      'old_new_bucket_confusion':{a:{b:oldconf[a,b] for b in ('B1','B2')} for a in ('B1','B2')},
      'old_new_bucket_agreement':old_agree,
      'manual_d1_d2_confusion':{a:{b:manualconf[a,b] for b in ('B1','B2')} for a in ('D1','D2')},
      'manual_d1_d2_agreement':manual_agree,
      'within_type_percentiles_by_bucket':{b:component_stats(type_pcts[b]) for b in ('B1','B2')},
      'spearman_normalized_event_only':spearman(scores,de),
      'spearman_normalized_schema_only':spearman(scores,ds),
      'spearman_old_vs_normalized':oldnew_corr,'adjacent_event_count_overlap':pair_overlap,
      'illustrative_cases':examples,'root_score_is_na':True}
    # Invariants and independent accounting.
    counts=summary['train_counts'];assert counts['root']+counts['positive']==counts['total']
    assert counts['b1']==counts['b2']==len(positive)//2
    for r in positive:
        n=r['gold_event_count'];p=r['sentence_min_percentile']
        devent=1-1/n;dsch=1-p;expected=(devent+dsch)/2
        assert abs(expected-r['difficulty_score_normalized'])<1e-12
        assert 0<=devent<1 and 0<=dsch<=1 and 0<=expected<=1
        assert abs(p-min(r['event_within_type_percentiles']))<1e-12
    assert sum(x['count'] for x in histrows)==len(positive)
    assert sum(type_events[b,t] for b in ('B1','B2') for t in TYPES)==sum(r['gold_event_count'] for r in positive)
    (out/'normalized_difficulty_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    # Requested comparisons as a tidy table.
    comparisons=[]
    for v in ('old','normalized'):
        for k,x in type_rates[v].items():
            if k=='rates':continue
            comparisons.append({'comparison':'type_skew','metric':k,'group':v,'value':x})
    for label,val in [('Spearman normalized vs event-only',summary['spearman_normalized_event_only']),
      ('Spearman normalized vs schema-only',summary['spearman_normalized_schema_only']),
      ('Spearman old vs normalized',oldnew_corr),('old/new bucket agreement',old_agree),
      ('manual/new bucket agreement',manual_agree)]:
        comparisons.append({'comparison':'formula_or_agreement','metric':label,'group':'all','value':val})
    for old_label in ('B1','B2'):
        for new_label in ('B1','B2'):
            comparisons.append({'comparison':'old_vs_normalized_confusion','metric':old_label,'group':new_label,
              'value':oldconf[old_label,new_label]})
    for old_label in ('D1','D2'):
        for new_label in ('B1','B2'):
            comparisons.append({'comparison':'manual_vs_normalized_confusion','metric':old_label,'group':new_label,
              'value':manualconf[old_label,new_label]})
    for variant in ('old','normalized'):
        for bucket in ('B1','B2'):
            for cat,x in summary['event_count_composition' if variant=='normalized' else 'old_event_count_composition'][bucket].items():
                comparisons.append({'comparison':'event_count_composition','metric':cat,'group':f'{variant}_{bucket}',
                  'value':x['count']})
    for a,vals in summary['adjacent_event_count_overlap'].items():
        for k,v in vals.items():
            comparisons.append({'comparison':'event_count_overlap','metric':k,'group':a,'value':v})
    comparisons.extend(rate_csv)
    writecsv(out/'normalized_difficulty_comparison.csv',comparisons,['comparison','metric','group','value'])
    checks={'root_plus_positive_equals_total':counts['root']+counts['positive']==counts['total'],
      'b1_plus_b2_equals_positive':counts['b1']+counts['b2']==counts['positive'],
      'equal_bucket_sizes':counts['b1']==counts['b2']==len(positive)//2,
      'event_and_schema_components_in_range':all(0<=1-1/r['gold_event_count']<1 and 0<=r['schema_incompleteness']<=1 for r in positive),
      'normalized_score_in_range':all(0<=r['difficulty_score_normalized']<=1 for r in positive),
      'formula_exact':all(abs(r['difficulty_score_normalized']-.5*((1-1/r['gold_event_count'])+(1-r['sentence_min_percentile'])))<1e-12 for r in positive),
      'pmin_matches_event_percentiles':all(abs(r['sentence_min_percentile']-min(r['event_within_type_percentiles']))<1e-12 for r in positive),
      'all_sentence_event_joins_match':True,'histogram_reconciles':sum(x['count'] for x in histrows)==len(positive),
      'type_event_counts_reconcile':sum(type_events[b,t] for b in ('B1','B2') for t in TYPES)==sum(r['gold_event_count'] for r in positive)}
    verification={'passed':all(checks.values()),'checks':checks,'counts':counts,
       'boundary_tie':summary['boundary_tie'],'old_new_agreement':old_agree,'manual_agreement':manual_agree}
    (out/'normalized_difficulty_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    make_report(out/'normalized_difficulty_report.md',summary)
    make_plots(out,positive)
    return summary

def sample_case(r):
    return {'sample_id':r['sample_id'],'event_count':r['gold_event_count'],
      'P_min':r['sentence_min_percentile'],'DIF_norm':r['difficulty_score_normalized'],
      'DIF_old':r['difficulty_score_old']}
def cross_examples(rows,lower_n,higher_n):
    low=[r for r in rows if r['gold_event_count']==lower_n]
    high=[r for r in rows if r['gold_event_count']==higher_n]
    pairs=[]
    for a in sorted(low,key=lambda r:(r['difficulty_score_normalized'],r['sample_id'])):
        candidates=[b for b in high if a['difficulty_score_normalized']<b['difficulty_score_normalized']]
        if candidates:
            b=max(candidates,key=lambda r:(r['difficulty_score_normalized'],r['sample_id']))
            pairs.append({'lower_event_count_sample':sample_case(a),'higher_event_count_sample':sample_case(b),
              'score_gap':b['difficulty_score_normalized']-a['difficulty_score_normalized']})
    pairs.sort(key=lambda x:(-x['score_gap'],x['lower_event_count_sample']['sample_id']))
    return pairs[:5]
def boundary_summary(rows,half):
    boundary=rows[half]['difficulty_score_normalized'];tie=[r for r in rows if r['difficulty_score_normalized']==boundary]
    return {'score':boundary,'n_equal':len(tie),'b1_equal':sum(r['difficulty_bucket_normalized']=='B1' for r in tie),
      'b2_equal':sum(r['difficulty_bucket_normalized']=='B2' for r in tie),
      'tie_preserving':{'below':sum(r['difficulty_score_normalized']<boundary for r in rows),
        'equal':len(tie),'above':sum(r['difficulty_score_normalized']>boundary for r in rows)}}
def composition_summary(rows,table):
    return {b:{r['category']:{'count':r['count'],'percent':r['percent_within_group']}
      for r in rows if r['table']==table and r['group']==b}
      for b in ('B1','B2')}
def make_report(path,s):
    d=s['normalized_difficulty'];lines=['# Normalized Curriculum Difficulty on Train','',
      'Fixed formula: DIF_norm = 0.5 * [(1 − 1/N_e) + (1 − P_min)]. Percentiles are train-only type-specific tie-aware midranks. Lower is easier. ROOT has NA.','',
      '## Q1. Overall normalized score','','| Statistic | Value |','|---|---:|']
    for key,label in [('count','Count'),('mean','Mean'),('std','SD'),('min','Min'),('max','Max'),('p10','P10'),('p25','P25'),('p50','P50'),('p75','P75'),('p90','P90')]:
        lines.append(f'| {label} | {d[key]:.6f} |')
    lines+=['','| DIF bin | Sentences | Percent |','|---|---:|---:|']+[f'| {x["bin"]} | {x["count"]} | {x["percent"]:.2%} |' for x in s['histogram']]
    lines+=['','## Q2. Event count and normalized difficulty','','| Event count | Samples | Mean | Median | P25 | P75 | Min | Max |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k in ('1','2','3','4+'):
        x=s['event_count_stats'][k]
        lines.append(f'| {k} | {x["samples"]} | {x["mean"]:.4f} | {x["median"]:.4f} | {x["p25"]:.4f} | {x["p75"]:.4f} | {x["min"]:.4f} | {x["max"]:.4f} |')
    lines+=['','### Adjacent-count overlap','',
      '| Pair | A min–max | B min–max | Shared range | A samples in range | B samples in range | Exact shared score values |','|---|---:|---:|---:|---:|---:|---:|']
    for label,x in s['adjacent_event_count_overlap'].items():
        shared=f'{x["overlap_low"]:.4f}–{x["overlap_high"]:.4f}' if x['range_overlap'] else 'none'
        lines.append(f'| {label} | {x["a_min"]:.4f}–{x["a_max"]:.4f} | {x["b_min"]:.4f}–{x["b_max"]:.4f} | {shared} | {x["samples_a_in_intersection"]} | {x["samples_b_in_intersection"]} | {x["shared_exact_score_values"]} |')
    lines+=['','Counts on both sides within a shared score range demonstrate realized overlap. Integer boundaries are not forced by event count; actual score distributions determine overlap.','',
      '## Q3. Equal-size B1/B2 and event-count composition','',
      f'B1={s["bucket_counts"]["B1"]:,}; B2={s["bucket_counts"]["B2"]:,}. Boundary score={s["boundary_tie"]["score"]:.6f}; {s["boundary_tie"]["n_equal"]} samples tie there, split B1/B2={s["boundary_tie"]["b1_equal"]}/{s["boundary_tie"]["b2_equal"]}. This tie split is administrative only; ordering is by normalized score then sample_id. Tie-preserving alternative (below/equal/above)={s["boundary_tie"]["tie_preserving"]["below"]}/{s["boundary_tie"]["tie_preserving"]["equal"]}/{s["boundary_tie"]["tie_preserving"]["above"]}.','',
      '| Formula/bucket | 1 event | 2 events | 3 events | 4+ events | Total |','|---|---:|---:|---:|---:|---:|']
    for b in ('B1','B2'):
        c=s['event_count_composition'][b]
        lines.append('| '+b+' | '+' | '.join(f'{c[k]["count"]} ({c[k]["percent"]:.2%})' for k in ('1','2','3','4+'))+f' | {s["bucket_counts"][b]} |')
    lines.append('| Old B1 | '+' | '.join(f'{s["old_event_count_composition"]["B1"][k]["count"]} ({s["old_event_count_composition"]["B1"][k]["percent"]:.2%})' for k in ('1','2','3','4+'))+f' | {sum(x["count"] for x in s["old_event_count_composition"]["B1"].values())} |')
    lines.append('| Old B2 | '+' | '.join(f'{s["old_event_count_composition"]["B2"][k]["count"]} ({s["old_event_count_composition"]["B2"][k]["percent"]:.2%})' for k in ('1','2','3','4+'))+f' | {sum(x["count"] for x in s["old_event_count_composition"]["B2"].values())} |')
    lines+=['','The old composition rows are recomputed from the prior bucket assignment CSV; they are shown as a direct comparison.','',
      '## Q4. Event-type assignment skew','','Rates use all events of a type as the denominator.','',
      '| Type | B1 events | % type in B1 | B2 events | % type in B2 |','|---|---:|---:|---:|---:|']
    for t in TYPES:
        b1=s['event_type_assignment_rates']['normalized'][t]
        z=s['type_rate_counts']['normalized'][t]
        lines.append(f'| {t} | {z["B1"]} | {b1:.2%} | {z["B2"]} | {1-b1:.2%} |')
    oldsk=s['type_skew']['old'];newsk=s['type_skew']['normalized']
    lines+=['',f'B1 type-rate max/min/range/SD: old={oldsk["max"]:.2%}/{oldsk["min"]:.2%}/{oldsk["range"]:.2%}/{oldsk["std"]:.4f}; normalized={newsk["max"]:.2%}/{newsk["min"]:.2%}/{newsk["range"]:.2%}/{newsk["std"]:.4f}. Range changed by {(newsk["range"]-oldsk["range"])*100:+.2f} percentage points and SD by {newsk["std"]-oldsk["std"]:+.4f}; the shift is small, and type skew remains.','',
      '## Q5. Component influence and old/new buckets','','',
      f'Spearman(normalized DIF, event-only)={s["spearman_normalized_event_only"]:.4f}; Spearman(normalized DIF, schema-only)={s["spearman_normalized_schema_only"]:.4f}. The new correlation gap is {s["spearman_normalized_event_only"]-s["spearman_normalized_schema_only"]:.4f}, versus the old-formula reference gap of 0.2749 (0.9293−0.6544): event count still has the stronger rank association, but the two are closer. These are rank associations, not causal component attributions. Spearman(old DIF, normalized DIF)={s["spearman_old_vs_normalized"]:.4f}.','',
      f'Old/new bucket agreement={s["old_new_bucket_agreement"]:.2%}.','',
      '| Old bucket | New B1 | New B2 |','|---|---:|---:|']
    for a in ('B1','B2'):lines.append(f'| {a} | {s["old_new_bucket_confusion"][a]["B1"]} | {s["old_new_bucket_confusion"][a]["B2"]} |')
    lines+=['',f'Previous manual D1/D2 agreement={s["manual_d1_d2_agreement"]:.2%}.','',
      '| Previous manual rule | New B1 | New B2 |','|---|---:|---:|']
    for a in ('D1','D2'):lines.append(f'| {a} | {s["manual_d1_d2_confusion"][a]["B1"]} | {s["manual_d1_d2_confusion"][a]["B2"]} |')
    lines+=['','## Q6. Cross-count ordering examples','','These are score comparisons only; no semantic interpretation is attached.','',
      '### High-difficulty 1-event examples']
    for x in s['illustrative_cases']['one_event_high_difficulty']:lines.append(f'- {x["sample_id"]}: P_min={x["P_min"]:.4f}, DIF_norm={x["DIF_norm"]:.4f}')
    lines+=['','### 2-event sentence easier than a 1-event sentence']
    for x in s['illustrative_cases']['two_event_easier_than_one_event']:
        a=x['lower_event_count_sample'];b=x['higher_event_count_sample']
        lines.append(f'- 2-event {a["sample_id"]} ({a["DIF_norm"]:.4f}) < 1-event {b["sample_id"]} ({b["DIF_norm"]:.4f})')
    lines+=['','### 3-event sentence easier than a 2-event sentence']
    for x in s['illustrative_cases']['three_event_easier_than_two_event']:
        a=x['lower_event_count_sample'];b=x['higher_event_count_sample']
        lines.append(f'- 3-event {a["sample_id"]} ({a["DIF_norm"]:.4f}) < 2-event {b["sample_id"]} ({b["DIF_norm"]:.4f})')
    lines+=['','## Q7–Q8. Interpretation and implementation checks','','The normalized formula allows cross-count score overlap: inspect the observed ranges and intersection counts above. Component correlations and bucket type rates quantify whether event multiplicity still dominates or type skew remains; the formula itself was not adjusted. Percentile and boundary ties are preserved/reported, with only the 50% bucket boundary split administratively. ROOT remains unscored.','',
      f'Counts reconcile: ROOT={s["train_counts"]["root"]:,}, positive={s["train_counts"]["positive"]:,}, total={s["train_counts"]["total"]:,}; B1/B2={s["bucket_counts"]["B1"]:,}/{s["bucket_counts"]["B2"]:,}. Verification details are in normalized_difficulty_verification.json.']
    Path(path).write_text(chr(10).join(lines)+chr(10),encoding='utf8')

def make_plots(out,positive):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.hist([r['difficulty_score_normalized'] for r in positive],bins=[i/10 for i in range(11)],color='#4472C4',edgecolor='white')
    ax.set(xlim=(0,1),xlabel='Normalized difficulty',ylabel='Positive train sentences',title='Normalized curriculum difficulty')
    fig.tight_layout();fig.savefig(out/'normalized_difficulty_histogram.png',dpi=160);plt.close(fig)
    groups=defaultdict(list)
    for r in positive:groups['4+' if r['gold_event_count']>=4 else str(r['gold_event_count'])].append(r['difficulty_score_normalized'])
    order=('1','2','3','4+')
    fig,ax=plt.subplots(figsize=(8,4.5));ax.boxplot([groups[x] for x in order],showfliers=False)
    ax.set_xticks(range(1,5),['1 event','2 events','3 events','4+ events'])
    ax.set_ylim(0,1);ax.set(xlabel='Gold event count',ylabel='Normalized difficulty',title='Normalized difficulty by event count')
    fig.tight_layout();fig.savefig(out/'normalized_difficulty_by_event_count.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--event-csv',type=Path,default=EVENTS);p.add_argument('--sentence-csv',type=Path,default=SENTENCES);p.add_argument('--old-csv',type=Path,default=OLD);p.add_argument('--manual-csv',type=Path,default=MANUAL);p.add_argument('--output-dir',type=Path,default=BASE);a=p.parse_args()
    print(json.dumps(analyze(a.event_csv,a.sentence_csv,a.old_csv,a.manual_csv,a.output_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
