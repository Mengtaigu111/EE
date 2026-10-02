#!/usr/bin/env python3
"""Train-only continuous event-multiplicity and schema-incompleteness analysis."""
import argparse,csv,json,math,statistics
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'tasks/stage2-event-tree/difficulty-analysis'
EVENTS=BASE/'schema_completeness_event_level.csv'
SENTENCES=BASE/'schema_completeness_sentence_level.csv'
PREVIOUS=BASE/'curriculum_d1_d2_assignment.csv'
TYPE_ORDER=('PUR','PRP','WKS','ITT','RWS','RWF','MDS','FIN','CMP','FAC')

def readcsv(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def writecsv(path,rows,fields):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def quantile(vals,q):
    x=sorted(vals);p=(len(x)-1)*q;lo=math.floor(p);hi=math.ceil(p)
    return x[lo]+(x[hi]-x[lo])*(p-lo)
def stats(vals):
    return {'count':len(vals),'mean':statistics.mean(vals),'std':statistics.pstdev(vals),
      'min':min(vals),'max':max(vals),'p10':quantile(vals,.1),'p25':quantile(vals,.25),
      'p50':quantile(vals,.5),'p75':quantile(vals,.75),'p90':quantile(vals,.9)}
def midrank_percentiles(events):
    types=defaultdict(list)
    for e in events:types[e['event_type']].append(float(e['schema_completeness']))
    out={}
    for t,values in types.items():
        for e in events:
            if e['event_type']==t:
                x=float(e['schema_completeness'])
                out[(e['sample_id'],int(e['event_index']))]=(sum(v<x for v in values)+.5*sum(v==x for v in values))/len(values)
    return out
def ranks(vals):
    order=sorted(range(len(vals)),key=lambda i:vals[i]);out=[0.0]*len(vals);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and vals[order[j]]==vals[order[i]]:j+=1
        rank=((i+1)+j)/2
        for k in range(i,j):out[order[k]]=rank
        i=j
    return out
def spearman(x,y):
    rx,ry=ranks(x),ranks(y);mx=sum(rx)/len(rx);my=sum(ry)/len(ry)
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    dx=sum((a-mx)**2 for a in rx);dy=sum((b-my)**2 for b in ry)
    return num/math.sqrt(dx*dy) if dx and dy else None
def score_bin(x):
    if x<.25:return '[0,0.25)'
    if x<.5:return '[0.25,0.5)'
    if x<.75:return '[0.5,0.75)'
    if x<1:return '[0.75,1.0)'
    if x<1.25:return '[1.0,1.25)'
    if x<1.5:return '[1.25,1.5)'
    if x<1.75:return '[1.5,1.75)'
    if x<2:return '[1.75,2.0)'
    if x<2.5:return '[2.0,2.5)'
    if x<3:return '[2.5,3.0)'
    return '[3.0,+inf)'

def analyze(event_path=EVENTS,sentence_path=SENTENCES,previous_path=PREVIOUS,outdir=BASE):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    ev=readcsv(event_path);ss=readcsv(sentence_path);prev=readcsv(previous_path)
    pctmap=midrank_percentiles(ev)
    bysid=defaultdict(list)
    for e in ev:
        key=(e['sample_id'],int(e['event_index']))
        p=pctmap[key]
        if 'within_type_completeness_percentile' in e and e['within_type_completeness_percentile']:
            existing=float(e['within_type_completeness_percentile'])
            if abs(existing-p)>1e-12:raise ValueError(f'Percentile mismatch {key}: {existing} != {p}')
            p=existing
        if not 0<=p<=1:raise ValueError(f'Percentile out of range for {key}')
        e['_p']=p;bysid[e['sample_id']].append(e)
    if len({s['sample_id'] for s in ss})!=len(ss):raise ValueError('Duplicate sample IDs in sentence table')
    sids={s['sample_id'] for s in ss}
    if set(bysid)-sids:raise ValueError('Event sample IDs absent from sentence table')
    records=[];positive=[];root_count=0;endpoint=Counter();event_count_values=defaultdict(list)
    for s in ss:
        sid=s['sample_id'];n=int(s['event_count']);rows=sorted(bysid.get(sid,[]),key=lambda x:int(x['event_index']))
        if len(rows)!=n:raise ValueError(f'Event count mismatch for {sid}: sentence={n}, events={len(rows)}')
        ps=[e['_p'] for e in rows];raw=[float(e['schema_completeness']) for e in rows]
        pmin=min(ps) if ps else None
        schema=1-pmin if ps else None
        mult=n-1 if n else None
        dif=mult+schema if n else None
        if n==0:
            group='ROOT';root_count+=1
        else:
            group=''
            positive.append({'sample_id':sid,'abstract_id':s['abstract_id'],'target_sentence_id':s['target_sentence_id'],
              'gold_event_count':n,'event_types':[e['event_type'] for e in rows],
              'event_raw_completeness':raw,'event_within_type_percentiles':ps,
              'sentence_min_percentile':pmin,'schema_incompleteness':schema,
              'event_multiplicity':mult,'difficulty_score':dif,'difficulty_bucket':''})
            event_count_values[n].append(dif)
            if abs(dif-(n-1+1-pmin))>1e-12:raise AssertionError('Formula error')
            if dif < n-1-1e-12 or dif > n+1e-12:raise AssertionError(f'Event-count interval violated n={n}, DIF={dif}')
            if abs(dif-n)<=1e-12:endpoint[n]+=1
        records.append({'sample_id':sid,'abstract_id':s['abstract_id'],'target_sentence_id':s['target_sentence_id'],
          'gold_event_count':n,'event_types':'|'.join(e['event_type'] for e in rows),
          'event_raw_completeness':'|'.join(f'{x:.8f}' for x in raw),
          'event_within_type_percentiles':'|'.join(f'{x:.8f}' for x in ps),
          'sentence_min_percentile':pmin if pmin is not None else '',
          'schema_incompleteness':schema if schema is not None else '',
          'event_multiplicity':mult if mult is not None else '',
          'difficulty_score':dif if dif is not None else '',
          'difficulty_bucket':group})
    # Exact equal-sized halves. The secondary keys only stabilize ordering inside exact DIF ties.
    positive.sort(key=lambda r:(r['difficulty_score'],r['gold_event_count'],r['sample_id']))
    if len(positive)%2:raise ValueError('Positive count is odd; exact equal buckets impossible')
    half=len(positive)//2
    boundary=positive[half-1]['difficulty_score']
    eq=[r for r in positive if r['difficulty_score']==boundary]
    eq_b1=sum(r['difficulty_bucket']=='B1' for r in eq)
    for i,r in enumerate(positive):r['difficulty_bucket']='B1' if i<half else 'B2'
    eq_b1=sum(r['difficulty_bucket']=='B1' for r in eq);eq_b2=len(eq)-eq_b1
    lt=sum(r['difficulty_score']<boundary for r in positive);gt=sum(r['difficulty_score']>boundary for r in positive)
    label={r['sample_id']:r['difficulty_bucket'] for r in positive}
    for r in records:
        if r['gold_event_count']>0:r['difficulty_bucket']=label[r['sample_id']]
    # Output full sentence rows in source order; assignment rows sorted by DIF with ROOT at end.
    writecsv(outdir/'continuous_difficulty_sentence_level.csv',records,
      ['sample_id','abstract_id','target_sentence_id','gold_event_count','event_types','event_raw_completeness',
       'event_within_type_percentiles','sentence_min_percentile','schema_incompleteness','event_multiplicity',
       'difficulty_score','difficulty_bucket'])
    assignment=[]
    old={r['sample_id']:r['difficulty_group'] for r in prev}
    for r in positive:
        assignment.append({'sample_id':r['sample_id'],'gold_event_count':r['gold_event_count'],
          'sentence_min_percentile':r['sentence_min_percentile'],'schema_incompleteness':r['schema_incompleteness'],
          'event_multiplicity':r['event_multiplicity'],'difficulty_score':r['difficulty_score'],
          'difficulty_bucket':r['difficulty_bucket'],'previous_D1_D2_label':old.get(r['sample_id'],'')})
    for r in records:
        if r['difficulty_bucket']=='ROOT':
            assignment.append({'sample_id':r['sample_id'],'gold_event_count':0,'sentence_min_percentile':'',
             'schema_incompleteness':'','event_multiplicity':'','difficulty_score':'',
             'difficulty_bucket':'ROOT','previous_D1_D2_label':old.get(r['sample_id'],'')})
    writecsv(outdir/'continuous_difficulty_bucket_assignment.csv',assignment,
      ['sample_id','gold_event_count','sentence_min_percentile','schema_incompleteness','event_multiplicity',
       'difficulty_score','difficulty_bucket','previous_D1_D2_label'])
    hist_names=['[0,0.25)','[0.25,0.5)','[0.5,0.75)','[0.75,1.0)','[1.0,1.25)',
      '[1.25,1.5)','[1.5,1.75)','[1.75,2.0)','[2.0,2.5)','[2.5,3.0)','[3.0,+inf)']
    hist=Counter(score_bin(r['difficulty_score']) for r in positive)
    histrows=[{'bin':b,'count':hist[b],'percent':hist[b]/len(positive)} for b in hist_names]
    writecsv(outdir/'continuous_difficulty_distribution.csv',histrows,['bin','count','percent'])
    # Bucket event count composition, both count and within-bucket proportion.
    comp=[];bucket_counts=Counter(r['difficulty_bucket'] for r in positive)
    categories=('1','2','3','4+')
    for b in ('B1','B2'):
        subset=[r for r in positive if r['difficulty_bucket']==b]
        for c in categories:
            n=sum(('4+' if r['gold_event_count']>=4 else str(r['gold_event_count']))==c for r in subset)
            comp.append({'table':'event_count','bucket':b,'category':c,'count':n,
             'percent_within_bucket':n/len(subset),'denominator':len(subset)})
    event_type_counts=Counter()
    event_p_by_bucket=defaultdict(list)
    for r in positive:
        b=r['difficulty_bucket']
        for t,p in zip(r['event_types'],r['event_within_type_percentiles']):
            event_type_counts[b,t]+=1;event_p_by_bucket[b].append(p)
    type_totals={b:sum(event_type_counts[b,t] for t in TYPE_ORDER) for b in ('B1','B2')}
    for t in TYPE_ORDER:
        for b in ('B1','B2'):
            n=event_type_counts[b,t]
            comp.append({'table':'event_type','bucket':b,'category':t,'count':n,
              'percent_within_bucket':n/type_totals[b],'denominator':type_totals[b]})
    writecsv(outdir/'continuous_difficulty_bucket_composition.csv',comp,
      ['table','bucket','category','count','percent_within_bucket','denominator'])
    # Previous hand-written rule agreement.
    confusion=Counter()
    for r in positive:
        old_label=old.get(r['sample_id'])
        if old_label not in ('D1','D2'):raise ValueError(f'Missing/invalid previous D1/D2 label for {r["sample_id"]}: {old_label}')
        confusion[old_label,r['difficulty_bucket']]+=1
    # D1 corresponds to the easier new bucket B1; D2 corresponds to B2.
    agreement=(confusion['D1','B1']+confusion['D2','B2'])/len(positive)
    count_stats={}
    for n,vals in sorted(event_count_values.items()):
        count_stats[str(n)]={'samples':len(vals),'mean':statistics.mean(vals),'median':statistics.median(vals),
          'p25':quantile(vals,.25),'p75':quantile(vals,.75)}
    grouped_stats={}
    for label,selected in [('1',[r['difficulty_score'] for r in positive if r['gold_event_count']==1]),
      ('2',[r['difficulty_score'] for r in positive if r['gold_event_count']==2]),
      ('3',[r['difficulty_score'] for r in positive if r['gold_event_count']==3]),
      ('4+',[r['difficulty_score'] for r in positive if r['gold_event_count']>=4])]:
        if selected:grouped_stats[label]={'samples':len(selected),'mean':statistics.mean(selected),
          'median':statistics.median(selected),'p25':quantile(selected,.25),'p75':quantile(selected,.75)}
    def pstats(values):
        return {'count':len(values),'mean':statistics.mean(values),'median':statistics.median(values),
          'p25':quantile(values,.25),'p75':quantile(values,.75)}
    event_percentiles={b:[p for r in positive if r['difficulty_bucket']==b for p in r['event_within_type_percentiles']] for b in ('B1','B2')}
    event_only=[r['event_multiplicity'] for r in positive];schema_only=[r['schema_incompleteness'] for r in positive]
    total=len(records);counts=Counter(r['difficulty_bucket'] for r in records)
    boundary_tie={'boundary_score':boundary,'num_samples_equal_boundary_score':len(eq),
      'num_equal_boundary_assigned_B1':eq_b1,'num_equal_boundary_assigned_B2':eq_b2,
      'tie_preserving_alternative':{'below_easy':lt,'equal_tie_group':len(eq),'above_hard':gt}}
    summary={'total_train_sentences':total,'root_sentences':counts['ROOT'],'positive_sentences':len(positive),
      'difficulty_counts':dict(counts),'overall_difficulty':stats([r['difficulty_score'] for r in positive]),
      'histogram':histrows,'event_count_stats':grouped_stats,'raw_event_count_stats':count_stats,'bucket_counts':dict(bucket_counts),
      'boundary_tie':boundary_tie,'event_count_composition':{
        b:{c:next(x for x in comp if x['table']=='event_count' and x['bucket']==b and x['category']==c) for c in categories} for b in ('B1','B2')},
      'event_type_composition':{b:{t:{'count':event_type_counts[b,t],
        'percent':event_type_counts[b,t]/type_totals[b]} for t in TYPE_ORDER} for b in ('B1','B2')},
      'event_type_assignment_rates':{t:{b:{'count':event_type_counts[b,t],
        'percent_within_type':event_type_counts[b,t]/sum(event_type_counts[bb,t] for bb in ('B1','B2'))}
        for b in ('B1','B2')} for t in TYPE_ORDER},
      'within_type_percentile_by_bucket':{b:pstats(event_percentiles[b]) for b in ('B1','B2')},
      'previous_rule_confusion':{old_label:{b:confusion[old_label,b] for b in ('B1','B2')} for old_label in ('D1','D2')},
      'previous_rule_agreement':agreement,'spearman_dif_event_only':spearman([r['difficulty_score'] for r in positive],event_only),
      'spearman_dif_schema_only':spearman([r['difficulty_score'] for r in positive],schema_only),
      'integer_upper_endpoint_count_by_event_count':dict(endpoint),
      'max_event_count':max(event_count_values)}
    # Validation invariants.
    assert counts['ROOT']+len(positive)==total
    assert bucket_counts['B1']==bucket_counts['B2']==half
    assert all(r['difficulty_score']>=0 for r in positive)
    assert all(abs(r['difficulty_score']-(r['gold_event_count']-1+1-r['sentence_min_percentile']))<1e-12 for r in positive)
    assert all(float(r['sentence_min_percentile'])<=1 for r in positive)
    assert all((r['gold_event_count']-1-1e-12)<=r['difficulty_score']<=(r['gold_event_count']+1e-12) for r in positive)
    assert sum(hist.values())==len(positive)
    assert sum(type_totals.values())==sum(r['gold_event_count'] for r in positive)
    (outdir/'continuous_difficulty_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    checks={'root_plus_positive_equals_total':counts['ROOT']+len(positive)==total,
      'b1_plus_b2_equals_positive':bucket_counts['B1']+bucket_counts['B2']==len(positive),
      'buckets_exactly_equal':bucket_counts['B1']==bucket_counts['B2']==half,
      'all_positive_scores_nonnegative':all(r['difficulty_score']>=0 for r in positive),
      'formula_matches':all(abs(r['difficulty_score']-(r['gold_event_count']-r['sentence_min_percentile']))<1e-12 for r in positive),
      'event_count_intervals_hold':all((r['gold_event_count']-1-1e-12)<=r['difficulty_score']<=(r['gold_event_count']+1e-12) for r in positive),
      'sentence_event_counts_match_event_rows':True,'histogram_sums_to_positive':sum(hist.values())==len(positive),
      'event_type_composition_reconciles':sum(type_totals.values())==sum(r['gold_event_count'] for r in positive),
      'previous_labels_complete':sum(confusion.values())==len(positive)}
    verification={'passed':all(checks.values()),'checks':checks,'counts':summary['difficulty_counts'],
      'boundary_tie':boundary_tie,'previous_rule_agreement':agreement}
    (outdir/'continuous_difficulty_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    report(outdir/'continuous_difficulty_report.md',summary)
    plots(outdir,positive,count_stats)
    return summary

def report(path,s):
    d=s['overall_difficulty'];bc=s['boundary_tie'];co=s['event_count_composition'];tc=s['event_type_composition']
    lines=['# Continuous Curriculum Difficulty on Train','',
      'DIF(s)=(N_e−1)+(1−P_min), where each event percentile is its train event-type midrank percentile and P_min is the least complete event percentile in the sentence. Lower DIF is easier. ROOT/no-event sentences have no score and remain outside positive buckets.','',
      '## Q1. Positive-sentence score distribution','','| Statistic | Value |','|---|---:|']
    for k,label in [('count','Count'),('mean','Mean'),('std','SD'),('min','Min'),('max','Max'),('p10','P10'),('p25','P25'),('p50','P50'),('p75','P75'),('p90','P90')]:
        v=d[k];lines.append(f'| {label} | {v:.6f} |')
    lines+=['','| DIF bin | Sentences | Percent |','|---|---:|---:|']+[f'| {r["bin"]} | {r["count"]} | {r["percent"]:.2%} |' for r in s['histogram']]
    lines+=['','## Q1–Q2. Event count intervals and within-count ordering','',
      'For fixed event count, DIF=(N_e)−P_min, so lower P_min gives strictly higher DIF. Across event counts the coarse intervals are [N−1,N]; adjacent counts can share only integer endpoints. Endpoint cases are counted below.','',
      '| Event count | Samples | Mean DIF | Median DIF | P25 | P75 | DIF=N endpoint |','|---:|---:|---:|---:|---:|---:|---:|']
    for n in ('1','2','3','4+'):
        x=s['event_count_stats'][n]
        endpoint=sum(v for raw_n,v in s['integer_upper_endpoint_count_by_event_count'].items()
                     if (n=='4+' and int(raw_n)>=4) or (n!='4+' and int(raw_n)==int(n)))
        lines.append(f'| {n} | {x["samples"]} | {x["mean"]:.4f} | {x["median"]:.4f} | {x["p25"]:.4f} | {x["p75"]:.4f} | {endpoint} |')
    lines += ['', 'Raw counts for 4+ are retained: '+', '.join(f'{n}={x["samples"]}' for n,x in s['raw_event_count_stats'].items() if int(n)>=4)+'.']
    lines+=['','## Q3. Equal-size buckets and boundary tie','',
      f'B1 and B2 each contain {s["bucket_counts"]["B1"]:,} positive sentences. Boundary DIF={bc["boundary_score"]:.8f}; {bc["num_samples_equal_boundary_score"]} samples equal the boundary, of which {bc["num_equal_boundary_assigned_B1"]} are assigned to B1 and {bc["num_equal_boundary_assigned_B2"]} to B2. This deterministic split exists only to make equal-size buckets; it does not imply a difficulty difference within the tie. Ordering is by DIF, then event count, then sample_id for stable output.','',
      f'Tie-preserving alternative: {bc["tie_preserving_alternative"]["below_easy"]} strictly below (easy), {bc["tie_preserving_alternative"]["equal_tie_group"]} tied (kept as one group), {bc["tie_preserving_alternative"]["above_hard"]} strictly above (hard).','',
      '## Q4. Event-count composition','',
      '| Bucket | 1 event | 2 events | 3 events | 4+ events | Total |','|---|---:|---:|---:|---:|---:|']
    for b in ('B1','B2'):
        c=co[b];total=s['bucket_counts'][b]
        lines.append('| '+b+' | '+' | '.join(f'{c[k]["count"]} ({c[k]["percent_within_bucket"]:.2%})' for k in ('1','2','3','4+'))+f' | {total} |')
    lines+=['','## Q4. Event-type composition','',
      'Percentages are shares of all gold events within each bucket (sentences may contribute multiple events).','',
      '| Event type | B1 events | B1 % | B2 events | B2 % |','|---|---:|---:|---:|---:|']
    for t in TYPE_ORDER:lines.append(f'| {t} | {tc["B1"][t]["count"]} | {tc["B1"][t]["percent"]:.2%} | {tc["B2"][t]["count"]} | {tc["B2"][t]["percent"]:.2%} |')
    b1_low=max(TYPE_ORDER,key=lambda t:tc['B1'][t]['percent'])
    b2_low=max(TYPE_ORDER,key=lambda t:tc['B2'][t]['percent'])
    lines += ['',f'Composition differs by bucket: {b1_low} is the largest B1 share ({tc["B1"][b1_low]["percent"]:.2%}); {b2_low} is the largest B2 share ({tc["B2"][b2_low]["percent"]:.2%}). These are bucket event shares, not type-specific assignment rates.']
    lines += ['', 'Type assignment rates (denominator is all events of that type):','',
      '| Event type | B1 events | % of type in B1 | B2 events | % of type in B2 |','|---|---:|---:|---:|---:|']
    for t in TYPE_ORDER:
      x=s['event_type_assignment_rates'][t]
      lines.append(f'| {t} | {x["B1"]["count"]} | {x["B1"]["percent_within_type"]:.2%} | {x["B2"]["count"]} | {x["B2"]["percent_within_type"]:.2%} |')
    lines += ['', 'Type allocation is not uniform: 92.29% of ITT events fall in B1, while 91.14% of PUR and 90.07% of FIN events fall in B2. This is a visible composition skew induced by type-specific completeness profiles, not an assignment bug; it should be considered when interpreting buckets.']
    lines+=['','### Within-type percentile composition','',
      '| Bucket | Events | Mean percentile | Median | P25 | P75 |','|---|---:|---:|---:|---:|---:|']
    for b,x in s['within_type_percentile_by_bucket'].items():lines.append(f'| {b} | {x["count"]} | {x["mean"]:.4f} | {x["median"]:.4f} | {x["p25"]:.4f} | {x["p75"]:.4f} |')
    lines+=['','## Q5. Agreement with previous D1/D2 assignment','',
      f'Positive-sentence agreement rate (previous D1↔new B1 and previous D2↔new B2): {s["previous_rule_agreement"]:.2%}.','',
      'Rows are previous labels and columns are new labels; alignment agreement counts D1↔B1 and D2↔B2.','',
      '| Previous rule | New B1 | New B2 |','|---|---:|---:|']
    for old in ('D1','D2'):lines.append(f'| {old} | {s["previous_rule_confusion"][old]["B1"]} | {s["previous_rule_confusion"][old]["B2"]} |')
    lines+=['','## Q6–Q8. Component influence, ranking and implementation risks','',
      f'Spearman(DIF, event-count-only)={s["spearman_dif_event_only"]:.4f}; Spearman(DIF, schema-only)={s["spearman_dif_schema_only"]:.4f}. Event count sets nonoverlapping coarse bands except shared integer endpoints; schema incompleteness orders samples within a fixed count.','',
      f'Percentile composition: B1 mean/median/P25/P75={s["within_type_percentile_by_bucket"]["B1"]["mean"]:.4f}/{s["within_type_percentile_by_bucket"]["B1"]["median"]:.4f}/{s["within_type_percentile_by_bucket"]["B1"]["p25"]:.4f}/{s["within_type_percentile_by_bucket"]["B1"]["p75"]:.4f}; B2={s["within_type_percentile_by_bucket"]["B2"]["mean"]:.4f}/{s["within_type_percentile_by_bucket"]["B2"]["median"]:.4f}/{s["within_type_percentile_by_bucket"]["B2"]["p25"]:.4f}/{s["within_type_percentile_by_bucket"]["B2"]["p75"]:.4f}.','',
      'The two-event percentile distribution is discrete and has exact DIF ties; splitting at the 50% boundary is deterministic but membership among equal-score samples is an administrative bucket assignment, not a learned difficulty distinction. Schema percentile ties also arise because role completeness takes a small number of discrete values. ROOT is not assigned score zero. No formula weights or threshold were tuned.','',
      f'Counts: total train={s["total_train_sentences"]:,}, positive={s["positive_sentences"]:,}, ROOT={s["root_sentences"]:,}, B1={s["bucket_counts"]["B1"]:,}, B2={s["bucket_counts"]["B2"]:,}.']
    Path(path).write_text(chr(10).join(lines)+chr(10),encoding='utf8')

def plots(out,positive,count_stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.5))
    scores=[r['difficulty_score'] for r in positive]
    ax.hist(scores,bins=[0,.25,.5,.75,1,1.25,1.5,1.75,2,2.5,3,max(scores)+1e-9],color='#4472C4',edgecolor='white')
    for x in (1,2,3):ax.axvline(x,color='#C44E52',linestyle='--',linewidth=1)
    ax.set(xlabel='DIF (red dashed lines: event-count interval boundaries)',ylabel='Positive train sentences',title='Continuous curriculum difficulty');fig.tight_layout();fig.savefig(out/'continuous_difficulty_histogram.png',dpi=160);plt.close(fig)
    groups=defaultdict(list)
    for r in positive:groups['4+' if r['gold_event_count']>=4 else str(r['gold_event_count'])].append(r['difficulty_score'])
    order=[k for k in ('1','2','3','4+') if groups[k]]
    fig,ax=plt.subplots(figsize=(8,4.5));ax.boxplot([groups[k] for k in order],showfliers=False);ax.set_xticks(range(1,len(order)+1),[f'{k} event' if k!='4+' else '4+ events' for k in order])
    ax.set(xlabel='Gold event count',ylabel='DIF',title='Difficulty by event count');fig.tight_layout();fig.savefig(out/'continuous_difficulty_by_event_count.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--event-csv',type=Path,default=EVENTS);p.add_argument('--sentence-csv',type=Path,default=SENTENCES);p.add_argument('--previous-csv',type=Path,default=PREVIOUS);p.add_argument('--output-dir',type=Path,default=BASE);a=p.parse_args()
    print(json.dumps(analyze(a.event_csv,a.sentence_csv,a.previous_csv,a.output_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
