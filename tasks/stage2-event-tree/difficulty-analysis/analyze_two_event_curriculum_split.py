#!/usr/bin/env python3
"""Apply the specified train-only two-event completeness percentile median split."""
import argparse, csv, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'tasks/stage2-event-tree/difficulty-analysis'
EVENT_CSV=BASE/'schema_completeness_event_level.csv'
SENTENCE_CSV=BASE/'schema_completeness_sentence_level.csv'

def read_csv(path):
    with Path(path).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))

def write_csv(path,rows,fields):
    with Path(path).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def quantile(values,q):
    x=sorted(values)
    if not x:return None
    pos=(len(x)-1)*q;lo=math.floor(pos);hi=math.ceil(pos)
    return x[lo]+(x[hi]-x[lo])*(pos-lo)

def midrank_percentiles(events):
    by_type=defaultdict(list)
    for e in events:by_type[e['event_type']].append(float(e['schema_completeness']))
    result={}
    for t,vals in by_type.items():
        ordered=sorted(vals)
        for e in events:
            if e['event_type']==t:
                x=float(e['schema_completeness'])
                result[(e['sample_id'],int(e['event_index']))]=(sum(v<x for v in ordered)+.5*sum(v==x for v in ordered))/len(ordered)
    return result

def analyze(event_path=EVENT_CSV,sentence_path=SENTENCE_CSV,outdir=BASE):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    events=read_csv(event_path);sentences=read_csv(sentence_path)
    per_sample=defaultdict(list)
    for e in events:per_sample[e['sample_id']].append(e)
    # Verify and reuse the saved train-only midrank values; recompute only if column absent.
    computed=midrank_percentiles(events)
    for e in events:
        key=(e['sample_id'],int(e['event_index']))
        if 'within_type_completeness_percentile' in e and e['within_type_completeness_percentile']!='':
            p=float(e['within_type_completeness_percentile'])
            if abs(p-computed[key])>1e-12:raise ValueError(f'Existing percentile differs from recomputed midrank: {key}')
        else:p=computed[key]
        if not 0<=p<=1:raise ValueError(f'Percentile outside [0,1]: {key}={p}')
        e['_pct']=p
    sentence_by_id={s['sample_id']:s for s in sentences}
    if len(sentence_by_id)!=len(sentences):raise ValueError('Duplicate sample_id in sentence CSV')
    if set(per_sample)-set(sentence_by_id):raise ValueError('Event rows contain sample IDs absent from sentence CSV')
    percentile_rows=[];metric_by_sample={}
    for s in sentences:
        sid=s['sample_id'];es=sorted(per_sample.get(sid,[]),key=lambda e:int(e['event_index']))
        n=int(s['event_count'])
        if len(es)!=n:raise ValueError(f'Event count mismatch for {sid}: sentence={n}, events={len(es)}')
        if n==2:
            if len(es)!=2:raise AssertionError('two-event sentence must have exactly two event rows')
            ps=[e['_pct'] for e in es]
            raws=[float(e['schema_completeness']) for e in es]
            row={'sample_id':sid,'abstract_id':s['abstract_id'],'target_sentence_id':s['target_sentence_id'],
              'event_type_1':es[0]['event_type'],'event_type_2':es[1]['event_type'],
              'raw_completeness_1':raws[0],'raw_completeness_2':raws[1],
              'within_type_percentile_1':ps[0],'within_type_percentile_2':ps[1],
              'sentence_min_percentile':min(ps),'sentence_mean_percentile':sum(ps)/2,
              'sentence_max_percentile':max(ps)}
            percentile_rows.append(row);metric_by_sample[sid]=row
    if not percentile_rows:raise ValueError('No two-event train sentences')
    mins=[r['sentence_min_percentile'] for r in percentile_rows]
    threshold=quantile(mins,.5)
    for r in percentile_rows:r['high_low_label']='High' if r['sentence_min_percentile']>=threshold else 'Low'
    bins=[]
    for i in range(10):
        lo=i/10;hi=(i+1)/10
        n=sum(lo<=v<hi for v in mins) if i<9 else sum(lo<=v<=hi for v in mins)
        bins.append({'bin':f'[{lo:.1f}, {hi:.1f}' + (']' if i==9 else ')'),
                     'lower':f'{lo:.1f}','upper':f'{hi:.1f}','count':n,'percent':n/len(mins)})
    write_csv(outdir/'two_event_sentence_percentiles.csv',percentile_rows,
      ['sample_id','abstract_id','target_sentence_id','event_type_1','event_type_2',
       'raw_completeness_1','raw_completeness_2','within_type_percentile_1',
       'within_type_percentile_2','sentence_min_percentile','sentence_mean_percentile',
       'sentence_max_percentile','high_low_label'])
    write_csv(outdir/'two_event_percentile_distribution.csv',bins,['bin','lower','upper','count','percent'])
    assignment=[];counts=Counter()
    for s in sentences:
        sid=s['sample_id'];n=int(s['event_count']);metric=metric_by_sample.get(sid)
        if n==0:group='ROOT'
        elif n==1:group='D1'
        elif n==2:group='D1' if metric['sentence_min_percentile']>=threshold else 'D2'
        else:group='D2'
        counts[group]+=1
        # Retain the actual within-type minimum for all nonempty sentences.
        es=per_sample.get(sid,[])
        pvals=[e['_pct'] for e in es]
        assignment.append({'sample_id':sid,'abstract_id':s['abstract_id'],'target_sentence_id':s['target_sentence_id'],
          'gold_event_count':n,'sentence_min_percentile':min(pvals) if pvals else '',
          'difficulty_group':group})
    write_csv(outdir/'curriculum_d1_d2_assignment.csv',assignment,
      ['sample_id','abstract_id','target_sentence_id','gold_event_count','sentence_min_percentile','difficulty_group'])
    positive=len(sentences)-counts['ROOT'];two_high=sum(r['high_low_label']=='High' for r in percentile_rows)
    two_low=len(percentile_rows)-two_high
    all_min=[r['sentence_min_percentile'] for r in percentile_rows]
    mean_vals=[r['sentence_mean_percentile'] for r in percentile_rows]
    max_vals=[r['sentence_max_percentile'] for r in percentile_rows]
    eq=sum(v==threshold for v in mins)
    mixed=sum(r['sentence_min_percentile']<.25 and r['sentence_max_percentile']>=.75 for r in percentile_rows)
    diffs=[r['sentence_max_percentile']-r['sentence_min_percentile'] for r in percentile_rows]
    type_diag={}
    for t in sorted({e['event_type'] for e in events}):
        vals=[e['_pct'] for e in events if e['event_type']==t]
        type_diag[t]={'events':len(vals),'unique_percentiles':len(set(vals)),
          'largest_percentile_tie_count':max(Counter(vals).values()),
          'largest_tie_percent':max(Counter(vals).values())/len(vals)}
    summary={'input_counts':{'total_train_sentences':len(sentences),'root_sentences':counts['ROOT'],
      'positive_sentences':positive,'one_event_sentences':sum(int(s['event_count'])==1 for s in sentences),
      'two_event_sentences':len(percentile_rows),'three_plus_event_sentences':sum(int(s['event_count'])>=3 for s in sentences)},
      'two_event_distribution':{'count':len(mins),'mean':statistics.mean(mins),
       'std':statistics.pstdev(mins),'min':min(mins),'max':max(mins),
       'p10':quantile(mins,.1),'p25':quantile(mins,.25),'p50':threshold,
       'p75':quantile(mins,.75),'p90':quantile(mins,.9)},
      'median_split':{'threshold':threshold,'high_count':two_high,'high_percent':two_high/len(mins),
       'low_count':two_low,'low_percent':two_low/len(mins),'equal_to_threshold':eq,
       'equal_percent':eq/len(mins)},
      'difficulty_counts':dict(counts),'difficulty_percentages':{
       'D1_of_positive':counts['D1']/positive,'D2_of_positive':counts['D2']/positive,
       'D1_of_all':counts['D1']/len(sentences),'D2_of_all':counts['D2']/len(sentences),
       'ROOT_of_all':counts['ROOT']/len(sentences)},
      'd1_d2_ratio':counts['D1']/counts['D2'] if counts['D2'] else None,
      'balance_assessment':'severe imbalance' if counts['D1']/positive>.7 or counts['D1']/positive<.3 else 'not severe by 70:30 criterion',
      'mean_median_diagnostics':{'median_min':statistics.median(all_min),
       'median_mean':statistics.median(mean_vals),'median_max':statistics.median(max_vals),
       'median_max_minus_min':statistics.median(diffs),'pairs_with_min_below_0.25_and_max_at_least_0.75':mixed},
      'type_percentile_tie_diagnostics':type_diag,'bins':bins}
    # Hard invariants before writing verification.
    assert len(percentile_rows)==summary['input_counts']['two_event_sentences']
    assert all(0<=r['within_type_percentile_1']<=1 and 0<=r['within_type_percentile_2']<=1 for r in percentile_rows)
    assert all(r['sentence_min_percentile']<=r['sentence_mean_percentile']<=r['sentence_max_percentile'] for r in percentile_rows)
    assert counts['D1']+counts['D2']==positive
    assert counts['ROOT']+counts['D1']+counts['D2']==len(sentences)
    assert sum(b['count'] for b in bins)==len(mins)
    if counts['D1']+counts['D2'] != positive:raise AssertionError('D1+D2 != positive')
    (outdir/'two_event_curriculum_split_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    verification={'passed':True,'checks':{'two_event_rows_match_sentence_count':True,
      'percentiles_in_range':True,'min_le_mean_le_max':True,'d1_plus_d2_equals_positive':True,
      'root_plus_d1_plus_d2_equals_all':True,'bins_sum_to_two_event_total':True,
      'existing_percentiles_match_midrank_recomputation':True},
      'counts':summary['input_counts'],'threshold':threshold,
      'assignment_counts':dict(counts)}
    (outdir/'two_event_curriculum_split_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+chr(10),encoding='utf8')
    make_report(outdir/'two_event_curriculum_split_report.md',summary)
    return summary

def pct(v):return f'{100*v:.2f}%'
def make_report(path,s):
    d=s['two_event_distribution'];m=s['median_split'];ic=s['input_counts'];dc=s['difficulty_counts'];p=s['difficulty_percentages'];diag=s['mean_median_diagnostics']
    lines=['# Two-event train sentence completeness split','',
      '仅分析 sentence-C train 的既有 gold completeness 表。事件 percentile 为 event type 内的 midrank percentile：小于当前 completeness 的事件数加同值事件数的一半，再除以该类型总事件数。同一类型、同一 completeness 的 event 得到相同 percentile。2-event sentence score 取两事件 percentile 的最小值。','',
      '## Q1. 2-event min-percentile distribution','','| Statistic | Value |','|---|---:|',
      f"| Count | {d['count']} |",f"| Mean | {d['mean']:.6f} |",f"| SD | {d['std']:.6f} |",f"| Min | {d['min']:.6f} |",f"| Max | {d['max']:.6f} |",f"| P10 | {d['p10']:.6f} |",f"| P25 | {d['p25']:.6f} |",f"| P50 / median | {d['p50']:.6f} |",f"| P75 | {d['p75']:.6f} |",f"| P90 | {d['p90']:.6f} |",'',
      'Ten-bin distribution:','',
      '| Bin | Count | Percent |','|---|---:|---:|']
    lines += [f"| {b['bin']} | {b['count']} | {pct(b['percent'])} |" for b in s['bins']]
    lines += ['', '## Q2. Median threshold and ties','',
      f"Median threshold = **{m['threshold']:.6f}**. High uses score ≥ threshold and Low uses score < threshold; median ties were kept together (no random split). Exactly equal to threshold: {m['equal_to_threshold']} ({pct(m['equal_percent'])}).",'',
      '| Group | Count | Percent of 2-event |','|---|---:|---:|',
      f"| High completeness | {m['high_count']} | {pct(m['high_percent'])} |",f"| Low completeness | {m['low_count']} | {pct(m['low_percent'])} |",'',
      '## Q3. Final D1 / D2 distribution','',
      f"Train totals: {ic['total_train_sentences']} sentences; ROOT={ic['root_sentences']}; positive={ic['positive_sentences']}; 1 event={ic['one_event_sentences']}; 2 events={ic['two_event_sentences']}; 3+ events={ic['three_plus_event_sentences']}; 2-event High={m['high_count']}; 2-event Low={m['low_count']}.",'',
      '| Group | Rule | Sentences | % of positive | % of all train |','|---|---|---:|---:|---:|',
      f"| D1 | 1-event + 2-event High | {dc['D1']} | {pct(p['D1_of_positive'])} | {pct(p['D1_of_all'])} |",f"| D2 | 2-event Low + 3+ event | {dc['D2']} | {pct(p['D2_of_positive'])} | {pct(p['D2_of_all'])} |",f"| ROOT | 0-event | {dc['ROOT']} | N/A | {pct(p['ROOT_of_all'])} |",'',
      '## Q4. Balance and source of imbalance','',
      f"D1:D2 = {dc['D1']}:{dc['D2']} = {s['d1_d2_ratio']:.3f}:1. {s['balance_assessment']}; no threshold adjustment was made. D1 has {ic['one_event_sentences']} one-event sentences plus {m['high_count']} two-event High; D2 has {m['low_count']} two-event Low plus {ic['three_plus_event_sentences']} 3+ sentences. The main source of the D1 excess is the 1-event versus 3+ pool size difference ({ic['one_event_sentences']-ic['three_plus_event_sentences']} samples); the median split itself contributes only a {m['high_count']-m['low_count']} sample High-over-Low difference.",'',
      '## Q5. Implementation diagnostics','',
      f"For 2-event sentences, median(min/mean/max event percentile) = {diag['median_min']:.6f}/{diag['median_mean']:.6f}/{diag['median_max']:.6f}. Median within-sentence max-minus-min gap is {diag['median_max_minus_min']:.6f}; {diag['pairs_with_min_below_0.25_and_max_at_least_0.75']} pairs have one event below 0.25 and the other at or above 0.75.",'',
      'Percentile ties are expected because completeness is discrete. Per-type event count, unique percentile count, largest tied block and its share:','',
      '| Type | Events | Unique percentiles | Largest tie | Tie share |','|---|---:|---:|---:|---:|']
    for t,x in s['type_percentile_tie_diagnostics'].items():lines.append(f"| {t} | {x['events']} | {x['unique_percentiles']} | {x['largest_percentile_tie_count']} | {pct(x['largest_tie_percent'])} |")
    lines += ['',f"In the 2-event min-score distribution, {m['equal_to_threshold']} samples lie exactly at the threshold, so the split is determined by the documented tie rule rather than forced to 50:50. High/Low counts and percentile bins are shown above. Type-specific tie blocks are preserved; the assignment does not randomly split event-level ties.",'',
      '## Verification','',
      f"2-event sentence rows={len(s['bins']) and d['count']}; D1+D2={dc['D1']+dc['D2']} equals positive sentences={ic['positive_sentences']}; ROOT+D1+D2={dc['ROOT']+dc['D1']+dc['D2']} equals total={ic['total_train_sentences']}. Percentile range/order, event-count joins, and bin totals are asserted in two_event_curriculum_split_verification.json.",'',
      'No model training/inference, dev/test data, redefinition of completeness, or threshold tuning was used.']
    Path(path).write_text(chr(10).join(lines)+chr(10),encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--event-csv',type=Path,default=EVENT_CSV);p.add_argument('--sentence-csv',type=Path,default=SENTENCE_CSV);p.add_argument('--output-dir',type=Path,default=BASE);a=p.parse_args()
    print(json.dumps(analyze(a.event_csv,a.sentence_csv,a.output_dir),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
