"""Audit train type balance and held-out dev type inventories using fixed train rules."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PREFIX = 'mean_percentile_bucket_audit'


def csv_rows(name):
    with (BASE / name).open(encoding='utf8', newline='') as f:
        return list(csv.DictReader(f))


def jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def metrics(rows):
    tp = sum(r['tp'] for r in rows)
    fp = sum(r['fp'] for r in rows)
    fn = sum(r['fn'] for r in rows)
    same_count = [r for r in rows if r['count_exact']]
    return dict(sentences=len(rows), inventory_errors=sum(not r['inventory_exact'] for r in rows),
                inventory_error_rate=sum(not r['inventory_exact'] for r in rows)/len(rows) if rows else None,
                count_errors=sum(not r['count_exact'] for r in rows),
                count_error_rate=sum(not r['count_exact'] for r in rows)/len(rows) if rows else None,
                count_matched_sentences=len(same_count),
                type_errors_given_correct_count=sum(not r['inventory_exact'] for r in same_count),
                type_error_rate_given_correct_count=sum(not r['inventory_exact'] for r in same_count)/len(same_count) if same_count else None,
                gold_events=tp+fn, predicted_events=tp+fp, tp=tp, fp=fp, fn=fn,
                type_inventory_micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def main():
    events = csv_rows('schema_completeness_event_level.csv')
    assignments = [r for r in csv_rows('mean_percentile_difficulty_assignments.csv')
                   if r['fraction']=='0.5' and r['tie_policy']=='all_ties_easy']
    groups = {r['sample_id']:r['bucket'] for r in assignments}
    counts = {b:Counter() for b in ('B1','B2')}
    laws = {}
    completeness = defaultdict(list)
    for e in events:
        counts[groups[e['sample_id']]][e['event_type']] += 1
        laws[e['event_type']] = set(e['legal_roles'].split('|'))
        completeness[e['event_type']].append(float(e['schema_completeness']))
    type_rows = [dict(event_type=t, total=counts['B1'][t]+counts['B2'][t],
                      B1=counts['B1'][t], B2=counts['B2'][t],
                      B1_percent=100*counts['B1'][t]/(counts['B1'][t]+counts['B2'][t])) for t in sorted(laws)]
    assert sum(r['total'] for r in type_rows)==19488
    pred_path=ROOT/'tasks/stage2-event-tree/runs/C-42-20260918/checkpoint-eval/dev-checkpoint-2406.predictions.jsonl'
    dev_path=ROOT/'tasks/stage2-event-tree/data/sentence-c/dev.jsonl'
    preds=jsonl(pred_path);dev=jsonl(dev_path)
    pred_by_id={r['id']:r for r in preds}
    assert len(pred_by_id)==len(preds)==len(dev)
    assert set(pred_by_id)=={r['id'] for r in dev}
    assert not any(r.get('invalid',False) for r in preds)
    threshold=json.loads((BASE/'mean_percentile_difficulty_summary.json').read_text())['splits']['0.5']['boundary']
    records=[]
    for row in dev:
        gold=row['target_events'];n=len(gold);ps=[]
        for e in gold:
            t=e['event_type'];observed={a['argument_type'] for a in e['arguments']}
            assert observed<=laws[t]
            c=len(observed)/len(laws[t]);train=completeness[t]
            ps.append((sum(v<c for v in train)+.5*sum(v==c for v in train))/len(train))
        score=.5*((1-1/n)+(1-sum(ps)/n)) if n else None
        bucket='ROOT' if not n else 'B1' if score<=threshold else 'B2'
        g=Counter(e['event_type'] for e in gold)
        p=Counter(e['type'] for e in pred_by_id[row['id']]['parsed_events'])
        tp=sum((g&p).values());fp=sum(p.values())-tp;fn=n-tp
        records.append(dict(sample_id=row['id'],bucket=bucket,event_count=n,score=score if n else 'NA',
                            gold_types='|'.join(sorted(g.elements())),pred_types='|'.join(sorted(p.elements())),
                            inventory_exact=g==p,count_exact=n==sum(p.values()),tp=tp,fp=fp,fn=fn))
    summaries={b:metrics([r for r in records if r['bucket']==b]) for b in ('B1','B2','ROOT')}
    strata={str(n):{b:metrics([r for r in records if r['bucket']==b and r['event_count']==n])
                    for b in ('B1','B2')} for n in sorted({r['event_count'] for r in records if r['event_count']})}
    assert sum(r['sentences'] for r in summaries.values())==len(dev)
    assert sum(r['gold_events'] for r in summaries.values())==sum(len(r['target_events']) for r in dev)
    summary=dict(train_sentence_counts=dict(Counter(groups.values())),train_type_distribution=type_rows,
                 type_B1_rate_range_pp=max(r['B1_percent'] for r in type_rows)-min(r['B1_percent'] for r in type_rows),
                 threshold=threshold,baseline_prediction=str(pred_path),dev_input=str(dev_path),
                 dev_bucket_metrics=summaries,dev_by_event_count=strata,
                 metric_definition='Type multiset comparison per sentence: preserves duplicate types, ignores triggers and arguments. Optimistic inventory diagnosis, not official TC or instance-aligned type accuracy.',
                 limitation='No train prediction JSONL found under stage2-event-tree/runs. Dev diagnosis uses train percentiles and fixed train threshold; no dev refitting. Group composition is observational and confounded by event count/type.')
    for name,rs in [('train_types',type_rows),('dev_sentences',records)]:
        with (BASE/(PREFIX+'_'+name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    (BASE/(PREFIX+'_summary.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# Mean percentile B1/B2 audit','',summary['metric_definition'],'',summary['limitation'],'',
           '```json',json.dumps(summary,ensure_ascii=False,indent=2),'```']
    (BASE/(PREFIX+'_report.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
