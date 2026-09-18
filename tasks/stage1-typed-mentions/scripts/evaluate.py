"""Exact token-set identification metrics and mutually exclusive error categories."""
import argparse
from collections import Counter
import json
from pathlib import Path
from align_mentions import TYPES, align_mentions


def prf(tp, predicted, gold):
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    return dict(tp=tp, fp=predicted-tp, fn=gold-tp, predicted=predicted, gold=gold,
                precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0)


def evaluate_rows(rows):
    gold_set, pred_set = set(), set()
    invalid = unaligned = total_mentions = 0
    errors = {'FN':Counter(BOUNDARY=0, TYPE_ONLY=0, MISSING=0),
              'FP':Counter(BOUNDARY=0, OTHER=0)}
    details = []
    seen_ids = set()
    for row in rows:
        if row['id'] in seen_ids:
            raise ValueError('Duplicate sample id: '+row['id'])
        seen_ids.add(row['id'])
        doc = row['document_id']
        gold = {(tuple(sorted(set(m['offsets']))), m['type']) for m in row['gold_mentions']}
        if any(not p or t not in TYPES for p,t in gold):
            raise ValueError('Invalid gold')
        mentions = row.get('parsed_mentions', [])
        is_invalid = bool(row.get('invalid', False))
        if is_invalid and mentions:
            raise ValueError('Invalid output must have empty predictions')
        invalid += is_invalid
        total_mentions += len(mentions)
        aligned = align_mentions(row['tokens'], mentions, row['document_start'])
        predicted = {(tuple(m['offsets']),m['type']) for m in aligned if m['offsets'] is not None}
        failed = [m for m in aligned if m['offsets'] is None]
        unaligned += len(failed)
        gold_set.update((doc,p,t) for p,t in gold)
        pred_set.update((doc,p,t) for p,t in predicted)
        sample_errors = []
        for p,t in sorted(gold - predicted):
            if any(p == pp for pp,tt in predicted):
                category = 'TYPE_ONLY'
            elif any(set(p).intersection(pp) for pp,tt in predicted):
                category = 'BOUNDARY'
            else:
                category = 'MISSING'
            errors['FN'][category] += 1
            sample_errors.append(dict(side='FN',category=category,offsets=list(p),type=t))
        for p,t in sorted(predicted - gold):
            # Wrong type at an exact gold span is OTHER, not a boundary error.
            category = ('BOUNDARY' if not any(p == gp for gp,gt in gold)
                        and any(set(p).intersection(gp) for gp,gt in gold) else 'OTHER')
            errors['FP'][category] += 1
            sample_errors.append(dict(side='FP',category=category,offsets=list(p),type=t))
        for mention in failed:
            errors['FP']['OTHER'] += 1
            sample_errors.append(dict(side='FP',category='OTHER',**mention))
        details.append(dict(id=row['id'], document_id=doc, invalid=is_invalid,
                            target_sentence=row.get('target_sentence',' '.join(row['tokens'])),
                            gold_mentions=row['gold_mentions'], aligned_mentions=aligned, errors=sample_errors))
    gold_untyped = {(d,p) for d,p,t in gold_set}
    pred_untyped = {(d,p) for d,p,t in pred_set}
    metrics = dict(samples=len(rows), gold_typed_mentions=len(gold_set),
                   parsed_mentions=total_mentions, invalid_samples=invalid, unaligned_mentions=unaligned,
                   typed=prf(len(gold_set & pred_set),len(pred_set)+unaligned,len(gold_set)),
                   untyped=prf(len(gold_untyped & pred_untyped),len(pred_untyped)+unaligned,len(gold_untyped)),
                   per_type={}, invalid_rate=invalid/len(rows) if rows else 0.,
                   unaligned_rate=unaligned/total_mentions if total_mentions else 0.,
                   average_predicted_mentions_per_sentence=total_mentions/len(rows) if rows else 0.,
                   errors={side:dict(counts) for side,counts in errors.items()})
    for typ in TYPES:
        gold = {m for m in gold_set if m[2] == typ}
        correct = gold & pred_set
        metrics['per_type'][typ] = dict(gold=len(gold),tp=len(correct), recall=len(correct)/len(gold) if gold else None)
    assert sum(errors['FN'].values()) == metrics['typed']['fn']
    assert sum(errors['FP'].values()) == metrics['typed']['fp']
    return metrics, details


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--predictions')
    parser.add_argument('--output',required=True,help='Metrics JSON; detailed errors written next to it')
    parser.add_argument('--oracle',action='store_true')
    parser.add_argument('--allow-subset',action='store_true')
    args = parser.parse_args()
    data = read_jsonl(args.data)
    by_id = {r['id']:r for r in data}
    if len(by_id) != len(data):
        raise ValueError('Duplicate data IDs')
    if args.oracle:
        rows = [dict(r,parsed_mentions=[{'text':m['text'],'type':m['type']} for m in r['gold_mentions']],invalid=False) for r in data]
    else:
        if not args.predictions:
            parser.error('--predictions or --oracle is required')
        predictions = read_jsonl(args.predictions)
        pred_ids = [r['id'] for r in predictions]
        if len(set(pred_ids)) != len(pred_ids) or not set(pred_ids).issubset(by_id):
            raise ValueError('Duplicate or unknown prediction IDs')
        if not args.allow_subset and set(pred_ids) != set(by_id):
            raise ValueError('Missing predictions; use --allow-subset only for declared E0 sanity')
        rows = [dict(by_id[p['id']], parsed_mentions=p['parsed_mentions'],invalid=p['invalid']) for p in predictions]
    metrics, details = evaluate_rows(rows)
    metrics.update(data=str(Path(args.data).resolve()), predictions=args.predictions, oracle=args.oracle,
                   scope='subset' if len(rows)!=len(data) else 'full', sample_ids=[r['id'] for r in rows])
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)
    with output.with_suffix('.errors.jsonl').open('x') as f:
        for row in details:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in metrics.items() if k!='sample_ids'},ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
