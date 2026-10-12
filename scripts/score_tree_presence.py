#!/usr/bin/env python3
"""Hierarchical text-presence evaluation of raw event trees.

TI: trigger; TC: trigger+type; AI: trigger+type+argument; AC: all plus role.
Each gold/predicted occurrence is correct when its complete signature exists in
that document on the other side. Matches are not consumed: repeated expressions
can reuse a counterpart. Precision counts predicted-side hits; recall counts
gold-side hits. EC retains argument-as-trigger text membership. No offset
recovery, one-to-one assignment, deduplication, or changes to original gold.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'tasks/stage2-event-tree'
FIXED = STAGE/'evaluation-v2-20261003/scripts'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules():
    load('ontology', FIXED/'ontology.py')
    parser = load('linearize', FIXED/'linearize.py')
    return parser


METRICS = ('TI','TC','AI','AC','EC_I','EC_C')


def text(mention):
    if 'tokens' in mention:
        return tuple(mention['tokens'])
    return tuple(mention['text'].split())


def signatures(events):
    result = {name:[] for name in METRICS}
    triggers = {text(event['trigger']) for event in events}
    for event in events:
        trigger = text(event['trigger'])
        parent = (trigger,event['event_type'])
        result['TI'].append(trigger)
        result['TC'].append(parent)
        for arg in event['arguments']:
            argument = text(arg)
            untyped = (*parent,argument)
            typed = (*untyped,arg['argument_type'])
            result['AI'].append(untyped)
            result['AC'].append(typed)
            if argument in triggers:
                result['EC_I'].append(untyped)
                result['EC_C'].append(typed)
    return result


def score_documents(golds, predictions):
    if len(golds) != len(predictions):
        raise ValueError('Gold and prediction document counts differ')
    totals = {name:Counter() for name in METRICS}
    per_document = []
    for index,(gold,pred) in enumerate(zip(golds,predictions)):
        gs,ps = signatures(gold),signatures(pred)
        counts = {}
        for name in METRICS:
            gset,pset = set(gs[name]),set(ps[name])
            gmatched = sum(item in pset for item in gs[name])
            pmatched = sum(item in gset for item in ps[name])
            counts[name] = {'gold_total':len(gs[name]),'predicted_total':len(ps[name]),
                            'gold_matched':gmatched,'predicted_matched':pmatched,
                            'fn':len(gs[name])-gmatched,'fp':len(ps[name])-pmatched}
            totals[name].update(counts[name])
        per_document.append({'document_index':index,'counts':counts})
    metrics = {}
    for name,c in totals.items():
        precision = c['predicted_matched']/c['predicted_total'] if c['predicted_total'] else 0.0
        recall = c['gold_matched']/c['gold_total'] if c['gold_total'] else 0.0
        metrics[name+'_p'] = precision
        metrics[name+'_r'] = recall
        metrics[name+'_f1'] = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'metrics':metrics,'counts':{k:dict(v) for k,v in totals.items()},
            'per_document':per_document}


def score(gold_documents, sentence_rows, predictions):
    """Score a complete dataset; retain event pairs and counts for inspection."""
    ids = [doc['doc_id'] for doc in gold_documents]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Gold IDs must be nonempty and unique')
    row_ids = [row['id'] for row in sentence_rows]
    pred_ids = [pred['id'] for pred in predictions]
    if len(row_ids) != len(set(row_ids)) or len(pred_ids) != len(set(pred_ids)):
        raise ValueError('Duplicate sentence or prediction ID')
    if set(row_ids) != set(pred_ids):
        raise ValueError('Predictions must exactly cover sentence data')
    groups = defaultdict(list)
    gold_by_id = {doc['doc_id']: doc for doc in gold_documents}
    for row in sentence_rows:
        groups[row['document_id']].append(row)
    if set(groups) != set(ids):
        raise ValueError('Sentence data and gold document coverage differ')
    pred_by_id = {pred['id']: pred for pred in predictions}
    parser = modules()
    documents, origins = [], []
    diagnostics = Counter()
    for doc_id in ids:
        rows = sorted(groups[doc_id], key=lambda r:r['sentence_index'])
        if [r['sentence_index'] for r in rows] != list(range(len(rows[0]['sentences']))):
            raise ValueError('Missing or duplicate sentence index: '+doc_id)
        events, event_origins = [], []
        for row in rows:
            gold = gold_by_id[doc_id]
            if row['sentences'] != rows[0]['sentences']:
                raise ValueError('Sentence contexts disagree: '+doc_id)
            if 'document' in gold and row.get('tokens') != gold['document']:
                raise ValueError('Sentence tokens differ from gold: '+row['id'])
            pred = pred_by_id[row['id']]
            for key in ('document_id', 'sentence_index'):
                if key in pred and pred[key] != row[key]:
                    raise ValueError('Prediction metadata differs: '+row['id']+' '+key)
            if not isinstance(pred.get('raw_output'), str):
                raise ValueError('Missing string raw_output: '+row['id'])
            parsed, diag = parser.parse_forest(pred['raw_output'])
            diagnostics.update({key:int(value) for key,value in diag.items()
                                if isinstance(value,(int,bool))})
            for index, event in enumerate(parsed):
                events.append({'event_type':event['type'],
                    'trigger':{'tokens':event['trigger'].split()},
                    'arguments':[{'argument_type':arg['role'],'tokens':arg['text'].split()}
                                 for arg in event['args']]})
                event_origins.append({'sentence_id':row['id'],'parsed_event_index':index})
        documents.append(events)
        origins.append({'doc_id':doc_id,'events':event_origins})
    result = score_documents([doc['events'] for doc in gold_documents], documents)
    result.update(documents=len(ids), sentence_rows=len(sentence_rows), document_ids=ids,
                  prediction_event_origins=origins, parser_diagnostics=dict(diagnostics),
                  gold_events=sum(len(doc['events']) for doc in gold_documents),
                  predicted_events=sum(map(len,documents)))
    return result


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--gold', type=Path, required=True)
    cli.add_argument('--sentence-data', type=Path, required=True)
    cli.add_argument('--predictions', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error('Output exists; choose a new path')
    paths = [args.gold,args.sentence_data,args.predictions,Path(__file__),
             FIXED/'ontology.py',FIXED/'linearize.py']
    hashes = {str(p.resolve()):sha(p) for p in paths}
    result = score(json.loads(args.gold.read_text()),read_jsonl(args.sentence_data),
                   read_jsonl(args.predictions))
    if hashes != {str(p.resolve()):sha(p) for p in paths}:
        raise RuntimeError('Source changed during evaluation')
    result.update(source_hashes=hashes, mode='hierarchical text presence; counterpart reuse allowed',
                  policy=__doc__, metric_scale='0..1; console prints percent')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    print(f"Documents={result['documents']}, sentences={result['sentence_rows']}, "
          f"gold_events={result['gold_events']}, predicted_events={result['predicted_events']}")
    print('metric    P(%)       R(%)      F1(%)     GoldHit PredHit      FP      FN')
    for metric in ('TI','TC','AI','AC','EC_I','EC_C'):
        values=' '.join(f'{result["metrics"][metric+suffix]*100:10.4f}' for suffix in ('_p','_r','_f1'))
        counts=' '.join(f'{result["counts"][metric][key]:7d}' for key in ('gold_matched','predicted_matched','fp','fn'))
        print(f'{metric:5} {values} {counts}')
    print('Saved:',args.output.resolve())


if __name__ == '__main__':
    main()
