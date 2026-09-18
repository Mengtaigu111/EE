"""Seven-way error analysis for Stage-2 predictions (v2 spec §21), on official-format outputs.

Trigger side (per gold event): MATCH / TYPE_ERROR (offsets match, type differs) /
TRIGGER_BOUNDARY (overlapping predicted trigger) / TRIGGER_MISSING.
Argument side (per gold argument of a type-matched event): MATCH / ROLE_ERROR /
ARG_BOUNDARY / WRONG_ATTACHMENT (same span predicted under another event) / ARG_MISSING.
Also: FP events by trigger status, per-event-type P/R, per-role P/R.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def key(o):
    return tuple(o)


def analyze(gold_docs, pred_docs):
    trig = Counter(); arg = Counter(); fp_events = Counter()
    type_conf = Counter(); role_conf = Counter()
    per_type = defaultdict(Counter); per_role = defaultdict(Counter)
    for g, p in zip(gold_docs, pred_docs):
        gev, pev = g['events'], p['events']
        p_by_trig = {}
        for e in pev:
            p_by_trig.setdefault(key(e['trigger']['offsets']), []).append(e)
        p_arg_spans = defaultdict(set)  # span -> set of predicted trigger keys
        for e in pev:
            for a in e['arguments']:
                p_arg_spans[key(a['offsets'])].add(key(e['trigger']['offsets']))
        g_trig_keys = {key(e['trigger']['offsets']) for e in gev}
        for e in gev:
            t = key(e['trigger']['offsets']); per_type[e['event_type']]['gold'] += 1
            cands = p_by_trig.get(t, [])
            same = [c for c in cands if c['event_type'] == e['event_type']]
            if same:
                trig['MATCH'] += 1; per_type[e['event_type']]['tp'] += 1
                pe = same[0]
                p_args = {}
                for a in pe['arguments']:
                    p_args.setdefault(key(a['offsets']), []).append(a['argument_type'])
                for a in e['arguments']:
                    s = key(a['offsets']); per_role[a['argument_type']]['gold'] += 1
                    if s in p_args:
                        if a['argument_type'] in p_args[s]:
                            arg['MATCH'] += 1; per_role[a['argument_type']]['tp'] += 1
                        else:
                            arg['ROLE_ERROR'] += 1; role_conf[(a['argument_type'], p_args[s][0])] += 1
                    elif any(set(s) & set(ps) for ps in p_args):
                        arg['ARG_BOUNDARY'] += 1
                    elif s in p_arg_spans:
                        arg['WRONG_ATTACHMENT'] += 1
                    else:
                        arg['ARG_MISSING'] += 1
            elif cands:
                trig['TYPE_ERROR'] += 1; type_conf[(e['event_type'], cands[0]['event_type'])] += 1
                arg['LOST_BY_TYPE_ERROR'] += len(e['arguments'])
            elif any(set(t) & set(k) for k in p_by_trig):
                trig['TRIGGER_BOUNDARY'] += 1; arg['LOST_BY_TRIGGER'] += len(e['arguments'])
            else:
                trig['TRIGGER_MISSING'] += 1; arg['LOST_BY_TRIGGER'] += len(e['arguments'])
        for e in pev:
            t = key(e['trigger']['offsets']); per_type[e['event_type']]['pred'] += 1
            for a in e['arguments']:
                per_role[a['argument_type']]['pred'] += 1
            if t in g_trig_keys:
                if not any(ge['event_type'] == e['event_type'] and key(ge['trigger']['offsets']) == t for ge in gev):
                    fp_events['FP_TYPE_ERROR'] += 1
            elif any(set(t) & set(key(ge['trigger']['offsets'])) for ge in gev):
                fp_events['FP_TRIGGER_BOUNDARY'] += 1
            else:
                fp_events['FP_SPURIOUS_TRIGGER'] += 1
    def prf(c):
        p = c['tp'] / c['pred'] if c['pred'] else 0; r = c['tp'] / c['gold'] if c['gold'] else 0
        return {'gold': c['gold'], 'pred': c['pred'], 'P': round(100 * p, 1), 'R': round(100 * r, 1), 'F1': round(200 * p * r / (p + r), 1) if p + r else 0}
    return {'trigger': dict(trig), 'argument': dict(arg), 'fp_events': dict(fp_events),
            'type_confusion_top': [(f'{a}->{b}', n) for (a, b), n in type_conf.most_common(8)],
            'role_confusion_top': [(f'{a}->{b}', n) for (a, b), n in role_conf.most_common(8)],
            'per_event_type': {t: prf(c) for t, c in sorted(per_type.items())},
            'per_role': {r: prf(c) for r, c in sorted(per_role.items(), key=lambda x: -x[1]['gold'])}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True, help='*.metrics.scievents.json')
    parser.add_argument('--split', default='test')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    gold = {d['doc_id']: d for d in json.loads((PROJECT_ROOT / f'data/SciEvents/{args.split}.json').read_text())}
    preds = json.loads(args.predictions.read_text())
    result = analyze([gold[p['doc_id']] for p in preds], preds)
    text = json.dumps(result, ensure_ascii=False, indent=1)
    if args.output:
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
