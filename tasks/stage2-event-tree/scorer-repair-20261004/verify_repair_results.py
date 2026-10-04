"""Independent CPU-only review gates; exhaustive small-instance matching oracle."""
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from instance_scorer import METRICS, score_documents
import utils


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts_oracle(gold, pred, mode):
    expr = lambda m: tuple(m['offsets'] if mode == 'offset' else m['tokens'])
    prepared = []
    for doc in (gold, pred):
        triggers = {expr(e['trigger']) for e in doc}
        side = []
        for e in doc:
            typed = Counter((expr(a), a['argument_type']) for a in e['arguments'])
            untyped = Counter(expr(a) for a in e['arguments'])
            ec_typed = Counter({k: v for k, v in typed.items() if k[0] in triggers})
            ec_untyped = Counter({k: v for k, v in untyped.items() if k in triggers})
            side.append(((e['event_type'], expr(e['trigger'])),
                         {'AC': typed, 'AI': untyped, 'EC_C': ec_typed, 'EC_I': ec_untyped}))
        prepared.append(side)
    g, p = prepared
    pairs_g, pairs_p = defaultdict(list), defaultdict(list)
    for side, groups in ((g, pairs_g), (p, pairs_p)):
        for key, fields in side:
            groups[key].append(fields)
    tp = dict.fromkeys(METRICS, 0)
    tp['TI'] = sum((Counter(k[1] for k, _ in g) & Counter(k[1] for k, _ in p)).values())
    priority = ('AC', 'AI', 'EC_C', 'EC_I')
    for key in pairs_g.keys() & pairs_p.keys():
        gs, ps = pairs_g[key], pairs_p[key]
        if len(gs) > len(ps):
            gs, ps = ps, gs
        best = None
        for permutation in itertools.permutations(range(len(ps)), len(gs)):
            candidate = tuple(sum(sum((ge[name] & ps[pi][name]).values()) for ge, pi in zip(gs, permutation))
                              for name in priority)
            if best is None or candidate > best:
                best = candidate
        tp['TC'] += len(gs)
        for name, value in zip(priority, best):
            tp[name] += value
    out = {}
    for name in METRICS:
        ng = len(g) if name in ('TI', 'TC') else sum(sum(f[name].values()) for _, f in g)
        np = len(p) if name in ('TI', 'TC') else sum(sum(f[name].values()) for _, f in p)
        out[name] = {'tp': tp[name], 'fp': np-tp[name], 'fn': ng-tp[name]}
    return out


def fixtures():
    rng = random.Random(4281)
    def make():
        doc = []
        for _ in range(rng.randint(0, 5)):
            tr = rng.randrange(3)
            doc.append({'event_type': rng.choice(('A', 'B')), 'trigger': {'offsets': [tr], 'tokens': [str(tr)]},
                        'arguments': [{'offsets': [a], 'tokens': [str(a)], 'argument_type': rng.choice(('x', 'y'))}
                                      for a in (rng.randrange(4) for _ in range(rng.randrange(5)))]})
        return doc
    for _ in range(500):
        g, p = make(), make()
        for mode in ('offset', 'token'):
            actual = score_documents([g], [p], mode)
            assert actual['counts'] == counts_oracle(g, p, mode)
            pairs = actual['matching'][0]['pairs']
            assert len({x['gold_event_index'] for x in pairs}) == len(pairs)
            assert len({x['pred_event_index'] for x in pairs}) == len(pairs)
            rng.shuffle(g); rng.shuffle(p)
            for e in g+p:
                rng.shuffle(e['arguments'])
            assert score_documents([g], [p], mode)['counts'] == actual['counts']
            validate_counts(actual)
    return 1000


def validate_counts(score):
    for name, c in score['counts'].items():
        assert all(isinstance(v, int) and v >= 0 for v in c.values()), (name, c)
        p = c['tp']/(c['tp']+c['fp']) if c['tp']+c['fp'] else 0
        r = c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else 0
        f = 2*p*r/(p+r) if p+r else 0
        for suffix, value in [('p', p), ('r', r), ('f1', f)]:
            assert abs(score['metrics'][name+'_'+suffix]-value) < 1e-14


def results():
    path = HERE/'evaluation-results.json'
    if not path.exists():
        return {'status': 'pending', 'reason': 'evaluation-results.json is not ready'}
    data = json.loads(path.read_text())
    locks = json.loads((HERE/'historical-source-locks.json').read_text())
    for mapping in (locks, data['source_hashes']):
        for file, digest in mapping.items():
            assert sha(file) == digest, file
    sys.path.insert(0, str(HERE.parent/'curriculum'))
    from verify_sorted_first_run import fixed_evaluation_modules
    _, merge = fixed_evaluation_modules()
    parser = sys.modules['linearize']
    runs = {'baseline': 'C-AppendixNoCopy-42-20261002', 'Easy2': 'C-CL-Mean-AppendixNoCopy-Easy2-42-20261003',
            'Sorted1': 'C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003'}
    checked = 0
    for variant, splits in data['variants'].items():
        for split, steps in splits.items():
            gold = json.loads((ROOT/'data/SciEvents'/f'{split}.json').read_text())
            for step, record in steps.items():
                for field in ('instance_offset', 'instance_aligned_token', 'instance_raw_token'):
                    validate_counts(record[field])
                checked += 1
                rows = [json.loads(line) for line in (HERE.parent/'data/sentence-c-appendix-no-copy-20261002'/f'{split}.jsonl').read_text().splitlines()]
                predictions = [json.loads(line) for line in (HERE.parent/'runs'/runs[variant]/'checkpoint-eval'/f'{split}-checkpoint-{step}.predictions.jsonl').read_text().splitlines()]
                assert len({r['id'] for r in rows}) == len(rows)
                assert len({p['id'] for p in predictions}) == len(predictions)
                assert {r['id'] for r in rows} == {p['id'] for p in predictions}
                by_id = {p['id']: p for p in predictions}
                raw_by_doc = defaultdict(list)
                for row in sorted(rows, key=lambda r: (r['document_id'], r['sentence_index'])):
                    events, _ = parser.parse_forest(by_id[row['id']]['raw_output'])
                    raw_by_doc[row['document_id']].extend({'event_type': e['type'], 'trigger': {'tokens': e['trigger'].split()},
                        'arguments': [{'tokens': a['text'].split(), 'argument_type': a['role']} for a in e['args']]} for e in events)
                aligned_rows = merge.merge_sentence_rows(rows, merge.reparse_raw_predictions(predictions))
                aligned_by_doc = {d['document_id']: d['aligned_events'] for d in aligned_rows}
                assert len(aligned_by_doc) == len(aligned_rows) == len(gold)
                assert set(aligned_by_doc) == {d['doc_id'] for d in gold} == set(raw_by_doc)
                gs = [d['events'] for d in gold]
                for ps, mode, field, old_field in [([aligned_by_doc[d['doc_id']] for d in gold], 'offset', 'instance_offset', 'official_offset'),
                    ([aligned_by_doc[d['doc_id']] for d in gold], 'token', 'instance_aligned_token', 'official_aligned_token'),
                    ([raw_by_doc[d['doc_id']] for d in gold], 'token', 'instance_raw_token', 'official_raw_token')]:
                    scored = score_documents(gs, ps, mode)
                    assert {k: scored[k] for k in ('counts', 'metrics')} == record[field], (variant, split, step, field)
                    assert dict(zip(scored['metrics'], map(float, utils.calculate_F1(gs, ps, mode)))) == record[old_field]
                for name in METRICS:
                    for field in ('instance_offset', 'instance_aligned_token'):
                        assert record[field]['counts'][name]['tp'] <= record[field]['counts'][name]['tp']+record[field]['counts'][name]['fn']
                if split != 'test':
                    continue
                g = [d['events'] for d in gold]
                for suffix, fields in [('aligned', [('offset', 'instance_offset', 'official_offset'),
                                                      ('token', 'instance_aligned_token', 'official_aligned_token')]),
                                      ('raw-token', [('token', 'instance_raw_token', 'official_raw_token')])]:
                    docs = json.loads((HERE/f'{variant}.{suffix}.predictions.json').read_text())
                    assert [d['doc_id'] for d in docs] == [d['doc_id'] for d in gold]
                    p = [d['events'] for d in docs]
                    for mode, field, official in fields:
                        score = score_documents(g, p, mode)
                        assert score['metrics'] == record[field]['metrics']
                        assert score['counts'] == record[field]['counts']
                        old = list(map(float, utils.calculate_F1(g, p, mode)))
                        assert dict(zip(score['metrics'], old)) == record[official]
                trace = data['TI_mode_decomposition'][variant]
                assert trace['token_tp']-trace['same_prediction_set_instance_TI_offset_tp'] == len(trace['matches'])
                assert len(trace['matches']) == trace['extra_same_text_different_position_matches']
                assert all(m['gold_offsets'] != m['prediction_offsets'] for m in trace['matches'])
        for field, step in data['dev_selections'][variant].items():
            dev = splits['dev']
            def val(s):
                score = dev[s][field]
                return (score['metrics']['AC_f1'] if 'metrics' in score else score['AC_f1'], int(s))
            assert int(max(dev, key=val)) == step
    return {'status': 'passed', 'records': checked, 'test_prediction_artifacts_recomputed': 6, 'all_18_records_original_sentence_predictions_rebuilt': True,
            'historical_and_current_source_hashes_verified': len(set(locks) | set(data['source_hashes']))}


def main():
    out = {'small_instance_exhaustive_oracle_cases': fixtures(), 'gold_self': {}}
    for split in ('dev', 'test'):
        docs = [d['events'] for d in json.loads((ROOT/'data/SciEvents'/f'{split}.json').read_text())]
        for mode in ('offset', 'token'):
            score = score_documents(docs, docs, mode)
            assert all(v == 1 for v in score['metrics'].values())
            validate_counts(score)
            out['gold_self'][split+'_'+mode] = 'all 18 metrics equal 1'
    out['results'] = results()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
