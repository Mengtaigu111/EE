"""Train-only diagnostic: average event type-local percentiles within each sentence."""
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
PREFIX = 'mean_percentile_difficulty'


def read(name):
    with (BASE / name).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (BASE / (PREFIX + '_' + name + '.csv')).open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    xs = sorted(values)
    def q(p):
        pos = (len(xs) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return dict(count=len(xs), mean=statistics.mean(xs), std=statistics.pstdev(xs),
                min=xs[0], max=xs[-1], p10=q(.1), p25=q(.25), p50=q(.5), p75=q(.75), p90=q(.9))


def main():
    events = read('schema_completeness_event_level.csv')
    prior = read('normalized_difficulty_sentence_level.csv')
    by_sample = defaultdict(list)
    by_type = defaultdict(Counter)
    for event in events:
        by_sample[event['sample_id']].append(event)
        by_type[event['event_type']][float(event['schema_completeness'])] += 1
    pmap = {}
    for typ, counts in by_type.items():
        below = 0
        total = sum(counts.values())
        for completeness, count in sorted(counts.items()):
            pmap[typ, completeness] = (below + .5 * count) / total
            below += count
    rows = []
    for old in prior:
        n = int(old['gold_event_count'])
        es = by_sample[old['sample_id']]
        assert len(es) == n
        ps = [pmap[e['event_type'], float(e['schema_completeness'])] for e in es]
        for e, p in zip(es, ps):
            assert abs(p - float(e['within_type_completeness_percentile'])) < 1e-12
        mean_p = statistics.mean(ps) if n else None
        score = .5 * ((1 - 1 / n) + (1 - mean_p)) if n else None
        old_score = float(old['difficulty_score_normalized']) if n else None
        if n:
            assert abs(old_score - .5 * ((1 - 1 / n) + (1 - min(ps)))) < 1e-12
            assert 0 <= score <= old_score + 1e-12 <= 1 + 1e-12
            assert n != 1 or abs(score - old_score) < 1e-12
        rows.append(dict(sample_id=old['sample_id'], gold_event_count=n, event_types=old['event_types'],
                         event_percentiles='|'.join(repr(p) for p in ps),
                         mean_percentile=mean_p if n else 'NA',
                         score_mean=score if n else 'NA', score_min=old_score if n else 'NA',
                         score_decrease=old_score - score if n else 'NA'))
    positive = [r for r in rows if r['gold_event_count']]
    assert len(rows) == 12828 and len(positive) == 10704 and len(events) == 19488
    assert len({r['sample_id'] for r in rows}) == len(rows)
    values = [r['score_mean'] for r in positive]
    frequencies = Counter(values)
    distribution = []
    for i in range(10):
        new_count = sum(i / 10 <= r['score_mean'] < (i + 1) / 10 for r in positive)
        old_count = sum(i / 10 <= r['score_min'] < (i + 1) / 10 for r in positive)
        distribution.append(dict(lower=i / 10, upper=(i + 1) / 10, count_mean=new_count,
                                 percent_mean=100 * new_count / len(positive), count_min=old_count,
                                 percent_min=100 * old_count / len(positive)))
    assert sum(r['count_mean'] for r in distribution) == len(positive)
    order = sorted(positive, key=lambda r: (r['score_mean'], r['sample_id']))
    split_summaries = {}
    assignments = []
    for fraction in (.5, .6):
        cutoff = int(len(order) * fraction + .5)
        boundary = order[cutoff - 1]['score_mean']
        below = sum(v < boundary for v in values)
        equal = frequencies[boundary]
        compositions = {}
        for policy, end in [('split', cutoff), ('all_ties_easy', below + equal)]:
            ids = {r['sample_id'] for r in order[:end]}
            groups = {'B1': order[:end], 'B2': order[end:]}
            compositions[policy] = {
                'B1': end, 'B2': len(order) - end, 'B1_percent': end / len(order) * 100,
                'event_counts': {b: dict(Counter(str(r['gold_event_count']) if r['gold_event_count'] < 4 else '4+'
                                                for r in selected)) for b, selected in groups.items()}}
            for r in rows:
                assignments.append(dict(sample_id=r['sample_id'], fraction=fraction, tie_policy=policy,
                                        score_mean=r['score_mean'], bucket='ROOT' if not r['gold_event_count']
                                        else 'B1' if r['sample_id'] in ids else 'B2'))
        split_summaries[str(fraction)] = dict(boundary=boundary, below=below, ties=equal,
                                             above=len(order)-below-equal, policies=compositions)
    old_boundary = .5457078313253012
    old_ties = [r for r in positive if r['score_min'] == old_boundary]
    old_tie_freq = Counter(r['score_mean'] for r in old_ties)
    summary = dict(formula='0.5*((1-1/N)+(1-mean(event_type_local_percentiles)))',
                   total=len(rows), ROOT=len(rows)-len(positive), positive=len(positive),
                   stats_mean=stats(values), stats_min=stats([r['score_min'] for r in positive]),
                   distinct_exact_mean_scores=len(frequencies),
                   distinct_scores_rounded_12=len({round(v, 12) for v in values}),
                   distinct_exact_min_scores=len({r['score_min'] for r in positive}),
                   largest_ties=[dict(score=v, count=c, percent=100*c/len(positive))
                                 for v, c in frequencies.most_common(10)], splits=split_summaries,
                   by_event_count={str(n):stats([r['score_mean'] for r in positive if r['gold_event_count']==n])
                                   for n in sorted({r['gold_event_count'] for r in positive})},
                   old_boundary_ties=dict(count=len(old_ties), new_stats=stats([r['score_mean'] for r in old_ties]),
                                          distinct_scores=len(old_tie_freq),
                                          largest_ties=[dict(score=v,count=c) for v,c in old_tie_freq.most_common(10)]))
    write('sentence_level', rows)
    write('distribution', distribution)
    write('exact_frequencies', [dict(score=v, count=c, percent=100*c/len(positive))
                                for v,c in sorted(frequencies.items())])
    write('assignments', assignments)
    (BASE / (PREFIX + '_summary.json')).write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Mean-percentile normalized curriculum difficulty', '',
             'Train-only. Full-precision type-local midrank percentiles, averaged equally across event instances.',
             'DIF = 0.5 * [(1 - 1/N) + (1 - mean(P_1,...,P_N))]. ROOT unscored. Prior results preserved.', '',
             '## Overall statistics', '', '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '',
             '## Score distribution', '', '| Interval | Mean formula | Min formula |', '|---|---:|---:|']
    for r in distribution:
        lines.append('| [{lower:.1f},{upper:.1f}) | {count_mean} ({percent_mean:.2f}%) | {count_min} ({percent_min:.2f}%) |'.format(**r))
    lines += ['', '## Verification', '',
              'Recomputed type-local midranks match event-table full-precision percentiles; event/sentence counts reconcile.',
              'All scores stay in [0,1]; mean-based DIF never exceeds min-based DIF; all single-event scores are unchanged.',
              'Bucket totals, histogram totals, and ROOT counts reconcile. Exact float ties are used for allocation;',
              'distinct scores rounded to 12 decimals are also reported to expose floating-point artifacts.',
              'Schema coverage and its percentile remain proxies for difficulty, not measured model error.']
    (BASE / (PREFIX + '_report.md')).write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
