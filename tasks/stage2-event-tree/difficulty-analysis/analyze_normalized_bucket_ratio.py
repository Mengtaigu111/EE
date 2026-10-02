"""Rebucket existing normalized train scores without changing the difficulty formula."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent


def analyze(easy_fraction=0.6, boundary_ties='split'):
    if not 0 < easy_fraction < 1:
        raise ValueError('easy_fraction must lie strictly between 0 and 1')
    source = BASE / 'normalized_difficulty_sentence_level.csv'
    with source.open(encoding='utf8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    positive = sorted((r for r in rows if int(r['gold_event_count']) > 0),
                      key=lambda r: (float(r['difficulty_score_normalized']), r['sample_id']))
    cutoff = int(len(positive) * easy_fraction + 0.5)
    if cutoff == 0 or cutoff == len(positive):
        raise ValueError('Requested ratio leaves a bucket empty')
    boundary = float(positive[cutoff - 1]['difficulty_score_normalized'])
    if boundary_ties == 'easy':
        cutoff = sum(float(r['difficulty_score_normalized']) <= boundary for r in positive)
    groups = {r['sample_id']: ('B1' if i < cutoff else 'B2') for i, r in enumerate(positive)}
    ties = [r for r in positive if float(r['difficulty_score_normalized']) == boundary]
    compositions = {}
    type_counts = {'B1': Counter(), 'B2': Counter()}
    for b in ('B1', 'B2'):
        selected = [r for r in positive if groups[r['sample_id']] == b]
        counts = Counter('4+' if int(r['gold_event_count']) >= 4 else r['gold_event_count'] for r in selected)
        compositions[b] = {c: {'count': counts[c], 'percent': counts[c] / len(selected)}
                           for c in ('1', '2', '3', '4+')}
        for row in selected:
            type_counts[b].update(row['event_types'].split('|'))
    type_rates = {t: {'B1': type_counts['B1'][t], 'B2': type_counts['B2'][t],
                     'B1_rate': type_counts['B1'][t] / (type_counts['B1'][t] + type_counts['B2'][t])}
                  for t in sorted(type_counts['B1'].keys() | type_counts['B2'].keys())}
    summary = {
        'boundary_tie_policy': boundary_ties,
        'easy_fraction_requested': easy_fraction, 'total': len(rows), 'positive': len(positive),
        'ROOT': len(rows) - len(positive), 'B1': cutoff, 'B2': len(positive) - cutoff,
        'B1_actual_fraction': cutoff / len(positive),
        'boundary_score': boundary, 'B1_max_score': boundary,
        'B2_min_score': float(positive[cutoff]['difficulty_score_normalized']),
        'boundary_ties': len(ties),
        'boundary_ties_B1': sum(groups[r['sample_id']] == 'B1' for r in ties),
        'boundary_ties_B2': sum(groups[r['sample_id']] == 'B2' for r in ties),
        'tie_preserving': {k: sum(compare(float(r['difficulty_score_normalized']), boundary) for r in positive)
                          for k, compare in [('below', lambda x, y: x < y),
                                             ('equal', lambda x, y: x == y),
                                             ('above', lambda x, y: x > y)]},
        'event_count_composition': compositions, 'event_type_assignment': type_rates,
    }
    prefix = f'normalized_difficulty_{easy_fraction * 100:.0f}_{(1-easy_fraction) * 100:.0f}'
    if boundary_ties != 'split':
        prefix += '_ties_easy'
    with (BASE / 'curriculum_d1_d2_assignment.csv').open(encoding='utf8') as stream:
        manual = {r['sample_id']: r['difficulty_group'] for r in csv.DictReader(stream)}
    summary['agreement_old_50_50'] = sum(groups[r['sample_id']] == r['difficulty_bucket_old'] for r in positive) / len(positive)
    summary['agreement_manual'] = sum(groups[r['sample_id']] == {'D1': 'B1', 'D2': 'B2'}[manual[r['sample_id']]] for r in positive) / len(positive)
    rates = [x['B1_rate'] for x in type_rates.values()]
    summary['type_B1_rate_range'] = max(rates) - min(rates)
    assignments = [{
        'sample_id': r['sample_id'], 'gold_event_count': r['gold_event_count'],
        'difficulty_score_normalized': r['difficulty_score_normalized'],
        'difficulty_bucket': groups.get(r['sample_id'], 'ROOT'),
    } for r in rows]
    assert len({r['sample_id'] for r in rows}) == len(rows)
    assert summary['B1'] + summary['B2'] == summary['positive']
    assert summary['ROOT'] + summary['positive'] == summary['total']
    assert sum(type_counts[b].total() if hasattr(type_counts[b], 'total') else sum(type_counts[b].values())
               for b in ('B1', 'B2')) == sum(int(r['gold_event_count']) for r in positive)
    with (BASE / f'{prefix}_assignment.csv').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignments[0]))
        writer.writeheader()
        writer.writerows(assignments)
    (BASE / f'{prefix}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    lines = [f'# Normalized difficulty {easy_fraction:.0%}:{1-easy_fraction:.0%} split', '',
             f'Positive={len(positive)}; ROOT={summary["ROOT"]}; B1={cutoff} ({cutoff/len(positive):.4%}); B2={len(positive)-cutoff} ({1-cutoff/len(positive):.4%}).', '',
             'Round the requested easy count to the nearest integer; sort by normalized score, then sample_id. ROOT remains unscored.', '',
             '| Bucket | 1 event | 2 events | 3 events | 4+ events |', '|---|---:|---:|---:|---:|']
    for b in ('B1', 'B2'):
        lines.append('| ' + b + ' | ' + ' | '.join(f'{compositions[b][c]["count"]} ({compositions[b][c]["percent"]:.2%})'
                                                   for c in ('1', '2', '3', '4+')) + ' |')
    lines += ['', f'Boundary={boundary:.9f}; tied={len(ties)}, B1/B2={summary["boundary_ties_B1"]}/{summary["boundary_ties_B2"]}. Tie splitting is administrative only.', '',
              f'Tie-preserving alternative below/equal/above={summary["tie_preserving"]}.', '',
              '| Event type | B1 events | B2 events | % of type in B1 |', '|---|---:|---:|---:|']
    for t, x in type_rates.items():
        lines.append(f'| {t} | {x["B1"]} | {x["B2"]} | {x["B1_rate"]:.2%} |')
    lines += ['', f'Type B1 assignment-rate range={summary["type_B1_rate_range"]:.2%}; agreement with old 50:50 buckets={summary["agreement_old_50_50"]:.2%}; agreement with manual D1/D2={summary["agreement_manual"]:.2%}.', '',
              f'Boundary tie policy={boundary_ties}. The difficulty formula and scores are unchanged.']
    (BASE / f'{prefix}_report.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--easy-fraction', type=float, default=0.6)
    parser.add_argument('--boundary-ties', choices=['split', 'easy'], default='split')
    args = parser.parse_args()
    print(json.dumps(analyze(args.easy_fraction, args.boundary_ties), ensure_ascii=False, indent=2))
