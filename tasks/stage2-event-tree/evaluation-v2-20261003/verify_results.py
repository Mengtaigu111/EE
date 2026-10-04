#!/usr/bin/env python3
"""Independent artifact and official-score verification; reads v2 output only."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import utils

STEPS = {'curriculum': (802, 814, 1604, 2406, 3208, 4010),
         'baseline': (802, 1604, 2406, 3208, 4010)}
NAMES = ['TI_f1', 'TC_f1', 'AI_f1', 'AC_f1', 'EC_I_f1', 'EC_C_f1',
         'TI_p', 'TC_p', 'AI_p', 'AC_p', 'EC_I_p', 'EC_C_p',
         'TI_r', 'TC_r', 'AI_r', 'AC_r', 'EC_I_r', 'EC_C_r']
COMPLEX = {'Consecutive_AC_f1', 'Overlap_TC_f1', 'Overlap_AC_f1', 'Reverse_AC_f1',
           'Subevent_TC_f1', 'Subevent_AC_f1', 'Subevent_EC_f1'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def verify_one(group, split, step):
    base = HERE / 'results' / group / f'{split}-checkpoint-{step}'
    metrics_path = base.with_suffix('.metrics.json')
    provenance_path = base.with_suffix('.provenance.json')
    merged_path = base.with_suffix('.merged.jsonl')
    scievents_path = base.with_suffix('.metrics.scievents.json')
    record, metrics = load(provenance_path), load(metrics_path)
    assert record['metrics_sha256'] == sha(metrics_path)
    assert record['merged_sha256'] == sha(merged_path)
    inp = record['input']
    assert inp['raw_sha256'] == sha(inp['raw_path'])
    marker = load(inp['raw_path'] + '.complete.json')
    assert marker['status'] == 'completed' and marker['output_sha256'] == inp['raw_sha256']
    assert inp['complete_marker_sha256'] == sha(inp['raw_path'] + '.complete.json')
    assert inp['adapter_sha256'] == sha(Path(inp['checkpoint']) / 'adapter_model.safetensors')
    assert record['document_data_sha256'] == sha(record['document_data'])
    assert all(record['code_sha256'][p.name] == sha(p) for p in (HERE / 'scripts').glob('*.py'))
    gold = {x['id']: x['gold_events'] for x in
            (json.loads(line) for line in Path(record['document_data']).read_text().splitlines() if line)}
    predictions = load(scievents_path)
    ids = [x['doc_id'] for x in predictions]
    assert len(ids) == len(set(ids)) == len(gold) == metrics['documents']
    assert set(ids) == set(gold)
    scores = utils.calculate_F1([gold[i] for i in ids], [x['events'] for x in predictions], 'offset')
    assert all(abs(metrics['official'][key] - score) < 1e-12 for key, score in zip(NAMES, scores))
    assert metrics['scope'] == 'full'
    if split == 'test':
        assert metrics['complex']['exit_code'] == 0
        assert set(metrics['complex']['scores']) == COMPLEX
    return metrics['official']['AC_f1']


def main():
    summary = load(HERE / 'results/dev_selection.json')
    checked = {}
    for group, steps in STEPS.items():
        scores = {str(step): verify_one(group, 'dev', step) for step in steps}
        assert scores == summary[group]['scores']
        assert summary[group]['best_step'] == max(steps, key=lambda s: (scores[str(s)], s))
        checked[group] = {'dev_count': len(steps), 'best_step': summary[group]['best_step']}
        test_path = HERE / 'results' / group / f"test-checkpoint-{summary[group]['best_step']}.provenance.json"
        if test_path.exists():
            checked[group]['test_AC_f1'] = verify_one(group, 'test', summary[group]['best_step'])
    manifest = load(HERE / 'version_manifest.json')
    locked = manifest['old_locked_sources']
    assert all(sha(path) == data['expected'] for path, data in locked.items())
    assert all(sha(ROOT / path) == digest for path, digest in manifest['v2_files'].items())
    assert sha(ROOT / 'utils.py') == manifest['official_scorer_sha256']
    assert sha(ROOT / 'eval_complex_quick_f1.py') == manifest['complex_scorer_sha256']
    print(json.dumps({'verified': checked, 'locked_source_count': len(locked)}, indent=2))


if __name__ == '__main__':
    main()
