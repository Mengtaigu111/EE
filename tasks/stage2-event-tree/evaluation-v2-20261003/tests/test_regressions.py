import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from align_events import align_forest
from linearize import canonical_events, linearize, parse_forest
from merge_sentence_predictions import merge_sentence_rows, reparse_raw_predictions
from evaluate import run_complex_eval


def event(kind='FAC', trigger='test', args=None):
    return {'type': kind, 'trigger': trigger, 'args': args or []}


def row(sentence_index, bounds, tokens, sentences):
    return {'id': f'd::sentence-{sentence_index}', 'document_id': 'd',
            'sentence_index': sentence_index, 'target_bounds': bounds,
            'tokens': tokens, 'sentences': sentences}


def test_trigger_history_stays_within_sentence_scope():
    tokens = 'we test . they test .'.split()
    events = [event('FAC'), event('PRP'), event('FAC')]
    scopes = [(0, 3), (3, 6), (3, 6)]
    aligned, _ = align_forest(tokens, ['we test .', 'they test .'], events, trigger_scopes=scopes)
    assert [x['trigger']['offsets'] for x in aligned] == [[1], [4], [4]]


@pytest.mark.parametrize('bad', ['(())', '(FAC (Trigger ()))'])
def test_empty_nested_nodes_do_not_crash(bad):
    events, diag = parse_forest('(ROOT ' + bad + ' (FAC (Trigger "test")))')
    assert events == [event()]
    assert diag['dropped_events'] + diag['dropped_args'] > 0
    single_events, single_diag = parse_forest('(ROOT ' + bad + ')')
    assert single_events == []
    assert single_diag['dropped_events'] + single_diag['dropped_args'] > 0


def test_same_type_occurrences_survive_merge_and_gold_roundtrip():
    tokens = 'we test and test .'.split()
    sentences = ['we test and test .']
    rows = [row(0, (0, 5), tokens, sentences)]
    gold = {'document': tokens, 'events': [
        {'event_type': 'FAC', 'trigger': {'offsets': [1]}, 'arguments': []},
        {'event_type': 'FAC', 'trigger': {'offsets': [3]}, 'arguments': []},
    ]}
    raw = linearize(canonical_events(gold))
    parsed, diag = parse_forest(raw)
    assert not diag['invalid']
    merged = merge_sentence_rows(rows, [{'id': rows[0]['id'], 'parsed_events': parsed}])
    assert [x['trigger']['offsets'] for x in merged[0]['aligned_events']] == [[1], [3]]


def test_shared_trigger_different_types_gold_roundtrip():
    tokens = 'we test .'.split()
    rows = [row(0, (0, 3), tokens, ['we test .'])]
    gold = {'document': tokens, 'events': [
        {'event_type': kind, 'trigger': {'offsets': [1]}, 'arguments': []}
        for kind in ('FAC', 'PRP')]}
    parsed, diagnostics = parse_forest(linearize(canonical_events(gold)))
    assert diagnostics['dropped_events'] == 0
    merged = merge_sentence_rows(rows, [{'id': rows[0]['id'], 'parsed_events': parsed}])
    assert [(x['event_type'], x['trigger']['offsets']) for x in merged[0]['aligned_events']] == [
        ('FAC', [1]), ('PRP', [1])]


def test_exact_duplicate_removed_after_alignment():
    tokens = 'we test .'.split()
    rows = [row(0, (0, 3), tokens, ['we test .'])]
    merged = merge_sentence_rows(rows, [{'id': rows[0]['id'], 'parsed_events': [event(), event()]}])
    assert len(merged[0]['aligned_events']) == 1
    assert merged[0]['merge_diagnostics']['duplicate_events'] == 1


def test_duplicate_identity_ignores_argument_order():
    tokens = 'we test results .'.split()
    rows = [row(0, (0, 4), tokens, ['we test results .'])]
    first = event(args=[{'role': 'Subject', 'text': 'we'}, {'role': 'Result', 'text': 'results'}])
    second = event(args=list(reversed(first['args'])))
    merged = merge_sentence_rows(rows, [{'id': rows[0]['id'], 'parsed_events': [first, second]}])
    assert len(merged[0]['aligned_events']) == 1


def test_merge_preserves_reparse_diagnostics():
    tokens = 'we test .'.split()
    rows = [row(0, (0, 3), tokens, ['we test .'])]
    merged = merge_sentence_rows(rows, [{'id': rows[0]['id'], 'parsed_events': [event()],
                                         'parse_diagnostics': {'dropped_args': 1,
                                                               'unknown_roles': 1}}])
    assert merged[0]['parse_diagnostics']['dropped_args'] == 1


def test_raw_reparse_ignores_stale_parsed_fields():
    [prediction] = reparse_raw_predictions([{'id': 'a', 'raw_output': '(ROOT (FAC (Trigger "test")))',
                                             'parsed_events': [event(trigger='stale')],
                                             'invalid': True}])
    assert prediction['parsed_events'] == [event()]
    assert prediction['invalid'] is False


def test_complex_subprocess_failure_raises(monkeypatch, tmp_path):
    class Result:
        returncode = 7
        stdout = ''
        stderr = 'failed'
    monkeypatch.setattr('evaluate.subprocess.run', lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match='exit_code=7'):
        run_complex_eval(tmp_path / 'prediction.json', sys.executable)


def test_complex_missing_subscore_raises(monkeypatch, tmp_path):
    class Result:
        returncode = 0
        stdout = 'Consecutive: AC F1=42.00\n'
        stderr = ''
    monkeypatch.setattr('evaluate.subprocess.run', lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match='missing='):
        run_complex_eval(tmp_path / 'prediction.json', sys.executable)
