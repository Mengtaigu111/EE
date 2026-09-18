import pytest


def gold(event_type, trigger, arguments):
    return {
        'event_type': event_type,
        'trigger': {'offsets': list(trigger)},
        'arguments': [
            {'argument_type': role, 'offsets': list(offsets)}
            for role, offsets in arguments
        ],
    }


def candidate(event_type, trigger, arguments=()):
    return {
        'type': event_type,
        'trigger_offsets': list(trigger),
        'args': [{'role': role, 'offsets': list(offsets)} for role, offsets in arguments],
    }


def test_labels_exact_event_and_argument_f1():
    from diagnostics import label_candidate

    result = label_candidate(
        candidate('PRP', [4], [('Content', [7]), ('Target', [9])]),
        [gold('PRP', [4], [('Content', [7]), ('Target', [9])])],
    )

    assert result == {
        'exists': 1,
        'type_ok': 1,
        'arg_p': 1.0,
        'arg_r': 1.0,
        'arg_f1': 1.0,
        'event_score': 1.0,
    }


def test_type_wrong_has_exists_but_zero_type_and_argument_score():
    from diagnostics import label_candidate

    result = label_candidate(
        candidate('FAC', [4], [('Subject', [7])]),
        [gold('PRP', [4], [('Content', [7])])],
    )

    assert result == {
        'exists': 1,
        'type_ok': 0,
        'arg_p': 0.0,
        'arg_r': 0.0,
        'arg_f1': 0.0,
        'event_score': 0.0,
    }


def test_same_trigger_multiple_gold_uses_best_type_match():
    from diagnostics import label_candidate

    result = label_candidate(
        candidate('PRP', [4], [('Content', [7])]),
        [
            gold('PRP', [4], [('Content', [7]), ('Target', [9])]),
            gold('PRP', [4], [('Content', [7])]),
        ],
    )

    assert result['type_ok'] == 1
    assert result['arg_p'] == pytest.approx(1.0)
    assert result['arg_r'] == pytest.approx(1.0)
    assert result['arg_f1'] == pytest.approx(1.0)


def test_type_correct_candidate_with_partial_arguments_gets_exact_f1():
    from diagnostics import label_candidate

    result = label_candidate(
        candidate('PRP', [4], [('Content', [7]), ('Target', [99])]),
        [gold('PRP', [4], [('Content', [7]), ('Target', [9])])],
    )

    assert result['exists'] == 1
    assert result['type_ok'] == 1
    assert result['arg_p'] == pytest.approx(0.5)
    assert result['arg_r'] == pytest.approx(0.5)
    assert result['arg_f1'] == pytest.approx(0.5)


def test_false_positive_trigger_has_zero_all_labels():
    from diagnostics import label_candidate

    result = label_candidate(candidate('PRP', [99], [('Content', [7])]), [])

    assert result == {
        'exists': 0,
        'type_ok': 0,
        'arg_p': 0.0,
        'arg_r': 0.0,
        'arg_f1': 0.0,
        'event_score': 0.0,
    }


def test_confusion_sets_are_derived_from_dev_trigger_matches_and_sorted():
    from diagnostics import build_confusion_sets

    result = build_confusion_sets([
        {
            'predicted': [candidate('FAC', [4])],
            'gold': [gold('PRP', [4], [])],
        },
        {
            'predicted': [candidate('FAC', [8])],
            'gold': [gold('PRP', [8], [])],
        },
        {
            'predicted': [candidate('CMP', [8])],
            'gold': [gold('PRP', [8], [])],
        },
    ], top_k=3)

    assert result['PRP'] == ['FAC', 'CMP']
    assert result['FAC'] == ['PRP']


def test_relative_diagnostics_counts_recovery_and_type_correction():
    from diagnostics import relative_diagnostics

    initial = [candidate('FAC', [4]), candidate('PRP', [99])]
    result = [candidate('PRP', [4]), candidate('PRP', [8])]
    gold_events = [gold('PRP', [4], []), gold('PRP', [8], [])]
    metrics = relative_diagnostics(initial, result, gold_events)
    assert metrics['type_corrected'] == 1
    assert metrics['events_recovered'] == 1
    assert metrics['events_added_fp'] == 0
