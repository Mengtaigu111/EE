def test_take_event_subtree_stops_at_balanced_event_boundary():
    from type_competition import take_event_subtree

    text, closed = take_event_subtree(' (Content "a") (Target "b")) (FAC (Trigger "later"))', max_tokens=160)
    assert text == ' (Content "a") (Target "b"))'
    assert closed is True


def test_take_event_subtree_reports_limit_without_silent_truncation():
    from type_competition import take_event_subtree

    text, closed = take_event_subtree(' ' + '(A "x")' * 20, max_tokens=3)
    assert closed is False
    assert text.startswith(' (A "x")')


def test_filter_invalid_roles_drops_roles_not_allowed_by_selected_schema():
    from type_competition import filter_invalid_roles

    event = {'type': 'PRP', 'trigger': 'propose', 'args': [
        {'role': 'Content', 'text': 'method'},
        {'role': 'Arg1', 'text': 'method'},
    ]}
    filtered, dropped = filter_invalid_roles(event, {'PRP': {'Content', 'Target'}})
    assert filtered['args'] == [{'role': 'Content', 'text': 'method'}]
    assert dropped == ['Arg1']


def test_score_sum_and_mean_include_forced_prefix_tokens():
    from type_competition import candidate_score

    result = candidate_score([0.0, -1.0, -2.0, -3.0], forced_prefix_length=2)
    assert result == {'score_sum': -6.0, 'score_mean': -1.5, 'token_count': 4}


def test_parse_forced_candidate_rejects_unclosed_generation():
    from type_competition import parse_forced_candidate

    candidate, diagnostics = parse_forced_candidate([], 'PRP', 'propose', ' (Content "method"')
    assert candidate is None
    assert diagnostics['closed'] is False


def test_parse_forced_candidate_returns_complete_event_subtree():
    from type_competition import parse_forced_candidate

    candidate, diagnostics = parse_forced_candidate(
        [{'type': 'ITT', 'trigger': 'study', 'args': []}],
        'PRP', 'propose', ' (Content "method"))',
    )
    assert diagnostics['closed'] is True
    assert candidate['type'] == 'PRP'
    assert candidate['args'][0]['role'] == 'Content'
