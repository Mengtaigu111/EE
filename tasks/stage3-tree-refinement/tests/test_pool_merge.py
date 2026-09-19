def event(event_type, trigger, args=()):
    return {
        'type': event_type,
        'trigger_offsets': list(trigger),
        'args': [{'role': role, 'offsets': list(offsets)} for role, offsets in args],
    }


def test_merge_deduplicates_by_event_key_and_counts_distinct_sources():
    from sample_pool import merge_document_candidates

    shared = event('PRP', [2], [('Content', [4])])
    result = merge_document_candidates([
        {'source': 'c0', 'events': [shared]},
        {'source': 'c1', 'events': [shared, event('FAC', [8])]},
        {'source': 'c2', 'events': [shared]},
    ])

    assert len(result) == 2
    first = next(row for row in result if row['type'] == 'PRP')
    assert first['support'] == 3
    assert first['sources'] == ['c0', 'c1', 'c2']


def test_merge_is_stable_and_records_forest_diversity():
    from sample_pool import forest_signature, merge_document_candidates

    a = event('PRP', [2])
    b = event('FAC', [8])
    assert forest_signature([b, a]) == forest_signature([a, b])
    result = merge_document_candidates([
        {'source': 'c2', 'events': [b, a]},
        {'source': 'c0', 'events': [a]},
    ])
    assert [row['type'] for row in result] == ['PRP', 'FAC']


def test_candidate_from_aligned_event_uses_spec_offsets_and_roles():
    from sample_pool import candidate_from_aligned_event

    result = candidate_from_aligned_event({
        'event_type': 'PRP',
        'trigger': {'text': 'propose', 'offsets': [6]},
        'arguments': [{'argument_type': 'Content', 'text': 'method', 'offsets': [9]}],
    })
    assert result == {
        'type': 'PRP', 'trigger_offsets': [6], 'trigger_text': 'propose',
        'args': [{'role': 'Content', 'offsets': [9], 'text': 'method'}],
    }


def test_align_text_events_uses_document_level_t0_parameters():
    from sample_pool import align_text_events

    candidates, stats = align_text_events(
        {'tokens': ['we', 'propose', 'method'], 'sentences': ['we propose method']},
        [{'type': 'PRP', 'trigger': 'propose', 'args': [{'role': 'Content', 'text': 'method'}]}],
    )
    assert candidates[0]['trigger_offsets'] == [1]
    assert candidates[0]['args'][0]['offsets'] == [2]
    assert stats['aligned_events'] == 1


def test_exists_oracle_keeps_max_event_score_per_gold_type_trigger():
    from sample_pool import build_pool_row, oracle_candidates

    sources = [
        {'source': 'c0', 'events': [event('PRP', [2], [('Content', [4])])]},
        {'source': 'c1', 'events': [event('PRP', [2], [('Content', [4]), ('Target', [5])])]},
    ]
    gold = [{'event_type': 'PRP', 'trigger': {'offsets': [2]},
             'arguments': [{'argument_type': 'Content', 'offsets': [4]},
                           {'argument_type': 'Target', 'offsets': [5]}]}]
    row = build_pool_row('doc', sources, gold)
    oracle = oracle_candidates(row['candidates'], gold)
    assert len(oracle) == 1
    assert oracle[0]['args'][-1]['offsets'] == [5]


def test_type_competition_candidates_are_aligned_and_tagged():
    from sample_pool import aligned_type_competition_candidates

    row = {'tokens': ['we', 'propose', 'method'], 'sentences': ['we propose method']}
    competitor = {'candidates': [{
        'event_index': 0, 'type': 'FAC', 'closed': True, 'score_sum': -2.0,
        'candidate': {'type': 'FAC', 'trigger': 'propose',
                      'args': [{'role': 'Subject', 'text': 'we'}]},
    }]}
    result = aligned_type_competition_candidates(row, competitor)
    assert result[0]['type'] == 'FAC'
    assert result[0]['trigger_offsets'] == [1]
    assert result[0]['source'] == 'type_competition'
    assert result[0]['score_sum'] == -2.0
