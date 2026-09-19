def scored(event_type, trigger, score, source='pool'):
    return {
        'type': event_type, 'trigger_offsets': list(trigger), 'args': [],
        'reward_score': score, 'source': source,
    }


def test_r0_keeps_one_best_candidate_per_type_and_all_types_above_tau():
    from select_r0 import filter_candidates

    result = filter_candidates([
        scored('PRP', [2], 0.8),
        scored('PRP', [2], 0.7),
        scored('FAC', [2], 0.75),
        scored('CMP', [9], 0.2),
    ], tau=0.7)
    assert [(row['type'], row['reward_score']) for row in result] == [
        ('PRP', 0.8), ('FAC', 0.75)
    ]


def test_r1_selects_highest_reward_with_stable_tie_break():
    from select_r1 import select_type_competitor

    result = select_type_competitor([
        scored('FAC', [2], 0.8),
        scored('PRP', [2], 0.8),
        scored('CMP', [2], 0.7),
    ])
    assert result['type'] == 'FAC'


def test_tau_grid_is_dev_only_and_has_inclusive_step():
    from select_r0 import tau_grid

    assert tau_grid() == [round(0.1 + 0.05 * index, 2) for index in range(17)]


def test_r1_document_replaces_only_event_slots_and_preserves_other_events():
    from select_r1 import select_document_events

    base = [scored('PRP', [2], 0.0), scored('FAC', [9], 0.0)]
    competitors = [
        {'event_index': 0, 'candidate': scored('CMP', [2], 0.9), 'score_sum': -1.0},
        {'event_index': 0, 'candidate': scored('PRP', [2], 0.8), 'score_sum': -2.0},
    ]
    result = select_document_events(base, competitors)
    assert [event['type'] for event in result] == ['CMP', 'FAC']


def test_r0_same_trigger_keeps_different_types_but_deduplicates_each_type():
    from select_r0 import filter_candidates

    result = filter_candidates([
        scored('PRP', [2], 0.8), scored('PRP', [2], 0.8),
        scored('FAC', [2], 0.75), scored('CMP', [2], 0.05),
    ], tau=0.7)
    assert [(row['type'], row['reward_score']) for row in result] == [
        ('PRP', 0.8), ('FAC', 0.75)]


def test_r0_caps_trigger_ambiguity_at_two_types():
    from select_r0 import filter_candidates

    result = filter_candidates([
        scored('PRP', [2], 0.8), scored('FAC', [2], 0.75), scored('CMP', [2], 0.7),
    ], tau=0.6)
    assert [row['type'] for row in result] == ['PRP', 'FAC']


def test_r0_can_select_dev_tau_by_official_ac_and_returns_curve():
    from select_r0 import choose_tau

    candidates = [scored('PRP', [2], 0.8), scored('FAC', [9], 0.2)]
    def metric(rows):
        return {'AC_f1': 1.0 if len(rows) == 1 else 0.5}

    tau, curve = choose_tau(candidates, metric)
    assert tau == 0.25
    assert curve[0]['tau'] == 0.1
    assert len(curve) == 17
