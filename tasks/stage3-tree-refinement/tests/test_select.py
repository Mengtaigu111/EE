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
