def test_fold_assignment_is_sha256_parity_and_document_disjoint():
    from build_folds import assign_fold, split_rows

    rows = [
        {'document_id': 'doc-a', 'value': 1},
        {'document_id': 'doc-b', 'value': 2},
        {'document_id': 'doc-a', 'value': 3},
    ]
    assert assign_fold('doc-a') in {0, 1}
    folds = split_rows(rows)
    assert {row['document_id'] for row in folds[0]}.isdisjoint(
        {row['document_id'] for row in folds[1]}
    )
    assert sum(len(value) for value in folds.values()) == 3


def test_synthetic_rows_are_capped_at_thirty_percent_of_real_rows():
    from build_reward_data import cap_synthetic_rows

    real = [{'id': f'r{i}', 'source': 'real'} for i in range(10)]
    synthetic = [{'id': f's{i}', 'source': 'synthetic'} for i in range(8)]
    result = cap_synthetic_rows(real, synthetic)
    assert len(result) == 13
    assert sum(row['source'] == 'synthetic' for row in result) == 3


def test_synthetic_negative_changes_type_and_removes_one_argument_deterministically():
    from build_reward_data import synthetic_negative

    event = {
        'type': 'PRP', 'trigger_offsets': [2],
        'args': [
            {'role': 'Content', 'offsets': [4]},
            {'role': 'Target', 'offsets': [5]},
        ],
    }
    result = synthetic_negative(event, 'FAC', {'FAC': {'Subject', 'Object'}})
    assert result['type'] == 'FAC'
    assert result['source'] == 'synthetic'
    assert result['args'] == []
