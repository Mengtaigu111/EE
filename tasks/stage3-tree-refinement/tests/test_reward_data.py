def test_fold_assignment_is_sha256_parity_and_document_disjoint():
    from build_folds import assign_fold, manifest, split_rows

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
    summary = manifest(rows)
    assert set(summary['folds']) == {'0', '1'}
    assert sum(summary['folds'][str(i)]['documents_count'] for i in (0, 1)) == 2
    assert sum(summary['folds'][str(i)]['rows'] for i in (0, 1)) == 3


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


def test_build_reward_rows_keeps_c0_context_and_labels_synthetic_source():
    from build_reward_data import build_reward_rows

    data = [{
        'id': 'doc', 'abstract': 'we propose method',
        'gold_events': [{'event_type': 'PRP', 'trigger': {'text': 'propose', 'offsets': [1], 'tokens': ['propose']},
                         'arguments': [{'argument_type': 'Content', 'text': 'method',
                                        'offsets': [2], 'tokens': ['method']}]}],
    }]
    pool = [{
        'document_id': 'doc',
        'sources': [{'source': 'c0', 'events': [{'type': 'PRP', 'trigger_text': 'propose',
                                                 'args': [{'role': 'Content', 'text': 'method'}]}]}],
        'candidates': [{'type': 'PRP', 'trigger_text': 'propose', 'args': [
            {'role': 'Content', 'text': 'method'}], 'label': {
                'exists': 1, 'type_ok': 1, 'event_score': 1.0}}],
    }]
    pool[0]['candidates'] = [dict(pool[0]['candidates'][0], id=f'candidate-{index}')
                             for index in range(10)]
    rows, manifest = build_reward_rows(pool, data, {'PRP': ['FAC']}, {'FAC': set()})
    assert rows[0]['document_id'] == 'doc'
    assert 'Forest context:' in rows[0]['text']
    assert manifest['synthetic_rows_used'] == 2


def test_synthetic_examples_have_distinct_type_and_argument_deletion_variants():
    from build_reward_data import build_synthetic_negatives

    gold = {
        'event_type': 'PRP', 'trigger': {'offsets': [1], 'text': 'propose'},
        'arguments': [
            {'argument_type': 'Content', 'offsets': [2], 'text': 'method'},
            {'argument_type': 'Target', 'offsets': [3], 'text': 'field'},
        ],
    }
    result = build_synthetic_negatives(
        gold, ['FAC'], {'FAC': {'Content', 'Target'}, 'PRP': {'Content', 'Target'}}, seed=42)
    assert len(result) == 2
    assert all(item['source'] == 'synthetic' for item in result)
    assert result[0]['type'] == 'FAC'
    assert len(result[0]['args']) == 2
    assert result[1]['type'] == 'PRP'
    assert len(result[1]['args']) == 1
