import json


def test_common_imports_stage2_helpers_without_mutating_stage2_defaults():
    import common

    assert common.parse_forest('(ROOT)') == ([], {
        'invalid': False, 'dropped_events': 0, 'dropped_args': 0,
        'unknown_event_types': 0, 'unknown_roles': 0, 'missing_trigger': 0,
    })
    assert common.align_forest.__module__ == 'align_events'


def test_event_and_trigger_keys_are_canonical_and_hashable():
    import common

    event = {
        'type': 'PRP',
        'trigger_offsets': [8, 7, 8],
        'args': [
            {'role': 'Target', 'offsets': [14, 13]},
            {'role': 'Content', 'offsets': [11]},
            {'role': 'Target', 'offsets': [14, 13]},
        ],
    }
    assert common.trigger_key(event) == (7, 8)
    assert common.event_key(event) == (
        'PRP', (7, 8), frozenset({('Target', (13, 14)), ('Content', (11,))})
    )
    assert common.event_key_string(event) == '["PRP",[7,8],[["Content",[11]],["Target",[13,14]]]]'


def test_atomic_json_write_publishes_complete_document(tmp_path):
    import common

    output = tmp_path / 'nested' / 'state.json'
    common.atomic_write_json(output, {'status': 'running', 'items': [1, 2]})

    assert json.loads(output.read_text()) == {'status': 'running', 'items': [1, 2]}
    assert not list(output.parent.glob('*.tmp'))


def test_sha256_file_is_stable(tmp_path):
    import common

    path = tmp_path / 'payload.txt'
    path.write_text('stage3\n', encoding='utf-8')
    assert common.sha256_file(path) == common.sha256_file(path)
    assert len(common.sha256_file(path)) == 64
