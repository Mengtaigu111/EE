import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def mask_module():
    assert (ROOT / 'action_update_mask.py').exists(), 'four-field mask implementation missing'
    import action_update_mask
    return action_update_mask


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('/opt/models/Qwen3-8B', local_files_only=True)


def sample(tokenizer, text=None, hits=(False, True, True, True)):
    text = text or 'NEW(WKS)\nTRG("uses")\nARG(BaseComponent, "BERT")\nEND\nSTOP'
    # Test fixtures explicitly locate each occurrence, without sharing production parsing.
    values = [('type', 'WKS'), ('trigger', 'uses'), ('role', 'BaseComponent'), ('argument', 'BERT')]
    fields = []
    for (field, value), hit in zip(values, hits):
        start = text.index(value)
        fields.append(dict(field=field, start=start, end=start+len(value), hit=hit,
                           event_index=0, argument_index=0 if field in ('role', 'argument') else None))
    ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id, tokenizer.pad_token_id]])
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[0, -1] = False
    return text, ids, valid, {'syntax_valid': True, 'fields': fields}


def test_field_selection_uses_advantage_sign_and_target_ids(tokenizer):
    m = mask_module()
    text, ids, valid, trace = sample(tokenizer)
    results = []
    for advantage in (1., -1., 0.):
        result, diag = m.build_field_mask(tokenizer, ids, valid, text, trace, advantage)
        assert result.shape == ids.shape and result.device == ids.device
        assert diag['fallback_reason'] is None
        assert diag['data_token_count'] > 0
        assert result[0, -2:].tolist() == [1., 1.]
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        for i, (start, end) in enumerate(offsets):
            contained = [f for f in trace['fields'] if f['start'] <= start < end <= f['end']]
            if contained:
                hit = contained[0]['hit']
                assert result[0, i].item() == (float(hit) if advantage > 0 else float(not hit) if advantage < 0 else 0.)
            else:
                assert result[0, i].item() == 1.
        results.append(result)
    assert not torch.equal(results[0], results[1])
    assert results[2].sum() < ids.numel()


def test_only_alignment_and_syntax_fallback(tokenizer):
    m = mask_module()
    text, ids, valid, trace = sample(tokenizer)
    for changed, expected in [({'syntax_valid': False}, 'syntax_invalid'),
                              ({'syntax_valid': True}, 'trace_unavailable')]:
        result, diag = m.build_field_mask(tokenizer, ids, valid, text, changed, 1.)
        assert torch.all(result == 1) and diag['fallback_reason'] == expected
    result, diag = m.build_field_mask(tokenizer, ids, valid, text + ' ', trace, 1.)
    assert torch.all(result == 1) and diag['fallback_reason'] == 'decoded_text_mismatch'
    bad_trace = {'syntax_valid': True, 'fields': [dict(trace['fields'][0], end=len(text)+1)]}
    _, diag = m.build_field_mask(tokenizer, ids, valid, text, bad_trace, 1.)
    assert diag['fallback_reason'] == 'invalid_field_span'
    with pytest.raises(ValueError, match='shape'):
        m.build_field_mask(tokenizer, ids, valid[:, :-1], text, trace, 1.)
    for advantage in (float('nan'), float('inf')):
        with pytest.raises(ValueError, match='finite'):
            m.build_field_mask(tokenizer, ids, valid, text, trace, advantage)


def test_empty_stop_is_normal_and_supports_one_dimensional_ids(tokenizer):
    m = mask_module()
    text = 'STOP'
    ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
    result, diag = m.build_field_mask(tokenizer, ids, torch.ones_like(ids), text,
                                    {'syntax_valid': True, 'fields': []}, 0.)
    assert torch.all(result == 1) and diag['fallback_reason'] is None
    assert diag['data_token_count'] == 0


def test_real_tokenizer_occurrences_escaping_unicode_and_mixed_boundaries(tokenizer):
    m = mask_module()
    text = 'NEW(WKS)\nTRG("uses")\nARG(Content, "BERT \\"quoted\\" 模型")\nEND\nNEW(MDS)\nTRG("uses")\nARG(BaseComponent, "BERT")\nEND\nSTOP'
    fields = []
    # Explicit fixture intervals preserve raw escape sequences and repeated occurrences.
    cursor = 0
    for event, values in enumerate([('WKS', 'uses', 'Content', 'BERT \\"quoted\\" 模型'),
                                    ('MDS', 'uses', 'BaseComponent', 'BERT')]):
        for name, value in zip(('type', 'trigger', 'role', 'argument'), values):
            start = text.index(value, cursor)
            fields.append(dict(field=name, start=start, end=start+len(value), hit=bool(event),
                               event_index=event, argument_index=0 if name in ('role', 'argument') else None))
            cursor = start + len(value)
    ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
    trace = dict(syntax_valid=True, fields=fields)
    result, diag = m.build_field_mask(tokenizer, ids, torch.ones_like(ids), text, trace, 1.)
    assert diag['fallback_reason'] is None
    assert 0 < diag['kept_data_token_count'] < diag['data_token_count']
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    for i, (start, end) in enumerate(offsets):
        overlaps = [f for f in fields if start < f['end'] and f['start'] < end]
        whole = [f for f in overlaps if f['start'] <= start < end <= f['end']]
        if overlaps and not whole:
            assert result[i] == 1
    assert diag['mixed_boundary_token_count'] > 0


class BoundaryTokenizer:
    all_special_ids = [99]
    def decode(self, ids, skip_special_tokens=True):
        return 'NEW(X)'
    def __call__(self, text, **kwargs):
        return {'input_ids': [1, 2], 'offset_mapping': [(0, 5), (5, 6)]}


def test_retokenized_id_mismatch_is_explicit_and_mixed_is_preserved():
    m = mask_module()
    tok = BoundaryTokenizer()
    trace = {'syntax_valid': True, 'fields': [dict(field='type', start=4, end=5, hit=False, event_index=0, argument_index=None)]}
    result, diag = m.build_field_mask(tok, torch.tensor([1, 2, 99]), torch.ones(3), 'NEW(X)', trace, 1.)
    assert result.tolist() == [1., 1., 1.]
    assert diag['mixed_boundary_token_count'] == 1 and diag['fallback_reason'] is None
    _, diag = m.build_field_mask(tok, torch.tensor([3, 2, 99]), torch.ones(3), 'NEW(X)', trace, 1.)
    assert diag['fallback_reason'] == 'token_ids_mismatch'


def test_unexpected_tokenizer_errors_propagate_and_known_offset_failure_falls_back():
    m = mask_module()
    field_trace = {'syntax_valid': True, 'fields': [dict(field='type', start=4, end=5, hit=False, event_index=0, argument_index=None)]}
    class Broken(BoundaryTokenizer):
        def __call__(self, text, **kwargs):
            raise RuntimeError('unexpected tokenizer bug')
    with pytest.raises(RuntimeError, match='unexpected tokenizer bug'):
        m.build_field_mask(Broken(), torch.tensor([1, 2]), torch.ones(2), 'NEW(X)',
                           field_trace, 1.)
    class Slow(BoundaryTokenizer):
        def __call__(self, text, **kwargs):
            raise NotImplementedError('offsets not supported')
    result, diag = m.build_field_mask(Slow(), torch.tensor([1, 2]), torch.ones(2), 'NEW(X)',
                                      field_trace, 1.)
    assert torch.all(result == 1)
    assert diag['fallback_reason'] == 'offset_mapping_unavailable'


def test_invalid_positions_and_special_tokens_are_never_field_targets(tokenizer):
    m = mask_module()
    text, ids, valid, trace = sample(tokenizer)
    prefix = torch.tensor([[42, 43]], dtype=ids.dtype)
    ids = torch.cat((prefix, ids), dim=1)
    valid = torch.cat((torch.zeros_like(prefix, dtype=torch.bool), valid), dim=1)
    result, diag = m.build_field_mask(tokenizer, ids, valid, text, trace, 0.)
    assert diag['fallback_reason'] is None
    assert result[0, :2].tolist() == [1., 1.]
    assert result[0, -2:].tolist() == [1., 1.]
    assert diag['kept_data_token_count'] == 0


def test_shared_reward_trace_keeps_semantically_invalid_type_role_fields(tokenizer):
    from action_reward_v2 import reward_action_r0_v2
    gold = {'target_events': [{'event_type': 'MDS', 'trigger': {'tokens': ['uses']},
                               'arguments': [{'argument_type': 'BaseComponent', 'tokens': ['BERT']}]}]}
    text = 'NEW(WKS)\nTRG("uses")\nARG(Content, "BERT")\nEND\nSTOP'
    trace = reward_action_r0_v2(text, gold)
    assert trace['reward'] == .5
    assert [field['hit'] for field in trace['fields']] == [False, True, False, True]
    ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
    result, diag = mask_module().build_field_mask(tokenizer, ids, torch.ones_like(ids), text, trace, 1.)
    assert diag['fallback_reason'] is None
    assert diag['field_coverage']['role']['kept_data_token_count'] == 0
    assert diag['field_coverage']['argument']['kept_data_token_count'] > 0


def test_nonempty_answer_with_missing_field_trace_falls_back(tokenizer):
    text, ids, valid, _ = sample(tokenizer)
    result, diag = mask_module().build_field_mask(tokenizer, ids, valid, text,
                                                 {'syntax_valid': True, 'fields': []}, 1.)
    assert torch.all(result == 1)
    assert diag['fallback_reason'] == 'trace_unavailable'


def test_valid_empty_stop_does_not_require_canonical_tokenization(tokenizer):
    # This is a possible sampled history even though canonical BPE merges STOP.
    ids = torch.tensor([tokenizer.encode(char, add_special_tokens=False)[0] for char in 'STOP'])
    assert tokenizer.decode(ids, skip_special_tokens=True) == 'STOP'
    assert ids.tolist() != tokenizer.encode('STOP', add_special_tokens=False)
    result, diag = mask_module().build_field_mask(tokenizer, ids, torch.ones_like(ids), 'STOP',
                                                 {'syntax_valid': True, 'fields': []}, 1.)
    assert torch.all(result == 1)
    assert diag['fallback_reason'] is None
    assert diag['data_token_count'] == 0
