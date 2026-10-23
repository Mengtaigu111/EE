"""Select policy-loss targets from the shared R0-v2 occurrence trace.

Retokenization is used only to obtain character offsets. The caller retains the
sampled IDs, original completion mask, and independently computed KL term.
"""

import math

import torch

FIELD_NAMES = ('type', 'trigger', 'role', 'argument')


def build_field_mask(tokenizer, completion_ids, valid_mask, text, trace, advantage):
    """Return a detached binary mask with the shape/device of sampled IDs.

    One candidate is accepted as [L] or [1,L]. Expected localization failures
    return all ones with a diagnostic reason. Shape and nonfinite-advantage
    errors, and unexpected tokenizer/program errors, propagate to the caller.
    """
    if completion_ids.shape != valid_mask.shape:
        raise ValueError('completion_ids and valid_mask shape must agree')
    if completion_ids.ndim not in (1, 2) or (completion_ids.ndim == 2 and completion_ids.shape[0] != 1):
        raise ValueError('one candidate shape must be [L] or [1,L]')
    value = float(advantage)
    if not math.isfinite(value):
        raise ValueError('advantage must be finite')
    sign = (value > 0) - (value < 0)
    result = torch.ones_like(completion_ids, dtype=torch.float32)
    diagnostics = {
        'fallback_reason': None,
        'advantage_sign': sign,
        'data_token_count': 0,
        'kept_data_token_count': 0,
        'mixed_boundary_token_count': 0,
        'field_coverage': {name: {'field_count': 0, 'data_token_count': 0,
                                  'kept_data_token_count': 0} for name in FIELD_NAMES},
    }

    def fallback(reason):
        diagnostics['fallback_reason'] = reason
        return result, diagnostics

    if not isinstance(trace, dict) or not trace.get('syntax_valid', False):
        return fallback('syntax_invalid')
    fields = trace.get('fields')
    if not isinstance(fields, list) or (not fields and text.strip() != 'STOP'):
        return fallback('trace_unavailable')
    previous_end = 0
    for field in fields:
        if not isinstance(field, dict):
            return fallback('invalid_field_span')
        start, end = field.get('start'), field.get('end')
        if (field.get('field') not in FIELD_NAMES or not isinstance(field.get('hit'), bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or not 0 <= start < end <= len(text) or start < previous_end
                or not isinstance(field.get('event_index'), int)
                or (field['field'] in ('role', 'argument')
                    and not isinstance(field.get('argument_index'), int))):
            return fallback('invalid_field_span')
        previous_end = end
    flat_ids = completion_ids.reshape(-1)
    active_positions = valid_mask.reshape(-1).bool().nonzero(as_tuple=False).flatten().tolist()
    active_ids = [int(flat_ids[i]) for i in active_positions]
    if tokenizer.decode(active_ids, skip_special_tokens=True) != text:
        return fallback('decoded_text_mismatch')
    if not fields:
        # A valid empty answer has no data positions to align. Sampled STOP
        # need not use canonical BPE merges to remain a normal all-one mask.
        return result, diagnostics
    special_ids = set(tokenizer.all_special_ids)
    ordinary = [(position, token_id) for position, token_id in zip(active_positions, active_ids)
                if token_id not in special_ids]
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError:
        # Slow tokenizers cannot provide verified offsets; no alternate ID path.
        return fallback('offset_mapping_unavailable')
    offsets = encoded.get('offset_mapping')
    encoded_ids = encoded.get('input_ids')
    if offsets is None or encoded_ids is None or len(offsets) != len(encoded_ids):
        return fallback('offset_mapping_unavailable')
    encoded_ordinary = [(token_id, offset) for token_id, offset in zip(encoded_ids, offsets)
                        if token_id not in special_ids]
    if [token_id for _, token_id in ordinary] != [token_id for token_id, _ in encoded_ordinary]:
        return fallback('token_ids_mismatch')
    # Validate the whole alignment before changing any target mask.
    for _, offset in encoded_ordinary:
        if (not isinstance(offset, (tuple, list)) or len(offset) != 2
                or not all(isinstance(index, int) for index in offset)
                or not 0 <= offset[0] <= offset[1] <= len(text)):
            return fallback('invalid_token_offsets')
    for field in fields:
        diagnostics['field_coverage'][field['field']]['field_count'] += 1
    flat_result = result.reshape(-1)
    for (position, _), (_, (start, end)) in zip(ordinary, encoded_ordinary):
        if start == end:
            continue  # No reliable character extent: preserve the original signal.
        overlapping = [field for field in fields if start < field['end'] and field['start'] < end]
        containing = [field for field in overlapping if field['start'] <= start and end <= field['end']]
        if len(containing) == 1 and len(overlapping) == 1:
            field = containing[0]
            keep = int(field['hit']) if sign > 0 else int(not field['hit']) if sign < 0 else 0
            flat_result[position] = keep
            diagnostics['data_token_count'] += 1
            diagnostics['kept_data_token_count'] += keep
            coverage = diagnostics['field_coverage'][field['field']]
            coverage['data_token_count'] += 1
            coverage['kept_data_token_count'] += keep
        elif overlapping:
            diagnostics['mixed_boundary_token_count'] += 1
    return result, diagnostics
