class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [ord(char) for char in text]}


def test_prefix_token_check_accepts_exact_string_prefix():
    from type_competition import prefix_token_check

    result = prefix_token_check([1, 2, 3, 4], [1, 2, 3, 4, 5], prefix_length=4)
    assert result == {'mismatches': 0, 'rate': 0.0, 'use_token_slice': False}


def test_prefix_token_check_switches_to_greedy_slice_above_one_percent():
    from type_competition import prefix_token_check

    result = prefix_token_check([1] * 100, [2] + [1] * 99, prefix_length=100)
    assert result['mismatches'] == 1
    assert result['rate'] == 0.01
    assert result['use_token_slice'] is False

    result = prefix_token_check([1] * 100, [2, 2] + [1] * 98, prefix_length=100)
    assert result['use_token_slice'] is True


def test_forced_prefix_uses_prior_events_and_fixed_trigger():
    from type_competition import build_forced_prefix

    result = build_forced_prefix(
        'PROMPT',
        ['(PRP (Trigger "propose"))', '(PUR (Trigger "induce"))'],
        'CMP',
        'improve',
    )
    assert result == 'PROMPT (ROOT (PRP (Trigger "propose")) (PUR (Trigger "induce")) (CMP (Trigger "improve")'
