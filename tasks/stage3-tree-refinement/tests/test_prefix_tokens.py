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
        ['(PRP (Trigger "propose"))', '(PUR (Trigger "induce"))'],
        'CMP',
        'improve',
    )
    assert result == '(ROOT (PRP (Trigger "propose")) (PUR (Trigger "induce")) (CMP (Trigger "improve")'


def test_forced_prefix_ids_sit_on_the_assistant_side_of_the_chat_template():
    """The forced tree must follow the generation prompt, not sit inside the user turn."""
    import os
    import pytest
    from transformers import AutoTokenizer
    from type_competition import forced_prefix_ids, forced_suffix

    model_path = '/opt/models/Qwen3-8B'
    if not os.path.isdir(model_path):
        pytest.skip('Qwen3-8B tokenizer not available')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prior_text = '(ROOT (PRP (Trigger "propose") (Proposer "we"))'
    ids, start, forced = forced_prefix_ids('Abstract text.\n\nEvent forest:', prior_text, 'CMP', 'improve', tokenizer)
    decoded = tokenizer.decode(ids)
    assert decoded.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n' + prior_text + forced_suffix('CMP', 'improve'))
    assert '(ROOT' not in decoded.split('<|im_end|>')[0]
    assert tokenizer.decode(ids[start:]) == forced_suffix('CMP', 'improve')
    assert forced == len(ids) - start


def test_greedy_prior_text_slices_raw_output_at_the_right_occurrence():
    from type_competition import greedy_prior_text

    raw = '(ROOT (PRP (Trigger "use") (Content "a")) (WKS (Trigger "use") (Content "b")))'
    events = [{'type': 'PRP', 'trigger': 'use', 'args': []}, {'type': 'WKS', 'trigger': 'use', 'args': []}]
    prior, from_raw = greedy_prior_text(raw, events, 1)
    assert from_raw is True
    assert prior == '(ROOT (PRP (Trigger "use") (Content "a"))'
    prior, from_raw = greedy_prior_text(raw, events, 0)
    assert prior == '(ROOT'
    prior, from_raw = greedy_prior_text('(ROOT (PRP (Trigger "made up")))', events, 1)
    assert from_raw is False
