import torch
import pytest


def test_reward_input_has_exact_three_sections_and_this_marker():
    from score_candidates import build_reward_input

    text = build_reward_input('abstract text', '(PRP [THIS])', '(PRP (Trigger "propose"))')
    assert text == (
        'Abstract:\nabstract text\n\n'
        'Forest context:\n(PRP [THIS])\n\n'
        'Candidate event:\n(PRP (Trigger "propose"))'
    )


def test_candidate_and_forest_context_are_canonical_sexpressions():
    from score_candidates import build_forest_context, candidate_to_sexpr

    candidate = {'type': 'PRP', 'trigger_text': 'propose', 'args': [
        {'role': 'Content', 'text': 'method'}
    ]}
    context = build_forest_context([candidate, {'type': 'FAC', 'trigger_text': 'improve', 'args': []}], candidate)
    assert context == '(PRP [THIS]) (FAC (Trigger "improve"))'
    assert candidate_to_sexpr(candidate) == '(PRP (Trigger "propose") (Content "method"))'


def test_multitask_loss_masks_type_and_event_heads_by_parent_labels():
    from train_reward import multitask_loss

    outputs = {
        'exists_logits': torch.tensor([0.0, 0.0]),
        'type_logits': torch.tensor([0.0, 10.0]),
        'event_score': torch.tensor([0.0, 0.0]),
    }
    labels = {
        'exists': torch.tensor([0.0, 1.0]),
        'type_ok': torch.tensor([0.0, 1.0]),
        'event_score': torch.tensor([0.0, 1.0]),
    }
    result = multitask_loss(outputs, labels)
    assert set(result) == {'loss', 'exists_loss', 'type_loss', 'event_loss'}
    assert result['type_loss'].item() < 0.01
    assert result['event_loss'].item() > 0.9


def test_reward_score_formula_combines_probabilities_and_event_score():
    from score_candidates import reward_score

    score = reward_score(0.8, 0.5, 0.6)
    assert score == 0.8 * 0.5 * (0.5 + 0.5 * 0.6)


def test_length_audit_rejects_qwen_overlength_without_truncation():
    from score_candidates import audit_lengths

    class Tokenizer:
        def __call__(self, texts, add_special_tokens=False, truncation=False):
            return {'input_ids': [[1] * len(text.split()) for text in texts]}

    try:
        audit_lengths(Tokenizer(), ['a b c'], max_len=2, strict=True)
    except ValueError as exc:
        assert 'exceed' in str(exc)
    else:
        raise AssertionError('expected strict length audit failure')


def test_reward_validation_metrics_are_available_without_extra_packages():
    from score_candidates import binary_auc, spearman_correlation

    assert binary_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert spearman_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
