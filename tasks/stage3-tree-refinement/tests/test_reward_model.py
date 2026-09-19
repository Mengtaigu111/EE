from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
import pytest


def test_reward_input_has_exact_three_sections_and_this_marker():
    from score_candidates import build_reward_input

    text = build_reward_input('abstract text', '(PRP [THIS])', '(PRP (Trigger "propose"))')
    assert text == (
        'Abstract:\nabstract text\n\n'
        'Forest context:\n(PRP [THIS])\n\n'
        'Candidate event:\n(PRP (Trigger "propose"))'
    )


def test_roberta_reward_input_drops_forest_context_but_keeps_candidate():
    from score_candidates import format_reward_text

    text = (
        'Abstract:\nabstract text\n\n'
        'Forest context:\n(PRP [THIS])\n\n'
        'Candidate event:\n(PRP (Trigger "propose"))'
    )

    assert format_reward_text(text, backend='roberta') == (
        'Abstract:\nabstract text\n\n'
        'Candidate event:\n(PRP (Trigger "propose"))'
    )
    assert format_reward_text(text, backend='qwen') == text


def test_candidate_and_forest_context_are_canonical_sexpressions():
    from score_candidates import build_forest_context, candidate_to_sexpr

    candidate = {'type': 'PRP', 'trigger_text': 'propose', 'args': [
        {'role': 'Content', 'text': 'method'}
    ]}
    context = build_forest_context([candidate, {'type': 'FAC', 'trigger_text': 'improve', 'args': []}], candidate)
    assert context == '(PRP [THIS]) (FAC (Trigger "improve"))'
    assert candidate_to_sexpr(candidate) == '(PRP (Trigger "propose") (Content "method"))'


def test_reward_row_has_stable_training_output_schema():
    from score_candidates import build_reward_row

    row = build_reward_row(
        {'id': 'doc-1', 'abstract': 'abstract text'},
        [{'type': 'PRP', 'trigger_text': 'propose', 'args': []}],
        {'id': 'candidate-1', 'type': 'PRP', 'trigger_text': 'propose', 'args': [],
         'label': {'exists': 1, 'type_ok': 1, 'event_score': 0.75}, 'source': 'real'},
    )

    assert set(row) == {
        'id', 'document_id', 'text', 'exists', 'type_ok', 'event_score', 'source',
    }
    assert row['id'] == 'candidate-1'
    assert row['exists'] == 1
    assert row['type_ok'] == 1
    assert row['event_score'] == pytest.approx(0.75)


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


def test_multitask_loss_masks_event_head_by_exists_and_type():
    from train_reward import multitask_loss

    outputs = {
        'exists_logits': torch.tensor([0.0, 0.0]),
        'type_logits': torch.tensor([0.0, 0.0]),
        'event_score': torch.tensor([0.0, 1.0]),
    }
    labels = {
        'exists': torch.tensor([0.0, 1.0]),
        # The first row is deliberately inconsistent to verify the parent mask.
        'type_ok': torch.tensor([1.0, 1.0]),
        'event_score': torch.tensor([0.0, 1.0]),
    }

    result = multitask_loss(outputs, labels)

    assert result['event_loss'].item() == pytest.approx(0.0)


class _Backbone(nn.Module):
    def __init__(self, model_type='qwen2'):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=1, model_type=model_type)

    def forward(self, input_ids, attention_mask, return_dict=True):
        del return_dict
        # The token id itself is the hidden state, which makes the selected
        # pooling position observable without a real transformer.
        return SimpleNamespace(last_hidden_state=input_ids.to(torch.float32).unsqueeze(-1))


class _BfloatBackbone(_Backbone):
    def __init__(self):
        super().__init__('qwen2')
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))


def _exists_logits(model, input_ids, attention_mask):
    model.exists_head.weight.data.fill_(1.0)
    model.exists_head.bias.data.zero_()
    return model(input_ids, attention_mask)['exists_logits']


def test_qwen_pooling_uses_last_non_padding_token():
    from train_reward import MultiHeadRewardModel

    model = MultiHeadRewardModel(_Backbone('qwen2'), pooling='last')
    input_ids = torch.tensor([[0, 4, 9], [7, 8, 0]])
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 0]])

    logits = _exists_logits(model, input_ids, attention_mask)

    assert logits.tolist() == [9.0, 8.0]


def test_roberta_pooling_uses_first_non_padding_token_as_cls():
    from train_reward import MultiHeadRewardModel

    model = MultiHeadRewardModel(_Backbone('roberta'), pooling='cls')
    input_ids = torch.tensor([[0, 4, 9], [7, 8, 0]])
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 0]])

    logits = _exists_logits(model, input_ids, attention_mask)

    assert logits.tolist() == [4.0, 7.0]


def test_auto_pooling_selects_roberta_cls_and_qwen_last_token():
    from train_reward import MultiHeadRewardModel

    roberta = MultiHeadRewardModel(_Backbone('roberta'))
    qwen = MultiHeadRewardModel(_Backbone('qwen2'))

    assert roberta.pooling == 'cls'
    assert qwen.pooling == 'last'


def test_reward_heads_follow_backbone_dtype():
    from train_reward import MultiHeadRewardModel

    model = MultiHeadRewardModel(_BfloatBackbone(), pooling='last')

    assert model.exists_head.weight.dtype == torch.bfloat16
    assert model.type_head.weight.dtype == torch.bfloat16
    assert model.event_head.weight.dtype == torch.bfloat16


def test_train_rows_accumulates_microbatches_and_records_memory_metrics():
    from train_reward import MultiHeadRewardModel, train_rows

    model = MultiHeadRewardModel(_Backbone(), pooling='last')
    rows = [{
        'input_ids': [index + 1],
        'labels': {'exists': 1.0, 'type_ok': 1.0, 'event_score': 0.5},
    } for index in range(5)]
    tokenizer = SimpleNamespace(pad_token_id=0)

    metrics = train_rows(
        rows, model, tokenizer, 'cpu', epochs=1, learning_rate=1e-2,
        grad_accumulation=2, seed=42, microbatch_size=1,
    )

    assert metrics['microbatch_size'] == 1
    assert metrics['grad_accumulation'] == 2
    assert metrics['effective_batch_size'] == 2
    assert metrics['microbatches'] == 5
    assert metrics['updates'] == 3
    assert metrics['peak_memory_bytes'] == 0


def test_reward_score_formula_combines_probabilities_and_event_score():
    from score_candidates import reward_score

    score = reward_score(0.8, 0.5, 0.6)
    assert score == 0.8 * 0.5 * (0.5 + 0.5 * 0.6)


def test_length_audit_rejects_qwen_overlength_without_truncation():
    from score_candidates import LengthAuditError, audit_lengths

    class Tokenizer:
        def __call__(self, texts, add_special_tokens=False, truncation=False):
            return {'input_ids': [[1] * len(text.split()) for text in texts]}

    with pytest.raises(LengthAuditError, match='exceed') as error:
        audit_lengths(Tokenizer(), ['a b c'], max_len=2, strict=True)

    assert error.value.audit['overlength_count'] == 1
    assert error.value.audit['overlength'][0] == {'index': 0, 'tokens': 3}


def test_length_audit_reports_overlength_without_truncating_in_non_strict_mode():
    from score_candidates import audit_lengths

    calls = []

    class Tokenizer:
        def __call__(self, texts, add_special_tokens=False, truncation=False):
            calls.append({'add_special_tokens': add_special_tokens, 'truncation': truncation})
            return {'input_ids': [[1] * len(text.split()) for text in texts]}

    audit = audit_lengths(Tokenizer(), ['a b c'], max_len=2, strict=False)

    assert audit['overlength_count'] == 1
    assert audit['truncation'] is False
    assert calls == [{'add_special_tokens': True, 'truncation': False}]


def test_tokenize_reward_rows_applies_backend_input_contract_before_audit():
    from train_reward import tokenize_reward_rows

    seen = []

    class Tokenizer:
        def __call__(self, texts, add_special_tokens=False, truncation=False):
            seen.append(list(texts))
            return {'input_ids': [[1] * len(text.split()) for text in texts]}

    text = (
        'Abstract:\nabstract\n\n'
        'Forest context:\n(PRP [THIS])\n\n'
        'Candidate event:\n(PRP (Trigger "propose"))'
    )
    token_rows, audit = tokenize_reward_rows(
        [{'text': text, 'exists': 1, 'type_ok': 1, 'event_score': 1}],
        Tokenizer(), max_len=20, backend='roberta',
    )

    assert 'Forest context:' not in seen[0][0]
    assert len(token_rows) == 1
    assert audit['count'] == 1


def test_reward_validation_metrics_are_available_without_extra_packages():
    from score_candidates import binary_auc, spearman_correlation

    assert binary_auc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0
    assert spearman_correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_binary_auc_uses_average_ranks_for_tied_scores():
    from score_candidates import binary_auc

    assert binary_auc([0, 1, 0, 1], [0.5, 0.5, 0.8, 0.9]) == pytest.approx(0.625)


def test_score_model_outputs_returns_one_reward_per_complete_output_row():
    from score_candidates import score_model_outputs

    scores = score_model_outputs({
        'exists_logits': torch.tensor([0.0, 2.0]),
        'type_logits': torch.tensor([0.0, 2.0]),
        'event_score': torch.tensor([0.0, 1.0]),
    })

    assert len(scores) == 2
    assert scores[0] == pytest.approx(0.125)
    assert scores[1] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item() ** 2)


def test_score_model_outputs_rejects_misaligned_heads():
    from score_candidates import score_model_outputs

    with pytest.raises(ValueError, match='same length'):
        score_model_outputs({
            'exists_logits': torch.tensor([0.0, 1.0]),
            'type_logits': torch.tensor([0.0]),
            'event_score': torch.tensor([0.0, 1.0]),
        })


@pytest.mark.parametrize('metric', ['binary_auc', 'spearman_correlation'])
def test_metrics_reject_mismatched_lengths(metric):
    from score_candidates import binary_auc, spearman_correlation

    function = {'binary_auc': binary_auc, 'spearman_correlation': spearman_correlation}[metric]
    with pytest.raises(ValueError, match='same length'):
        function([0, 1], [0.5])


def test_collate_rejects_empty_rows_and_missing_pad_id():
    from train_reward import collate_reward_rows

    with pytest.raises(ValueError, match='at least one'):
        collate_reward_rows([], pad_token_id=0)
    with pytest.raises(ValueError, match='pad_token_id'):
        collate_reward_rows([{'input_ids': [1], 'labels': {
            'exists': 0.0, 'type_ok': 0.0, 'event_score': 0.0,
        }}], pad_token_id=None)


def test_train_reward_script_exposes_cli_help_without_starting_gpu():
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'train_reward.py'

    result = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0
    assert '--backend' in result.stdout
    assert '--max-len' in result.stdout
