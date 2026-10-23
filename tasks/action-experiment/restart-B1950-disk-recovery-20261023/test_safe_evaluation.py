"""CPU tests exercise the production generation retry seam without GPU jobs."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location('safe_evaluation', HERE / 'safe_evaluation.py')
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


class Tokenizer:
    padding_side = 'right'
    eos_token_id = 99
    pad_token_id = 0
    bos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def __call__(self, prompts, **kwargs):
        return {'input_ids': torch.tensor([[int(prompt)] for prompt in prompts])}

    def decode(self, tokens, **kwargs):
        return str(tokens[0])


class Model(torch.nn.Module):
    def __init__(self, failures):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.generation_config = SimpleNamespace(eos_token_id=99)
        self.failures = list(failures)
        self.calls = []

    def generate(self, input_ids, **kwargs):
        ids = input_ids[:, 0].tolist()
        self.calls.append((ids, kwargs))
        if self.failures:
            error = self.failures.pop(0)
            if error is not None:
                raise error
        return torch.cat((input_ids, input_ids, torch.full_like(input_ids, 99)), dim=1)


def rows(count=7):
    return [dict(id=str(i), document_id='doc', sentence_index=i, prompt=str(i)) for i in range(count)]


def fake_score(source, predictions):
    assert [p['id'] for p in predictions] == [r['id'] for r in source]
    assert [p['raw_output'] for p in predictions] == [r['id'] for r in source]
    return {'metrics': {'TC_f1': 1.0}, 'parsed_predictions': [{'id': p['id']} for p in predictions]}


def test_retry_same_rows_reduce_and_keep_reduced_batch(tmp_path):
    model = Model([torch.cuda.OutOfMemoryError('two too large')])
    tokenizer = Tokenizer()
    with patch.object(safe._original, 'score_predictions', side_effect=fake_score), patch.object(torch.cuda, 'empty_cache') as clear:
        result = safe.evaluate_model(model, tokenizer, rows(), tmp_path / 'out', max_new_tokens=768, batch_size=4)
    assert [ids for ids, _ in model.calls] == [[0, 1], [0], [1], [2], [3], [4], [5], [6]]
    assert all(call['generation_config'].max_new_tokens == 768 and call['generation_config'].do_sample is False for _, call in model.calls)
    policy = result['adaptive_generation']
    assert policy['requested_batch_size'] == 4
    assert policy['actual_batch_row_counts'] == {'1': 7}
    assert policy['oom_retry_count'] == 1
    assert policy['effective_max_batch_size'] == 2
    assert policy['oom_retries'][0]['retry_batch_size'] == 1
    assert clear.call_count >= 2
    assert model.training and tokenizer.padding_side == 'right'
    assert len((tmp_path / 'out' / 'predictions.jsonl').read_text().splitlines()) == 7


@pytest.mark.parametrize('error', [torch.cuda.OutOfMemoryError('one too large'), RuntimeError('other failure')])
def test_terminal_errors_no_outputs_restore_state(tmp_path, error):
    model = Model([None, error])
    model.eval()
    tokenizer = Tokenizer()
    tokenizer.padding_side = 'left'
    with patch.object(torch.cuda, 'empty_cache'), pytest.raises(type(error), match=str(error)):
        safe.evaluate_model(model, tokenizer, rows(3), tmp_path / 'out', batch_size=1)
    assert not (tmp_path / 'out').exists()
    assert not model.training and tokenizer.padding_side == 'left'
    assert len(model.calls) == 2


def test_non_oom_at_batch_four_never_retries(tmp_path):
    model = Model([RuntimeError('not a CUDA OOM')])
    tokenizer = Tokenizer()
    with patch.object(torch.cuda, 'empty_cache'), pytest.raises(RuntimeError, match='not a CUDA OOM'):
        safe.evaluate_model(model, tokenizer, rows(), tmp_path / 'out', batch_size=4)
    assert len(model.calls) == 1
    assert model.training and tokenizer.padding_side == 'right'
    assert not (tmp_path / 'out').exists()


def test_requested_four_is_capped_at_two_without_oom(tmp_path):
    model = Model([])
    with patch.object(safe._original, 'score_predictions', side_effect=fake_score), patch.object(torch.cuda, 'empty_cache'):
        result = safe.evaluate_model(model, Tokenizer(), rows(), tmp_path / 'out', batch_size=4)
    assert [ids for ids, _ in model.calls] == [[0, 1], [2, 3], [4, 5], [6]]
    policy = result['adaptive_generation']
    assert policy['requested_batch_size'] == 4
    assert policy['effective_max_batch_size'] == 2
    assert policy['final_batch_size'] == 2
    assert policy['oom_retry_count'] == 0
    assert policy['actual_batch_row_counts'] == {'2': 6, '1': 1}


def test_failed_generation_tensor_frame_released_before_cache_clear(tmp_path):
    import weakref

    class FrameModel(Model):
        reference = None

        def generate(self, input_ids, **kwargs):
            if self.reference is None:
                scratch = torch.ones(3)
                self.reference = weakref.ref(scratch)
                error = torch.cuda.OutOfMemoryError('injected frame-held scratch')
                raise error
            return super().generate(input_ids, **kwargs)

    model = FrameModel([])

    def check_released():
        if model.reference is not None:
            assert model.reference() is None, 'OOM traceback retains failed-generation tensors'

    with patch.object(safe._original, 'score_predictions', side_effect=fake_score), patch.object(torch.cuda, 'empty_cache', side_effect=check_released):
        result = safe.evaluate_model(model, Tokenizer(), rows(), tmp_path / 'out', batch_size=4)
    assert result['adaptive_generation']['oom_retry_count'] == 1


def test_transfer_oom_is_not_retried(tmp_path):
    class TransferFailure:
        def to(self, device):
            raise torch.cuda.OutOfMemoryError('transfer OOM')

    class TransferTokenizer(Tokenizer):
        def __call__(self, prompts, **kwargs):
            return {'input_ids': TransferFailure()}

    model = Model([])
    tokenizer = TransferTokenizer()
    with patch.object(torch.cuda, 'empty_cache'), pytest.raises(torch.cuda.OutOfMemoryError, match='transfer OOM'):
        safe.evaluate_model(model, tokenizer, rows(), tmp_path / 'out')
    assert not model.calls
    assert model.training and tokenizer.padding_side == 'right'
    assert not (tmp_path / 'out').exists()


def test_three_rows_retry_two_to_one(tmp_path):
    model = Model([torch.cuda.OutOfMemoryError('two too large')])
    with patch.object(safe._original, 'score_predictions', side_effect=fake_score), patch.object(torch.cuda, 'empty_cache'):
        result = safe.evaluate_model(model, Tokenizer(), rows(3), tmp_path / 'out', batch_size=4)
    assert [ids for ids, _ in model.calls] == [[0, 1], [0], [1], [2]]
    assert result['adaptive_generation']['oom_retries'][0]['retry_batch_size'] == 1
