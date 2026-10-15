import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_diagnostics import probability_statistics, assert_probability_consistency, ProbabilityGuard


def test_masked_statistics_and_actual_ratio():
    old = torch.zeros(1, 5)
    current = torch.tensor([[0., .1, -.3, .3, float('nan')]])
    mask = torch.tensor([[True, True, True, True, False]])
    result = probability_statistics(current, old, mask)
    assert result['valid_tokens'] == 4
    assert result['abs_difference_mean'] == pytest.approx(.175)
    assert result['abs_difference_max'] == pytest.approx(.3)
    assert result['ratio_outside_clip_fraction'] == .5
    assert result['ratio_mean'] == pytest.approx(float(current[0, :4].exp().mean()))


def test_identical_distribution_passes():
    values = torch.randn(2, 8)
    result = probability_statistics(values, values, torch.ones_like(values, dtype=torch.bool))
    assert_probability_consistency(result)
    assert result['ratio_mean'] == 1.


def test_large_distribution_change_is_rejected():
    values = torch.zeros(1, 100)
    result = probability_statistics(values + 1.01, values, torch.ones_like(values, dtype=torch.bool))
    with pytest.raises(RuntimeError, match='probability consistency'):
        assert_probability_consistency(result)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_valid_token_rejected(value):
    with pytest.raises(ValueError, match='finite'):
        probability_statistics(torch.tensor([[value]]), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))


def test_empty_mask_and_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        probability_statistics(torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.bool))
    with pytest.raises(ValueError):
        probability_statistics(torch.zeros(1, 2), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))


def test_single_short_outlier_warns_without_stopping():
    current = torch.zeros(1, 57)
    current[0, -1] = .2395
    old = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)
    stats = probability_statistics(current, old, mask)
    assert_probability_consistency(stats)
    guard = ProbabilityGuard()
    result = guard.observe(current, old, mask)
    assert result['sample_warning']
    assert not result['checked_window']
    assert result['window_statistics']['valid_tokens'] == 57
    assert result['sample_statistics']['ratio_max'] == pytest.approx(1.2706, abs=.0001)


def test_window_needs_both_response_and_token_counts():
    guard = ProbabilityGuard(window_responses=2, min_valid_tokens=5)
    x = torch.zeros(1, 2)
    for count in (1, 2):
        result = guard.observe(x, x, torch.ones_like(x))
        assert result['response_count'] == count
        assert not result['checked_window']
    result = guard.observe(x, x, torch.ones_like(x))
    assert result['checked_window']
    assert result['window_statistics']['valid_tokens'] == 6
    assert result['window_response_count'] == 3
    result = guard.observe(x, x, torch.ones_like(x))
    assert not result['checked_window']
    assert result['window_statistics']['valid_tokens'] == 2


def test_window_statistics_weight_tokens_not_responses():
    guard = ProbabilityGuard(window_responses=2, min_valid_tokens=100)
    guard.observe(torch.tensor([[.3]]), torch.zeros(1, 1), torch.ones(1, 1))
    result = guard.observe(torch.zeros(1, 99), torch.zeros(1, 99), torch.ones(1, 99))
    assert result['checked_window']
    assert result['window_statistics']['ratio_outside_clip_fraction'] == pytest.approx(.01)
    assert result['bad_windows'] == 0


def test_three_bad_windows_stop_and_good_window_resets():
    guard = ProbabilityGuard(window_responses=1, min_valid_tokens=1)
    bad = torch.tensor([[.3]])
    good = torch.zeros_like(bad)
    assert guard.observe(bad, good, torch.ones_like(bad))['bad_windows'] == 1
    assert guard.observe(bad, good, torch.ones_like(bad))['bad_windows'] == 2
    assert guard.observe(good, good, torch.ones_like(bad))['bad_windows'] == 0
    guard.observe(bad, good, torch.ones_like(bad))
    guard.observe(bad, good, torch.ones_like(bad))
    with pytest.raises(RuntimeError, match='persistent') as failure:
        guard.observe(bad, good, torch.ones_like(bad))
    assert 'ratio_outside_clip_fraction' in str(failure.value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 1.01, -1.01])
def test_guard_hard_failures_stop_immediately(value):
    guard = ProbabilityGuard()
    with pytest.raises(RuntimeError, match='probability'):
        guard.observe(torch.tensor([[value]]), torch.zeros(1, 1), torch.ones(1, 1))


def test_guard_preserves_loss_clip_and_has_separate_ratio_bounds():
    guard = ProbabilityGuard()
    result = guard.observe(torch.tensor([[.3]]), torch.zeros(1, 1), torch.ones(1, 1))
    assert result['sample_statistics']['ratio_outside_clip_fraction'] == 1.
    assert result['sample_statistics']['guard_ratio_outside_fraction'] == 0.
    assert guard.configuration()['guard_ratio_bounds'] == [.5, 2.]
    assert guard.configuration()['loss_clip'] == .2


def test_masked_nan_ignored_by_guard():
    result = ProbabilityGuard().observe(torch.tensor([[0., float('nan')]]),
                                        torch.zeros(1, 2), torch.tensor([[1, 0]]))
    assert result['sample_statistics']['valid_tokens'] == 1
    assert not result['sample_warning']


def test_guard_result_is_json_serializable():
    import json
    json.dumps(ProbabilityGuard().observe(torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1)))


@pytest.mark.parametrize('kwargs', [{'window_responses': 0}, {'min_valid_tokens': 0},
    {'persistence': 0}, {'guard_ratio_bounds': (2., .5)}, {'max_clip_fraction': -1}])
def test_invalid_guard_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ProbabilityGuard(**kwargs)
