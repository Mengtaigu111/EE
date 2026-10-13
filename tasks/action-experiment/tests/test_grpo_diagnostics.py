import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_diagnostics import probability_statistics, assert_probability_consistency


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
    result = probability_statistics(values + .3, values, torch.ones_like(values, dtype=torch.bool))
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


def test_configurable_thresholds_are_not_silent():
    result = probability_statistics(torch.tensor([[.2]]), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))
    with pytest.raises(RuntimeError):
        assert_probability_consistency(result)
    assert_probability_consistency(result, max_p99=.25, max_difference=.25, max_outside_clip_fraction=1.)
