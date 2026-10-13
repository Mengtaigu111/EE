"""Same-parameter GRPO probability checks; diagnostic thresholds never alter loss."""


def probability_statistics(current, old, mask, clip=.2):
    """Summarize original, valid completion tokens, including their first EOS.

    The asymmetric log thresholds implied by exp(delta) matter: testing
    abs(delta) > clip is not equivalent to testing the actual policy ratio.
    Invalid padding is excluded before finite checks, subtraction and exp.
    """
    import torch
    if current.shape != old.shape or current.shape != mask.shape:
        raise ValueError('Probability tensors and mask must have equal shapes')
    if not 0 < clip < 1:
        raise ValueError('Clip must be between zero and one')
    selected = mask.bool()
    a = current.detach()[selected].float()
    b = old.detach()[selected].float()
    if not a.numel():
        raise ValueError('Probability diagnostic needs valid completion tokens')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Valid token log probabilities must be finite')
    difference = (a - b).abs()
    ratio = (a - b).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('Valid token policy ratios must be finite')
    return dict(valid_tokens=a.numel(), abs_difference_mean=float(difference.mean()),
                abs_difference_p95=float(torch.quantile(difference, .95)),
                abs_difference_p99=float(torch.quantile(difference, .99)),
                abs_difference_max=float(difference.max()), ratio_mean=float(ratio.mean()),
                ratio_outside_clip_fraction=float(((ratio < 1-clip) | (ratio > 1+clip)).float().mean()))


def assert_probability_consistency(stats, *, max_p99=.15, max_difference=.5,
                                   max_outside_clip_fraction=.01):
    """Fail closed before updates when cached/teacher paths should agree.

    BF16 cache/full-sequence kernels need not agree bitwise. These explicit
    bounds allow limited roundoff, while at least 99% of ratios must still
    lie inside the actual GRPO clip interval. Smoke reports observed values
    before this guard. Threshold changes must be recorded, not inferred from
    finite loss or silently increased after a failed check.
    """
    checks = [('abs_difference_p99', max_p99),
              ('abs_difference_max', max_difference),
              ('ratio_outside_clip_fraction', max_outside_clip_fraction)]
    violations = [f'{key}={stats[key]:.8g} > {limit:.8g}' for key, limit in checks if stats[key] > limit]
    if violations:
        raise RuntimeError('GRPO probability consistency failed: ' + '; '.join(violations))
