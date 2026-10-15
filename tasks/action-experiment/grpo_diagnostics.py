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
                ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                ratio_outside_clip_fraction=float(((ratio < 1-clip) | (ratio > 1+clip)).float().mean()))


def assert_probability_consistency(stats, *, max_p99=.15, max_difference=1.,
                                   max_outside_clip_fraction=.01):
    """Single responses only hard-fail for nonfinite or severe differences.

    Legacy percentile/fraction arguments remain accepted for callers; isolated
    clip violations are warnings handled by ProbabilityGuard across responses.
    The actual GRPO loss clip is never changed by this diagnostic.
    """
    import math
    if any(not math.isfinite(value) for value in stats.values()):
        raise RuntimeError('GRPO probability consistency failed: nonfinite statistics')
    if stats['abs_difference_max'] > max_difference:
        raise RuntimeError('GRPO probability consistency failed: ' + str(stats))


class ProbabilityGuard:
    """Token-weighted, nonoverlapping windows for same-parameter diagnostics.

    Hard failures stop immediately. Moderate BF16/cache discrepancies require
    consecutive bad windows, each covering enough responses *and* valid tokens.
    Independent guard ratio bounds do not modify the loss clipping interval.
    """

    def __init__(self, window_responses=16, min_valid_tokens=512, persistence=3,
                 hard_max_difference=1., warning_p99=.15, max_clip_fraction=.01,
                 guard_ratio_bounds=(.5, 2.), max_guard_ratio_fraction=.01):
        import math
        parameters = [hard_max_difference, warning_p99, max_clip_fraction,
                      max_guard_ratio_fraction, *guard_ratio_bounds]
        if (any(not math.isfinite(value) for value in parameters)
                or any(not isinstance(value, int) or value < 1
                       for value in (window_responses, min_valid_tokens, persistence))
                or hard_max_difference <= 0 or warning_p99 < 0
                or not 0 <= max_clip_fraction <= 1
                or not 0 <= max_guard_ratio_fraction <= 1
                or not 0 < guard_ratio_bounds[0] < 1 < guard_ratio_bounds[1]):
            raise ValueError('Invalid probability guard configuration')
        self.window_responses = window_responses
        self.min_valid_tokens = min_valid_tokens
        self.persistence = persistence
        self.hard_max_difference = hard_max_difference
        self.warning_p99 = warning_p99
        self.max_clip_fraction = max_clip_fraction
        self.guard_ratio_bounds = tuple(guard_ratio_bounds)
        self.max_guard_ratio_fraction = max_guard_ratio_fraction
        self.response_count = 0
        self.bad_windows = 0
        self.windows_checked = 0
        self._deltas = []

    def configuration(self):
        return dict(window_responses=self.window_responses,
                    min_valid_tokens=self.min_valid_tokens,
                    persistence=self.persistence,
                    hard_max_difference=self.hard_max_difference,
                    warning_p99=self.warning_p99,
                    max_clip_fraction=self.max_clip_fraction,
                    guard_ratio_bounds=list(self.guard_ratio_bounds),
                    max_guard_ratio_fraction=self.max_guard_ratio_fraction,
                    loss_clip=.2)

    def _statistics(self, delta):
        import torch
        stats = probability_statistics(delta, torch.zeros_like(delta),
                                       torch.ones_like(delta, dtype=torch.bool))
        ratio = delta.exp()
        low, high = self.guard_ratio_bounds
        stats['guard_ratio_outside_fraction'] = int(((ratio < low) | (ratio > high)).sum()) / delta.numel()
        # Fractions use integer counts to avoid float32 boundary roundoff.
        stats['ratio_outside_clip_fraction'] = int(((ratio < .8) | (ratio > 1.2)).sum()) / delta.numel()
        return stats

    def _warnings(self, stats):
        return [key for key, limit in (
            ('abs_difference_p99', self.warning_p99),
            ('ratio_outside_clip_fraction', self.max_clip_fraction),
            ('guard_ratio_outside_fraction', self.max_guard_ratio_fraction))
            if stats[key] > limit]

    def observe(self, current, old, mask):
        import torch
        if current.shape != old.shape or current.shape != mask.shape:
            raise ValueError('Probability tensors and mask must have equal shapes')
        selected = mask.bool()
        a = current.detach()[selected].float().cpu()
        b = old.detach()[selected].float().cpu()
        if not a.numel():
            raise ValueError('Probability diagnostic needs valid completion tokens')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise RuntimeError('GRPO probability consistency failed: nonfinite valid log probabilities; '
                               + str(dict(valid_tokens=a.numel(),
                                          nonfinite_current=int((~torch.isfinite(a)).sum()),
                                          nonfinite_old=int((~torch.isfinite(b)).sum()))))
        delta = a - b
        if not torch.isfinite(delta).all() or delta.abs().max() > self.hard_max_difference:
            raise RuntimeError('GRPO probability consistency failed: severe token difference; '
                               + str(dict(valid_tokens=delta.numel(),
                                          abs_difference_max=float(delta.abs().max()))))
        sample = self._statistics(delta)
        self.response_count += 1
        self._deltas.append(delta)
        combined = torch.cat(self._deltas)
        window = self._statistics(combined)
        window_response_count = len(self._deltas)
        checked = (window_response_count >= self.window_responses
                   and combined.numel() >= self.min_valid_tokens)
        warnings = self._warnings(window)
        if checked:
            self.windows_checked += 1
            self.bad_windows = self.bad_windows + 1 if warnings else 0
            self._deltas.clear()
        result = dict(sample_statistics=sample,
                      sample_warning=bool(self._warnings(sample)),
                      sample_warning_reasons=self._warnings(sample),
                      window_statistics=window,
                      window_warning=bool(warnings),
                      window_warning_reasons=warnings,
                      checked_window=checked, bad_windows=self.bad_windows,
                      windows_checked=self.windows_checked,
                      response_count=self.response_count,
                      window_response_count=window_response_count)
        if checked and self.bad_windows >= self.persistence:
            raise RuntimeError('GRPO probability consistency failed: persistent bad windows; '
                               + str(result))
        return result
