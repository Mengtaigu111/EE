"""Enumerate exact text spans without gold or arbitrary position selection.

All strictly increasing subsequences are eligible, including contiguous spans.
Explicit source positions preserve order and repetition, allowing reverse spans.
Ambiguous or truncated enumeration never returns a guessed `offsets` value.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from itertools import islice


def resolve_span(tokens, text, scope=None, explicit_offsets=None, max_candidates=256):
    """Return status, offsets, candidates, and candidate_limit_reached.

    `scope=(lo, hi)` uses half-open global source token bounds. Enumeration is
    lexicographic by ordered index sequence and retains at most max_candidates;
    an additional candidate must be observed before declaring truncation.
    Invalid bounds, limits or explicit positions raise ValueError. Only exact
    token matches are considered; there is no case folding or normalization.
    """
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or max_candidates < 1:
        raise ValueError('max_candidates must be a positive integer')
    lo, hi = (0, len(tokens)) if scope is None else scope
    if any(not isinstance(i, int) or isinstance(i, bool) for i in (lo, hi)) or not 0 <= lo <= hi <= len(tokens):
        raise ValueError('scope must contain valid half-open global token bounds')
    query = text.split()
    if explicit_offsets is not None:
        positions = list(explicit_offsets)
        if not positions:
            raise ValueError('explicit positions must identify a nonempty mention')
        if any(not isinstance(i, int) or isinstance(i, bool) or not lo <= i < hi for i in positions):
            raise ValueError('explicit positions must lie inside the source scope')
        if [tokens[i] for i in positions] != query:
            raise ValueError('explicit positions must reproduce the exact ordered text tokens')
        return {'status':'explicit', 'offsets':positions, 'candidates':[positions.copy()],
                'candidate_limit_reached':False}
    candidates = []
    if query:
        occurrences = defaultdict(list)
        for position in range(lo, hi): occurrences[tokens[position]].append(position)
        # Latest feasible position for each query token prunes branches with no
        # possible complete suffix; candidate count cannot cause exponential
        # exploration of dead-end prefixes. No recursive call depth is needed.
        latest = []; ceiling = hi
        for token in reversed(query):
            positions = occurrences[token]
            j = bisect_left(positions, ceiling)-1
            if j < 0: break
            ceiling = positions[j]; latest.append(ceiling)
        if len(latest) == len(query):
            latest.reverse()
            def choices(depth, previous):
                positions = occurrences[query[depth]]
                return islice(positions, bisect_right(positions, previous), bisect_right(positions, latest[depth]))
            prefix = []; stack = [choices(0, lo-1)]
            while stack:
                try: position = next(stack[-1])
                except StopIteration:
                    stack.pop()
                    if prefix: prefix.pop()
                    continue
                if len(stack) == len(query):
                    candidates.append(prefix + [position])
                    if len(candidates) > max_candidates: break
                else:
                    prefix.append(position); stack.append(choices(len(stack), position))
    truncated = len(candidates) > max_candidates
    candidates = candidates[:max_candidates]
    status = 'truncated' if truncated else 'unique' if len(candidates)==1 else 'ambiguous' if candidates else 'unaligned'
    return {'status':status, 'offsets':candidates[0].copy() if status=='unique' else None,
            'candidates':candidates, 'candidate_limit_reached':truncated}
