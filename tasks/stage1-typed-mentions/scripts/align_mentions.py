"""Deterministic target-sentence text -> document token-set alignment."""
import json

TYPES = ('OG', 'APP', 'MOD', 'FEA', 'TAK', 'DST', 'LIM', 'STR', 'WEA', 'DEG')


def parse_output(raw):
    """Strict JSON first, then the first complete object; schema errors are invalid."""
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        value = None
        if isinstance(raw, str):
            start = raw.find('{')
            depth, in_string, escaped = 0, False, False
            for i in range(start, len(raw)) if start >= 0 else []:
                char = raw[i]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == chr(92):
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            value = json.loads(raw[start:i+1])
                        except json.JSONDecodeError:
                            pass
                        break
    if not isinstance(value, dict) or not isinstance(value.get('mentions'), list):
        return [], True
    mentions = value['mentions']
    for m in mentions:
        if not isinstance(m, dict) or not isinstance(m.get('text'), str) or not m['text'].split() or m.get('type') not in TYPES:
            return [], True
    return [{'text': m['text'], 'type': m['type']} for m in mentions], False


def _contiguous(tokens, query):
    return [tuple(range(i, i + len(query))) for i in range(len(tokens) - len(query) + 1)
            if tokens[i:i + len(query)] == query]


def _subsequences(tokens, query):
    # One earliest completion per start gives a bounded, left-to-right policy.
    result = []
    for start, token in enumerate(tokens):
        if token != query[0]:
            continue
        positions = [start]
        cursor = start + 1
        for wanted in query[1:]:
            while cursor < len(tokens) and tokens[cursor] != wanted:
                cursor += 1
            if cursor == len(tokens):
                break
            positions.append(cursor)
            cursor += 1
        if len(positions) == len(query):
            result.append(tuple(positions))
    return result


def align_mentions(tokens, mentions, document_start=0):
    """Allocate occurrences separately per type so one span can carry two types.

    Repeated same-type predictions consume occurrences from left to right.
    If exact occurrences exist but are exhausted, do not invent a subsequence.
    """
    used = {}
    aligned = []
    folded = [t.casefold() for t in tokens]
    for mention in mentions:
        query = mention['text'].split()
        key = (tuple(t.casefold() for t in query), mention['type'])
        occupied = used.setdefault(key, set())
        exact = _contiguous(tokens, query) if query else []
        insensitive = _contiguous(folded, list(key[0])) if query else []
        candidates = [(p, 'exact') for p in exact]
        candidates += [(p, 'case_insensitive') for p in insensitive if p not in exact]
        if not candidates and query:
            sensitive = _subsequences(tokens, query)
            secondary = _subsequences(folded, list(key[0]))
            candidates = [(p, 'subsequence') for p in sensitive]
            candidates += [(p, 'subsequence_case_insensitive') for p in secondary if p not in sensitive]
        match = next(((p, method) for p, method in candidates if p not in occupied), None)
        item = dict(mention)
        if match is None:
            item.update(offsets=None, method='unaligned')
        else:
            p, method = match
            occupied.add(p)
            item.update(offsets=[i + document_start for i in p], method=method)
        aligned.append(item)
    return aligned
