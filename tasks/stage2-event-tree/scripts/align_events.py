"""Deterministic text -> document token offsets for generated event forests.

Policy (validated on gold roundtrip: trigger 99.1%, argument 97.5% exact):
  triggers: left-to-right cursor over events in output order; first contiguous
            occurrence at or after the cursor, else first occurrence anywhere;
  arguments: contiguous occurrences preferring the trigger's sentence, then the
             smallest distance to the trigger; an argument whose text equals an
             aligned trigger text prefers that trigger's offsets (sub-events);
  fallbacks: case-insensitive, then left-to-right subsequence (discontinuous)
             inside the trigger sentence, then in the whole document.
Unaligned triggers drop the event; unaligned arguments drop the argument.
"""


def sentence_bounds(sentences):
    bounds, start = [], 0
    for sentence in sentences:
        n = len(sentence.split())
        bounds.append((start, start + n))
        start += n
    return bounds


def _occurrences(tokens, query):
    n = len(query)
    return [list(range(i, i + n)) for i in range(len(tokens) - n + 1) if tokens[i:i + n] == query]


def _subsequences(tokens, query, lo, hi):
    result = []
    for start in range(lo, hi):
        if tokens[start] != query[0]:
            continue
        positions, cursor = [start], start + 1
        for wanted in query[1:]:
            while cursor < hi and tokens[cursor] != wanted:
                cursor += 1
            if cursor == hi:
                break
            positions.append(cursor)
            cursor += 1
        if len(positions) == len(query):
            result.append(positions)
    return result


def _candidates(tokens, folded, text, scope=None, subsequence_scope=None):
    query = text.split()
    if not query:
        return [], 'empty'
    if scope is not None:
        lo, hi = scope
        if not (0 <= lo <= hi <= len(tokens)):
            raise ValueError(f'invalid alignment scope: {scope}')
        for toks, q, method in ((tokens[lo:hi], query, 'exact'),
                                (folded[lo:hi], [w.casefold() for w in query], 'case_insensitive')):
            found = [[position + lo for position in positions]
                     for positions in _occurrences(toks, q)]
            if found:
                return found, method + '_scope'
        for toks, q, method in ((tokens, query, 'subsequence'),
                                (folded, [w.casefold() for w in query], 'subsequence_case_insensitive')):
            found = _subsequences(toks, q, lo, hi)
            if found:
                return found, method + '_scope'
        return [], 'unaligned_scope'
    for toks, q, method in ((tokens, query, 'exact'), (folded, [w.casefold() for w in query], 'case_insensitive')):
        found = _occurrences(toks, q)
        if found:
            return found, method
    for toks, q, method in ((tokens, query, 'subsequence'),
                            (folded, [w.casefold() for w in query], 'subsequence_case_insensitive')):
        if subsequence_scope is not None:
            lo, hi = subsequence_scope
            found = _subsequences(toks, q, lo, hi)
            if found:
                return found, method + '_sentence'
        found = _subsequences(toks, q, 0, len(toks))
        if found:
            return found, method
    return [], 'unaligned'


def _sentence_of(bounds, position):
    for index, (lo, hi) in enumerate(bounds):
        if lo <= position < hi:
            return index
    return None


def align_forest(tokens, sentences, events, trigger_scopes=None, argument_sentence_preference=True):
    """Align text events, optionally restricting each trigger to a token scope.

    The optional arguments preserve the original document-level behavior used by
    T0 while allowing Experiment C to constrain triggers to target sentences.
    Arguments always search the full document; the sentence preference is kept
    configurable for backward-compatible T0 scoring.
    """
    if trigger_scopes is not None and len(trigger_scopes) != len(events):
        raise ValueError('trigger_scopes must have one entry per event')
    folded = [t.casefold() for t in tokens]
    bounds = sentence_bounds(sentences)
    stats = {'events': len(events), 'aligned_events': 0, 'unaligned_triggers': 0,
             'arguments': 0, 'aligned_arguments': 0, 'unaligned_arguments': 0,
             'trigger_methods': {}, 'argument_methods': {}}
    aligned, cursor, trigger_by_text = [], 0, {}
    last_by_type_text = {}
    for index, event in enumerate(events):
        scope = trigger_scopes[index] if trigger_scopes is not None else None
        if scope is not None:
            cursor = max(cursor, scope[0])
        found, method = _candidates(tokens, folded, event['trigger'], scope=scope)
        if not found:
            stats['unaligned_triggers'] += 1
            stats['arguments'] += len(event['args'])
            stats['unaligned_arguments'] += len(event['args'])
            continue
        contiguous = [p for p in found if p[-1] - p[0] + 1 == len(p)]
        pool = contiguous or found
        type_text = (event['type'], event['trigger'])
        if trigger_scopes is not None and type_text in last_by_type_text:
            ahead = [p for p in pool if p[0] > last_by_type_text[type_text][-1]]
            trigger = (ahead or [last_by_type_text[type_text]])[0]
        elif trigger_scopes is not None and event['trigger'] in trigger_by_text:
            # Different event types can legitimately share one trigger nugget.
            trigger = trigger_by_text[event['trigger']][-1]
        else:
            ahead = [p for p in pool if p[0] >= cursor]
            trigger = (ahead or pool)[0]
        cursor = trigger[0]
        stats['trigger_methods'][method] = stats['trigger_methods'].get(method, 0) + 1
        trigger_by_text.setdefault(event['trigger'], []).append(trigger)
        last_by_type_text[type_text] = trigger
        aligned.append({'event_type': event['type'],
                        'trigger': {'text': event['trigger'], 'tokens': [tokens[i] for i in trigger], 'offsets': trigger},
                        'arguments': [], '_args': event['args']})
        stats['aligned_events'] += 1
    for event in aligned:
        trigger = event['trigger']['offsets']
        sentence = _sentence_of(bounds, trigger[0])
        lo, hi = bounds[sentence] if sentence is not None else (0, len(tokens))
        for arg in event.pop('_args'):
            stats['arguments'] += 1
            subsequence_scope = (lo, hi) if argument_sentence_preference else None
            found, method = _candidates(tokens, folded, arg['text'],
                                         subsequence_scope=subsequence_scope)
            if not found:
                stats['unaligned_arguments'] += 1
                continue
            if arg['text'] in trigger_by_text:
                # Prefer an aligned trigger with identical text: recovers sub-event links.
                found = [p for p in trigger_by_text[arg['text']] if p in found] or found
            best = min(found, key=lambda p: (0 if argument_sentence_preference and lo <= p[0] < hi else 1,
                                             abs(p[0] - trigger[0]), p[0]))
            stats['argument_methods'][method] = stats['argument_methods'].get(method, 0) + 1
            stats['aligned_arguments'] += 1
            event['arguments'].append({'text': arg['text'], 'tokens': [tokens[i] for i in best],
                                       'offsets': best, 'argument_type': arg['role']})
    return aligned, stats
