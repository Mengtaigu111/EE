"""Optional positioned text-tree protocol, independent of legacy linearizers.

Grammar: (ROOT (TYPE (Trigger "text" (At I ...)) (ROLE "text" (At I ...)) ...)
...). Every mention requires nonempty, ordered, document-global zero-based
token indices. Reversed/discontinuous/repeated indices, repeated roles, and
duplicate events survive exactly. JSON quoted strings handle escaping; TYPE
and ROLE must be codes/names in the checked-in SciEvents ontology, and each
event starts with exactly one Trigger. No natural labels or semantic wrappers.

Selected source tokens must equal text.split() exactly. Supplied tokens/text
are checked during encoding; missing text is generated from selected tokens.
Decode returns SciEvents event_type/trigger/arguments with text, tokens,
offsets and argument_type. Other annotation fields are outside this protocol.
There is no alignment, deduplication, tolerance or fallback when At is absent.
This tool does not recover locations from old text-only model predictions or
change training, prompts, old gold or old evaluators.
"""
import json
from pathlib import Path
import re


_ONTOLOGY = json.loads((Path(__file__).resolve().parents[3] /
                        'data/SciEvents/ontology.json').read_text())['event_types']
_INTEGER = re.compile(r'-?(?:0|[1-9][0-9]*)\Z')


def _source(doc_tokens):
    if not isinstance(doc_tokens, (list, tuple)) or not all(isinstance(t, str) for t in doc_tokens):
        raise ValueError('Document tokens must be a list/tuple of strings')
    return doc_tokens


def _resolved(doc_tokens, offsets, text=None):
    if not isinstance(offsets, (list, tuple)) or not offsets:
        raise ValueError('Every mention requires a nonempty ordered At index list')
    if any(type(i) is not int or i < 0 or i >= len(doc_tokens) for i in offsets):
        raise ValueError(f'Invalid or out-of-bounds At indices: {offsets!r}')
    selected = [doc_tokens[i] for i in offsets]
    text = ' '.join(selected) if text is None else text
    if not isinstance(text, str) or not text.split() or text.split() != selected:
        raise ValueError(f'Mention text does not exactly match source tokens at {offsets!r}')
    return {'text': text, 'tokens': selected, 'offsets': list(offsets)}


def _encode_mention(doc_tokens, label, mention):
    if not isinstance(mention, dict) or 'offsets' not in mention:
        raise ValueError('Encoding requires mention offsets; text is never aligned')
    resolved = _resolved(doc_tokens, mention['offsets'], mention.get('text'))
    if 'tokens' in mention and mention['tokens'] != resolved['tokens']:
        raise ValueError('Mention tokens do not exactly match source indices')
    at = ' '.join(str(i) for i in resolved['offsets'])
    text = json.dumps(resolved['text'], ensure_ascii=False)
    return f'({label} {text} (At {at}))'


def encode_forest(doc_tokens, events):
    """Encode SciEvents in original event/argument/index order, without mutation."""
    _source(doc_tokens)
    trees = []
    for event in events:
        kind = event.get('event_type')
        if kind not in _ONTOLOGY:
            raise ValueError(f'Unknown event type: {kind!r}')
        if 'trigger' not in event or 'arguments' not in event:
            raise ValueError('Event requires trigger and arguments')
        parts = [_encode_mention(doc_tokens, 'Trigger', event['trigger'])]
        for arg in event['arguments']:
            role = arg.get('argument_type')
            if role not in _ONTOLOGY[kind]:
                raise ValueError(f'Invalid role {role!r} for event type {kind}')
            parts.append(_encode_mention(doc_tokens, role, arg))
        trees.append(f'({kind} ' + ' '.join(parts) + ')')
    return '(ROOT ' + ' '.join(trees) + ')' if trees else '(ROOT)'


def _lex(raw):
    if not isinstance(raw, str):
        raise ValueError('Forest must be a string')
    decoder = json.JSONDecoder()
    index, tokens = 0, []
    while index < len(raw):
        char = raw[index]
        if char.isspace():
            index += 1
        elif char in '()':
            tokens.append((char, char))
            index += 1
        elif char == '"':
            try:
                value, end = decoder.raw_decode(raw, index)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid quoted JSON string at character {index}') from error
            tokens.append(('str', value))
            index = end
        else:
            end = index + 1
            while end < len(raw) and not raw[end].isspace() and raw[end] not in '()"':
                end += 1
            tokens.append(('sym', raw[index:end]))
            index = end
    return tokens


class _Reader:
    def __init__(self, raw):
        self.tokens = _lex(raw)
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self, kind, value=None):
        token = self.peek()
        if token is None or token[0] != kind or (value is not None and token[1] != value):
            raise ValueError(f'Expected {kind} {value!r} at token {self.index}, got {token!r}')
        self.index += 1
        return token[1]


def _decode_mention(reader, doc_tokens, allowed):
    reader.take('(')
    label = reader.take('sym')
    if label not in allowed:
        raise ValueError(f'Invalid mention label: {label!r}')
    text = reader.take('str')
    reader.take('(')
    reader.take('sym', 'At')
    offsets = []
    while reader.peek() != (')', ')'):
        integer = reader.take('sym')
        if not _INTEGER.fullmatch(integer):
            raise ValueError(f'At indices must be decimal integers: {integer!r}')
        offsets.append(int(integer))
    reader.take(')')
    reader.take(')')
    return label, _resolved(doc_tokens, offsets, text)


def decode_forest(doc_tokens, raw):
    """Strictly decode positioned trees; fail rather than guess missing positions."""
    _source(doc_tokens)
    reader = _Reader(raw)
    reader.take('(')
    reader.take('sym', 'ROOT')
    events = []
    while reader.peek() != (')', ')'):
        reader.take('(')
        kind = reader.take('sym')
        if kind not in _ONTOLOGY:
            raise ValueError(f'Unknown event type: {kind!r}')
        _, trigger = _decode_mention(reader, doc_tokens, {'Trigger'})
        arguments = []
        while reader.peek() != (')', ')'):
            role, mention = _decode_mention(reader, doc_tokens, _ONTOLOGY[kind])
            arguments.append(dict(mention, argument_type=role))
        reader.take(')')
        events.append({'event_type': kind, 'trigger': trigger, 'arguments': arguments})
    reader.take(')')
    if reader.peek() is not None:
        raise ValueError('Extra tokens after ROOT forest')
    return events
