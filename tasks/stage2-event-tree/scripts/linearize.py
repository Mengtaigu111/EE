"""Canonical event-forest linearization and its tolerant inverse parser.

Tree grammar (one abstract = one forest under a virtual ROOT):

    (ROOT (PRP (Trigger "propose") (Proposer "we") (Content "a new method")) (CMP ...))

Canonical order: events by (first trigger offset, event type); Trigger first;
roles in ontology order; same-role arguments by first offset. Nugget types never
appear in the tree. Strings escape backslash and double quote.
"""
import re

from ontology import EVENT_TYPES, NAME_TO_CODE, ROLE_ORDER, event_label, role_index


def span_text(tokens, offsets):
    return ' '.join(tokens[i] for i in sorted(set(offsets)))


def escape(text):
    return text.replace('\\', '\\\\').replace('"', '\\"')


def canonical_events(doc):
    """Gold SciEvents events -> canonical list of text-level events (offsets kept for audit)."""
    tokens = doc['document']
    events = []
    for event in doc['events']:
        trigger = sorted(set(event['trigger']['offsets']))
        args = []
        for arg in event['arguments']:
            offsets = sorted(set(arg['offsets']))
            args.append({'role': arg['argument_type'], 'text': span_text(tokens, offsets), 'offsets': offsets})
        args.sort(key=lambda a: (role_index(event['event_type'], a['role']), a['offsets'][0], a['offsets'][-1]))
        events.append({'type': event['event_type'], 'trigger': span_text(tokens, trigger),
                       'trigger_offsets': trigger, 'args': args})
    events.sort(key=lambda e: (e['trigger_offsets'][0], e['trigger_offsets'][-1], e['type']))
    return events


def linearize(events, natural=False):
    parts = []
    for event in events:
        body = [f'(Trigger "{escape(event["trigger"])}")']
        body += [f'({a["role"]} "{escape(a["text"])}")' for a in event['args']]
        parts.append(f'({event_label(event["type"], natural)} ' + ' '.join(body) + ')')
    return '(ROOT ' + ' '.join(parts) + ')' if parts else '(ROOT)'


def linearize_substructures(events, natural=False):
    """Text2Event curriculum stage: flat (label "span") pairs, no tree nesting."""
    parts = []
    for event in events:
        parts.append(f'({event_label(event["type"], natural)} "{escape(event["trigger"])}")')
        parts += [f'({a["role"]} "{escape(a["text"])}")' for a in event['args']]
    return ' '.join(parts) if parts else '()'


_TOKEN = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')


def _lex(text):
    out = []
    for match in _TOKEN.finditer(text):
        tok = match.group(0)
        if tok.startswith('"'):
            body = tok[1:-1] if tok.endswith('"') and len(tok) > 1 else tok[1:]
            out.append(('str', body.replace('\\"', '"').replace('\\\\', '\\')))
        elif tok in '()':
            out.append((tok, tok))
        else:
            out.append(('sym', tok))
    return out


def _parse_sexpr(tokens):
    """Tolerant S-expression reader: unbalanced closes are ignored, EOF closes all."""
    stack = [[]]
    for kind, value in tokens:
        if kind == '(':
            stack.append([])
        elif kind == ')':
            if len(stack) > 1:
                node = stack.pop()
                stack[-1].append(node)
        else:
            stack[-1].append((kind, value))
    while len(stack) > 1:
        node = stack.pop()
        stack[-1].append(node)
    return stack[0]


def parse_forest(raw):
    """Model text -> (events, diagnostics). Never raises; drops malformed pieces and counts them."""
    diag = {'invalid': False, 'dropped_events': 0, 'dropped_args': 0, 'unknown_event_types': 0,
            'unknown_roles': 0, 'missing_trigger': 0}
    if not isinstance(raw, str) or '(' not in raw:
        diag['invalid'] = True
        return [], diag
    trees = _parse_sexpr(_lex(raw))
    root = None
    for node in trees:
        if isinstance(node, list) and node and node[0] == ('sym', 'ROOT'):
            root = node
            break
    if root is None:
        # Accept a bare sequence of event trees without ROOT.
        root = [('sym', 'ROOT')] + [n for n in trees if isinstance(n, list)]
        if len(root) == 1:
            diag['invalid'] = True
            return [], diag
    events = []
    for node in root[1:]:
        if not isinstance(node, list) or not node or node[0][0] != 'sym':
            diag['dropped_events'] += 1
            continue
        event_type = NAME_TO_CODE.get(node[0][1], node[0][1])
        if event_type not in EVENT_TYPES:
            diag['unknown_event_types'] += 1
            diag['dropped_events'] += 1
            continue
        trigger, args = None, []
        for child in node[1:]:
            if (not isinstance(child, list) or len(child) < 2 or child[0][0] != 'sym'
                    or child[1][0] != 'str'):
                diag['dropped_args'] += 1
                continue
            label, text = child[0][1], child[1][1].strip()
            if not text:
                diag['dropped_args'] += 1
                continue
            if label == 'Trigger':
                if trigger is None:
                    trigger = text
                else:
                    diag['dropped_args'] += 1
            elif label in ROLE_ORDER[event_type]:
                args.append({'role': label, 'text': text})
            else:
                diag['unknown_roles'] += 1
                diag['dropped_args'] += 1
        if trigger is None:
            diag['missing_trigger'] += 1
            diag['dropped_events'] += 1
            continue
        events.append({'type': event_type, 'trigger': trigger, 'args': args})
    return events, diag
