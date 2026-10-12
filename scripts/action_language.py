"""Strict text Action codec; preserve event/role multiplicity and mention word order."""
from collections import Counter
import json
from pathlib import Path
import re


def load_schema(prompt_path, ontology_path=None):
    prompt = Path(prompt_path).read_text(encoding='utf-8')
    section = prompt.split('APPENDIX F.2 EVENT SCHEMA\n', 1)[1].split('\nFORMAT EXAMPLE', 1)[0]
    schema = {}
    current = None
    for line in section.splitlines():
        match = re.fullmatch(r'([A-Z]{3}) \(.+\)', line)
        if match:
            current = match[1]
            if current in schema:
                raise ValueError('Duplicate event schema')
        elif line.startswith('Roles: '):
            if current is None or current in schema:
                raise ValueError('Missing/duplicate type for role schema')
            roles = tuple(line[7:].split(', '))
            if not roles or len(roles) != len(set(roles)):
                raise ValueError('Empty/duplicate roles')
            schema[current] = roles
    if len(schema) != 10:
        raise ValueError('Expected ten event types')
    if ontology_path:
        ontology = json.loads(Path(ontology_path).read_text())['event_types']
        if set(schema) != set(ontology) or any(set(schema[t]) != set(ontology[t]) for t in schema):
            raise ValueError('Prompt schema differs from ontology role inventory')
    return schema


def mention_text(mention):
    tokens = mention.get('tokens')
    text = ' '.join(tokens) if tokens is not None else mention['text']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty/non-string mention')
    return text


def canonical_events(events, schema):
    """Offsets only determine ordering; never sort tokens inside a mention."""
    output = []
    for event in sorted(events, key=lambda e: min(e['trigger']['offsets'])):
        kind = event['event_type']
        if kind not in schema:
            raise ValueError(f'Unknown event type: {kind}')
        arguments = event['arguments']
        if any(a['argument_type'] not in schema[kind] for a in arguments):
            raise ValueError(f'Illegal role for {kind}')
        arguments = sorted(arguments, key=lambda a: (schema[kind].index(a['argument_type']), min(a['offsets'])))
        output.append({'type': kind, 'trigger': mention_text(event['trigger']),
                       'arguments': [{'role': a['argument_type'], 'text': mention_text(a)} for a in arguments]})
    return output


def parse_actions(output, schema):
    if not isinstance(output, str) or not output:
        raise ValueError('Empty/non-string actions')
    events, current, state = [], None, 'between'
    lines = output.splitlines()
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or state == 'done':
            raise ValueError(f'Unexpected content at line {number}')
        if line == 'STOP' and state == 'between':
            state = 'done'
            continue
        match = re.fullmatch(r'NEW\(([A-Z]{3})\)', line)
        if match and state == 'between' and match[1] in schema:
            current = {'type': match[1], 'arguments': []}
            state = 'trigger'
            continue
        if line.startswith('TRG(') and line.endswith(')') and state == 'trigger':
            current['trigger'] = _string(line[4:-1])
            state = 'arguments'
            continue
        match = re.fullmatch(r'ARG\(([A-Za-z][A-Za-z0-9]*),\s*(.*)\)', line)
        if match and state == 'arguments' and match[1] in schema[current['type']]:
            current['arguments'].append({'role': match[1], 'text': _string(match[2])})
            continue
        if line == 'END' and state == 'arguments':
            events.append(current)
            current, state = None, 'between'
            continue
        raise ValueError(f'Illegal action/state at line {number}: {line!r}')
    if state != 'done':
        raise ValueError('Missing END/STOP')
    return events


def _string(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError('Invalid JSON mention string') from error
    if not isinstance(result, str) or not result.strip():
        raise ValueError('Empty/non-string mention')
    return result


def render_actions(events, schema):
    lines = []
    for event in events:
        lines.extend([f"NEW({event['type']})", f"TRG({json.dumps(event['trigger'], ensure_ascii=False)})"])
        for arg in event['arguments']:
            lines.append(f"ARG({arg['role']}, {json.dumps(arg['text'], ensure_ascii=False)})")
        lines.append('END')
    lines.append('STOP')
    result = '\n'.join(lines)
    if parse_actions(result, schema) != events:
        raise ValueError('Action roundtrip differs')
    return result


def render_tree(events):
    nodes = []
    for event in events:
        args = ''.join(f" ({a['role']} {json.dumps(a['text'], ensure_ascii=False)})" for a in event['arguments'])
        nodes.append(f"({event['type']} {json.dumps(event['trigger'], ensure_ascii=False)}{args})")
    return '(ROOT' + ((' ' + ' '.join(nodes)) if nodes else '') + ')'


def event_multiset(events):
    """Ignore ordering while retaining event and argument occurrence counts."""
    return Counter((e['type'], e['trigger'], tuple(sorted((a['role'], a['text']) for a in e['arguments']))) for e in events)
