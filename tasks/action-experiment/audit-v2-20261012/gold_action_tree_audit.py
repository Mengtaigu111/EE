"""Audit raw SciEvents gold against Actions and independently parsed compact trees.

This audit deliberately does not use the production tree renderer/parser for tree
verification. Counters retain repeated events and repeated argument occurrences.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'tasks/action-experiment/data/action-v2-goldorder-20261012'
OUT = Path(__file__).parent / 'gold-action-tree-report.json'
sys.path.insert(0, str(ROOT / 'scripts'))
from action_language import load_schema, parse_actions, render_tree


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class TreeReader:
    """A recursive grammar reader that distinguishes JSON strings from symbols."""
    def __init__(self, text):
        self.text, self.pos = text, 0

    def space(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def char(self, value):
        self.space()
        assert self.text[self.pos:self.pos + 1] == value, ('expected', value, self.pos)
        self.pos += 1

    def symbol(self):
        self.space()
        match = re.match(r'[A-Za-z][A-Za-z0-9]*', self.text[self.pos:])
        assert match is not None, ('missing symbol', self.pos)
        self.pos += match.end()
        return match.group()

    def string(self):
        self.space()
        assert self.text[self.pos:self.pos + 1] == '"', ('missing JSON string', self.pos)
        value, length = json.JSONDecoder().raw_decode(self.text[self.pos:])
        assert isinstance(value, str) and value.strip(), 'invalid mention'
        self.pos += length
        return value

    def peek(self):
        self.space()
        return self.text[self.pos:self.pos + 1]

    def read(self, schema):
        self.char('(')
        assert self.symbol() == 'ROOT'
        events = []
        while self.peek() == '(':
            self.char('(')
            kind = self.symbol()
            assert kind in schema, ('invalid event type', kind)
            trigger = self.string()  # Trigger is direct text, never a child node.
            arguments = []
            while self.peek() == '(':
                self.char('(')
                role = self.symbol()
                assert role in schema[kind], ('invalid role', kind, role)
                arguments.append({'role': role, 'text': self.string()})
                self.char(')')  # Nested span/fragment nodes cannot pass this grammar.
            self.char(')')
            events.append({'type': kind, 'trigger': trigger, 'arguments': arguments})
        self.char(')')
        self.space()
        assert self.pos == len(self.text), 'trailing content'
        return events


def raw_signature(event):
    return (event['event_type'], ' '.join(event['trigger']['tokens']),
            tuple(sorted((arg['argument_type'], ' '.join(arg['tokens']))
                         for arg in event['arguments'])))


def signature(event):
    return (event['type'], event['trigger'],
            tuple(sorted((arg['role'], arg['text']) for arg in event['arguments'])))


def main():
    schema = load_schema(DATA / 'prompt_template.txt', ROOT / 'data/SciEvents/ontology.json')
    # Independent parser edge cases: quoted brackets and escaped quotes stay text.
    probe = [{'type': 'ITT', 'trigger': 'said "(go)"',
              'arguments': [{'role': 'Target', 'text': 'a \\ b'}]}]
    assert TreeReader(render_tree(probe)).read(schema) == probe
    for malformed in ('(ROOT (ITT (Trigger "go")))',
                      '(ROOT (ITT "go" (Target (Span "a"))))'):
        try:
            TreeReader(malformed).read(schema)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('Nested trigger/fragment node accepted')

    result = {'status': 'passed', 'raw_gold_source': 'data/SciEvents/{train,dev,test}.json',
              'tree_parser': 'independent recursive JSON-string grammar reader',
              'offset_reconstruction_claimed': False,
              'splits': {}, 'failures': [], 'sha256': {}}
    for split in ('train', 'dev', 'test'):
        raw_path = ROOT / f'data/SciEvents/{split}.json'
        action_path = DATA / f'{split}.jsonl'
        result['sha256'][str(raw_path)] = sha(raw_path)
        result['sha256'][str(action_path)] = sha(action_path)
        docs = {doc['doc_id']: doc for doc in json.loads(raw_path.read_text())}
        reconstructed_documents = {doc_id: Counter() for doc_id in docs}
        counts = Counter()
        seen = set()
        for line in action_path.open():
            row = json.loads(line)
            counts['rows'] += 1
            try:
                assert row['id'] not in seen, 'duplicate row id'
                seen.add(row['id'])
                doc = docs[row['document_id']]
                assert row['tokens'] == doc['document'], 'document tokens changed'
                assert row['sentences'] == doc['sentences'], 'sentences changed'
                lower, upper = row['target_bounds']
                raw = [event for event in doc['events'] if event['trigger']['offsets']
                       and all(lower <= pos < upper for pos in event['trigger']['offsets'])]
                assert raw == row['target_events'], 'stored gold differs from raw source'
                actions = parse_actions(row['completion'], schema)
                saved_tree = TreeReader(row['compact_tree']).read(schema)
                generated_tree = TreeReader(render_tree(actions)).read(schema)
                gold = Counter(map(raw_signature, raw))
                assert Counter(map(signature, actions)) == gold, 'gold/action multiset differs'
                assert Counter(map(signature, saved_tree)) == gold, 'gold/saved tree multiset differs'
                assert Counter(map(signature, generated_tree)) == gold, 'gold/generated tree multiset differs'
                assert saved_tree == actions == generated_tree == row['action_events'], 'ordered serialization differs'
                assert len(raw) == row['n_events'], 'stored event count differs'
                reconstructed_documents[row['document_id']].update(map(signature, actions))
                counts['events'] += len(raw)
                counts['arguments'] += sum(len(event['arguments']) for event in raw)
                counts['none_rows'] += not raw
                counts['duplicate_event_rows'] += any(n > 1 for n in gold.values())
                counts['duplicate_argument_events'] += sum(
                    any(n > 1 for n in Counter((arg['argument_type'], ' '.join(arg['tokens']))
                                              for arg in event['arguments']).values())
                    for event in raw)
                counts['shared_trigger_rows'] += len({event['trigger'] for event in actions}) < len(actions)
                counts['passed_rows'] += 1
            except Exception as error:
                result['failures'].append({'split': split, 'id': row['id'], 'error': str(error)})
        for doc_id, doc in docs.items():
            if reconstructed_documents[doc_id] != Counter(map(raw_signature, doc['events'])):
                result['failures'].append({'split': split, 'document_id': doc_id,
                                           'error': 'All sentence actions do not cover raw document gold'})
            else:
                counts['complete_gold_documents'] += 1
        result['splits'][split] = dict(counts)
    result['totals'] = dict(sum((Counter(values) for values in result['splits'].values()), Counter()))
    result['status'] = 'failed' if result['failures'] else 'passed'
    result['sha256'][str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    return bool(result['failures'])


if __name__ == '__main__':
    sys.exit(main())
