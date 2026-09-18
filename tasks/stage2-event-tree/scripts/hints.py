"""Typed-mention hints for Stage-2: gold, Stage-1 predictions, and calibrated noise."""
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from ontology import NUGGET_TYPES

STAGE1_SCRIPTS = Path(__file__).resolve().parents[2] / 'stage1-typed-mentions/scripts'


def gold_hints(doc):
    """Unique (offsets, type) argument mentions of the 10 nugget types, in document order."""
    tokens = doc['document']
    seen, out = set(), []
    for event in doc['events']:
        for arg in event['arguments']:
            kind = arg['nugget_type']
            if kind not in NUGGET_TYPES:
                continue
            offsets = tuple(sorted(set(arg['offsets'])))
            if (offsets, kind) in seen:
                continue
            seen.add((offsets, kind))
            out.append({'text': ' '.join(tokens[i] for i in offsets), 'type': kind, 'offsets': list(offsets)})
    out.sort(key=lambda m: (m['offsets'][0], m['offsets'][-1], m['type']))
    return out


def load_stage1_hints(predictions_path):
    """Stage-1 sentence predictions -> {document_id: [hint]} aligned with the Stage-1 aligner."""
    if str(STAGE1_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(STAGE1_SCRIPTS))
    from align_mentions import align_mentions  # noqa: E402
    per_doc, seen = {}, {}
    with Path(predictions_path).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            doc = row['document_id']
            per_doc.setdefault(doc, [])
            keys = seen.setdefault(doc, set())
            if row.get('invalid'):
                continue
            for m in align_mentions(row['tokens'], row['parsed_mentions'], row['document_start']):
                if m['offsets'] is None:
                    continue
                key = (tuple(m['offsets']), m['type'])
                if key in keys:
                    continue
                keys.add(key)
                per_doc[doc].append({'text': m['text'], 'type': m['type'], 'offsets': list(m['offsets'])})
    for hints in per_doc.values():
        hints.sort(key=lambda m: (m['offsets'][0], m['offsets'][-1], m['type']))
    return per_doc


def noise_profile_from_stage1(metrics_path):
    """Calibrate the training-time noise model from a Stage-1 dev metrics JSON."""
    m = json.loads(Path(metrics_path).read_text())
    fn, fp = m['errors']['FN'], m['errors']['FP']
    total_fn = sum(fn.values())
    return {'source': str(Path(metrics_path).resolve()),
            'per_type_recall': {t: (v['recall'] if v['recall'] is not None else 0.5) for t, v in m['per_type'].items()},
            'error_mix': {k: fn[k] / total_fn for k in ('BOUNDARY', 'TYPE_ONLY', 'MISSING')},
            'spurious_per_sentence': fp['OTHER'] / m['samples'],
            'boundary_longer_share': 0.6}


def _rng(seed, doc_id):
    digest = hashlib.sha256(f'{seed}:{doc_id}'.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def noised_hints(doc, gold, profile, seed):
    """Simulate Stage-1 errors on gold hints: per-type misses, type flips, boundary jitter, spurious spans."""
    rng = _rng(seed, doc['doc_id'])
    tokens = doc['document']
    mix = profile['error_mix']
    kinds = ['BOUNDARY', 'TYPE_ONLY', 'MISSING']
    weights = [mix[k] for k in kinds]
    occupied = {tuple(m['offsets']) for m in gold}
    out = []
    for m in gold:
        if rng.random() < profile['per_type_recall'][m['type']]:
            out.append(dict(m))
            continue
        kind = rng.choices(kinds, weights)[0]
        if kind == 'MISSING':
            continue
        if kind == 'TYPE_ONLY':
            out.append(dict(m, type=rng.choice([t for t in NUGGET_TYPES if t != m['type']])))
            continue
        lo, hi = m['offsets'][0], m['offsets'][-1]
        longer = rng.random() < profile['boundary_longer_share']
        if longer:
            if rng.random() < 0.5 and lo > 0:
                lo -= 1
            elif hi < len(tokens) - 1:
                hi += 1
            else:
                lo = max(0, lo - 1)
        elif hi > lo:
            if rng.random() < 0.5:
                lo += 1
            else:
                hi -= 1
        else:
            continue  # single token cannot shrink: treat as missing
        offsets = list(range(lo, hi + 1))
        out.append({'text': ' '.join(tokens[i] for i in offsets), 'type': m['type'], 'offsets': offsets})
    n_sentences = len(doc['sentences'])
    for _ in range(n_sentences):
        count = rng.choices([0, 1, 2, 3], [0.40, 0.35, 0.18, 0.07])[0]
        for _ in range(count):
            for _ in range(5):
                length = rng.choice([1, 1, 2, 2, 3])
                start = rng.randrange(0, max(1, len(tokens) - length))
                offsets = tuple(range(start, start + length))
                if offsets in occupied or any(not tokens[i].isalnum() and tokens[i] not in '-' for i in offsets):
                    continue
                occupied.add(offsets)
                out.append({'text': ' '.join(tokens[i] for i in offsets), 'type': rng.choice(NUGGET_TYPES), 'offsets': list(offsets)})
                break
    out.sort(key=lambda m: (m['offsets'][0], m['offsets'][-1], m['type']))
    return out


def render_flat(hints):
    if not hints:
        return 'Candidate mentions: (none)'
    return 'Candidate mentions (text -> type):\n' + '\n'.join(f'- "{m["text"]}" -> {m["type"]}' for m in hints)


def render_typed(hints):
    groups = {t: [] for t in NUGGET_TYPES}
    for m in hints:
        groups[m['type']].append(m['text'])
    lines = [f'{t}: ' + (' | '.join(f'"{x}"' for x in groups[t]) if groups[t] else '(none)') for t in NUGGET_TYPES]
    return 'Candidate mentions grouped by nugget type:\n' + '\n'.join(lines)


def summarize(hints_by_doc):
    counts = Counter(m['type'] for hints in hints_by_doc.values() for m in hints)
    return {'documents': len(hints_by_doc), 'mentions': sum(counts.values()), 'by_type': dict(counts)}
