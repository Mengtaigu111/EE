"""Stage-2 prompts: schema-guided event forest generation with optional typed-mention guidance."""
from ontology import EVENT_GLOSS, EVENT_TYPES, ROLE_TYPES, TYPE_GLOSS, NUGGET_TYPES, event_label
from hints import render_flat, render_typed

MODES = ('none', 'flat', 'typed', 'noschema', 'natural', 'sub')
NATURAL_MODES = ('natural',)
HINT_MODES = ('flat', 'typed')

SUB_TASK = '''You are an expert in scientific information extraction.
Read the abstract and list every event trigger and every event argument it contains.

Write one (Label "text") pair per item, separated by spaces, in order of appearance:
- for a trigger, Label is its event type, e.g. (PRP "propose")
- for an argument, Label is its argument role, e.g. (Proposer "we")
Copy every text exactly as it appears in the abstract. If there is nothing, output ().'''

TASK = '''You are an expert in scientific information extraction.
Read the abstract and generate ALL scientific events it contains as one event forest.

Each event is a tree: the event type is the root, the trigger is the first child,
and each argument is a (Role "text") child. Nugget types are never written in the tree.

Output format (one line, S-expression):
(ROOT (PRP (Trigger "propose") (Proposer "we") (Content "a new framework")) (CMP (Trigger "outperforms") (Arg1 "our method") (Arg2 "baselines") (Dataset "SciEvents")))

Rules:
- Copy every trigger and argument text exactly as it appears in the abstract (same tokens, same order).
- Order events by trigger position in the abstract. Inside an event: Trigger first, then roles in the schema order below; several arguments of the same role go in text order.
- An argument may be the trigger of another event (a sub-event); then write that trigger text as the argument.
- If the abstract has no events, output (ROOT).'''


SENTENCE_TASK = '''You are an expert in scientific information extraction.
Read the full abstract for context, but extract only the scientific events whose
TRIGGER occurs in the TARGET SENTENCE. Arguments may appear anywhere in the
full abstract.

Each event is a tree: the event type is the root, the trigger is the first child,
and each argument is a (Role "text") child. Nugget types are never written in
the tree. Copy every trigger and argument text exactly as it appears in the
abstract. Order events by trigger position in the target sentence. Inside an
event: Trigger first, then roles in the schema order below; several arguments
of the same role go in text order.

If the target sentence has no event, output (ROOT).'''


def schema_block(with_types, natural=False):
    lines = []
    for event in EVENT_TYPES:
        roles = ROLE_TYPES[event]
        if with_types:
            role_text = ', '.join(f'{role}[{"/".join(types)}]' for role, types in roles.items())
        else:
            role_text = ', '.join(roles)
        lines.append(f'{event_label(event, natural)} ({EVENT_GLOSS[event]}): {role_text}')
    block = 'Event schema (event type: roles' + (' with allowed nugget types; E-X means the trigger of an X event' if with_types else '') + '):\n' + '\n'.join(lines)
    if with_types:
        block += '\n\nNugget types:\n' + '\n'.join(f'{t}: {TYPE_GLOSS[t]}' for t in NUGGET_TYPES)
    return block


def guidance_block(mode, hints):
    if mode == 'none':
        return None
    intro = ('The following candidate mentions were proposed by a separate mention extractor. '
             'They are hints only: they may be incomplete, wrongly typed, or spurious. '
             'Use them as evidence, but always ground every argument in the abstract itself.')
    return intro + '\n\n' + (render_flat(hints) if mode == 'flat' else render_typed(hints))


def make_prompt(abstract, mode='none', hints=None):
    if mode not in MODES:
        raise ValueError(f'Unknown guidance mode: {mode}')
    natural = mode in NATURAL_MODES
    if mode == 'sub':
        parts = [SUB_TASK, schema_block(with_types=False), 'Abstract:\n' + abstract, 'Pairs:']
        return '\n\n'.join(parts)
    task = TASK
    if natural:
        task = task.replace('(PRP (Trigger', '(Proposal (Trigger').replace('(CMP (Trigger', '(Comparison (Trigger')
    parts = [task]
    if mode == 'noschema':
        parts[0] = task.replace('then roles in the schema order below', 'then its argument roles')
    else:
        parts.append(schema_block(with_types=(mode == 'typed'), natural=natural))
    guidance = guidance_block(mode, hints or []) if mode in HINT_MODES else None
    if guidance:
        parts.append(guidance)
    parts.append('Abstract:\n' + abstract)
    parts.append('Event forest:')
    return '\n\n'.join(parts)


def make_sentence_prompt(abstract, sentences, target_index):
    """Build C's full-context prompt with one explicitly selected target sentence."""
    if not sentences or not 0 <= target_index < len(sentences):
        raise ValueError('target_index must identify a sentence')
    if ' '.join(sentences).split() != abstract.split():
        raise ValueError('sentences must tile the abstract')
    marked = '\n'.join(f'[S{i + 1}] {sentence}' for i, sentence in enumerate(sentences))
    parts = [SENTENCE_TASK, schema_block(with_types=False),
             'Full Abstract:\n' + marked,
             f'TARGET SENTENCE [S{target_index + 1}]:\n{sentences[target_index]}',
             'Event forest:']
    return '\n\n'.join(parts)
