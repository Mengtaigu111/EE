"""Stage-2 prompts: schema-guided event forest generation with optional typed-mention guidance."""
from pathlib import Path
import re

from ontology import EVENT_GLOSS, EVENT_TYPES, ROLE_TYPES, TYPE_GLOSS, NUGGET_TYPES, event_label
from hints import render_flat, render_typed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
APPENDIX_F2_EXACT = PROJECT_ROOT / 'task/scievents_appendix_f2_exact_reference_v2.md'

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


APPENDIX_SENTENCE_TASK = '''You are an expert in scientific information extraction.
Read the full abstract for context, but extract only the scientific events whose
TRIGGER occurs in the TARGET SENTENCE. Arguments may appear anywhere in the
full abstract.

Each event is a tree: the event type is the root, the trigger is the first child,
and each argument is a (Role "text") child. Order events by trigger position in
the target sentence. Inside an event: Trigger first, then roles in the schema
order below; several arguments of the same role go in text order.

If the target sentence has no event, output (ROOT).'''


SEMANTIC_SENTENCE_TASK = '''You are an expert in scientific event extraction.

Given a full scientific abstract and one TARGET SENTENCE, identify all scientific events expressed in the target sentence.

For each event, determine its semantic path and event type first, then extract
its Trigger and Arguments. Use the full abstract as context when necessary,
including arguments mentioned outside the target sentence.

A sentence may contain multiple events.

====================
Semantic Event Tree
====================

Background
  -> ITT

RelatedWork
  -> PriorWorkProcedure -> RWS
  -> PriorWorkLimitation -> RWF

Methodology
  -> ProposerIntroducesContent -> PRP
  -> ResearcherPerformsWork -> WKS
  -> ComponentAssistedOperation -> MDS

Results
  -> FinderReport -> FIN
  -> ExplicitComparison -> CMP
  -> SubjectOutcome -> FAC

Purpose
  -> PUR


====================
Semantic Prototypes
====================

Use these templates as semantic prototypes for event-type discrimination.
They describe typical semantic structures, not mandatory surface forms.

PUR:
In order to deal with <Aim:arg1> under <Condition:arg2> circumstance on
<Dataset:arg3> datasets.

ITT:
<Target:arg1> is the abstract research target under <Condition:arg2>
circumstance on <Dataset:arg3> datasets in this paper.

RWS:
Previously <Subject:arg1> on <Target:arg2> are mostly based on
<BaseComponent:arg3> with <TriedComponent:arg4> under <Condition:arg5>
circumstance on <Dataset:arg6> datasets.

RWF:
Aiming to <Target:arg1>, to <Extent:arg5> degree, <Concern:arg2> has some
<Fault:arg6> faults under <Condition:arg3> circumstance on
<Dataset:arg4> datasets.

PRP:
In this paper, <Proposer:arg1> propose <Content:arg2> for <Target:arg3>.

WKS:
<Researcher:arg1> report <Content:arg2> under <Condition:arg3> circumstance
on <Dataset:arg4> datasets for <Target:arg5>.

MDS:
Our approach adopt <BaseComponent:arg1> with <TriedComponent:arg2> under
<Condition:arg3> circumstance on <Dataset:arg4> datasets for <Target:arg5>.

FIN:
In experiments, <Finder:arg1> find or demonstrate findings that <Content:arg2>.

CMP:
Experimental results show that the <Metrics:arg6> of <Arg1:arg1> is
<Extent:arg2> <Result:arg3> than <Arg2:arg4> under <Condition:arg5>
circumstance on <Dataset:arg7> datasets.

FAC:
Experimental results show that <Subject:arg1> can <Extent:arg2> provide
<Object:arg3> for <Target:arg4> under <Condition:arg5> circumstance on
<Dataset:arg6> datasets because <Reason:arg7> reasons.

====================
Output
====================

Generate one event forest. Each event is an independent branch under ROOT:

(ROOT
  (Region
    (SemanticDecision
      (EVENT_TYPE
        (Trigger "...")
        (Role "..."))))
  (Region
    ...
  ))

Background and Purpose have no SemanticDecision node.


For multiple events, include all event branches under the same ROOT.


Rules:
- Extract an event only if its Trigger occurs in the TARGET SENTENCE.
- Arguments may appear out of target sentence.
- Copy Trigger and Argument text exactly as it appears in the abstract.
- Use only roles licensed by the selected prototype.
- Do not invent missing arguments.
- Each event forms an independent branch under ROOT.
- Order event branches by Trigger position in the TARGET SENTENCE.
- Do not output YES/NO decisions.
- Do not predict Trigger or Arguments before the Event Type.
- If no event is expressed, output (ROOT).'''


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


def appendix_schema_block():
    """Render Appendix F.2 exact templates instead of ontology.py's informal glosses."""
    text = APPENDIX_F2_EXACT.read_text(encoding='utf-8')
    lines = []
    for code in ('PUR', 'ITT', 'RWS', 'RWF', 'PRP', 'WKS', 'MDS', 'FIN', 'CMP', 'FAC'):
        match = re.search(
            rf'^### {code} — ([^\n]+)\n\n```text\n(.*?)\n```\n\nRoles:\n\n```text\n(.*?)\n```',
            text,
            re.S | re.M,
        )
        if not match:
            raise RuntimeError(f'Appendix F.2 exact schema not found for {code}')
        full_name, template, roles = match.groups()
        role_text = ', '.join(role.strip() for role in roles.splitlines() if role.strip())
        lines.append(f'{code} ({full_name.strip()})\nTemplate: {template.strip()}\nRoles: {role_text}')
    return 'Appendix F.2 exact event schema (event type: full name, template, roles):\n' + '\n'.join(lines)


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


def make_appendix_sentence_prompt(abstract, sentences, target_index):
    """Build C's sentence prompt, but with Appendix F.2 exact schema text."""
    if not sentences or not 0 <= target_index < len(sentences):
        raise ValueError('target_index must identify a sentence')
    if ' '.join(sentences).split() != abstract.split():
        raise ValueError('sentences must tile the abstract')
    marked = '\n'.join(f'[S{i + 1}] {sentence}' for i, sentence in enumerate(sentences))
    parts = [APPENDIX_SENTENCE_TASK, appendix_schema_block(),
             'Full Abstract:\n' + marked,
             f'TARGET SENTENCE [S{target_index + 1}]:\n{sentences[target_index]}',
             'Event forest:']
    return '\n\n'.join(parts)


def make_semantic_sentence_prompt(abstract, sentences, target_index):
    """Build a semantic-path full-context prompt with one selected target sentence."""
    if not sentences or not 0 <= target_index < len(sentences):
        raise ValueError('target_index must identify a sentence')
    if ' '.join(sentences).split() != abstract.split():
        raise ValueError('sentences must tile the abstract')
    marked = '\n'.join(f'[S{i + 1}] {sentence}' for i, sentence in enumerate(sentences))
    parts = [SEMANTIC_SENTENCE_TASK,
             'Full Abstract:\n' + marked,
             f'TARGET SENTENCE [S{target_index + 1}]:\n{sentences[target_index]}',
             'Event forest:']
    return '\n\n'.join(parts)
