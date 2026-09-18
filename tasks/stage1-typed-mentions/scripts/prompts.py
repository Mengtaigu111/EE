"""Independent sentence-only and document-context Stage-1 prompts."""

TYPES = ('OG', 'APP', 'MOD', 'FEA', 'TAK', 'DST', 'LIM', 'STR', 'WEA', 'DEG')

INSTRUCTIONS = '''You are an expert in scientific information extraction.

a nugget is a minimal semantic unit that participates
in a scientific event. In this task, extract only argument nuggets
that belong to one of the 10 nugget types defined below.

OG: research team, author group, or organization.
APP: complete method, model, algorithm, system, or approach.
MOD: component of a method or architecture.
FEA: feature or information used by an approach.
TAK: research task, objective, focus, or evaluation metric.
DST: dataset, benchmark, or training data.
LIM: condition or environmental limitation.
STR: positive evaluation or strength.
WEA: weakness, limitation, or failure.
DEG: expression of degree, quantity, or magnitude.'''


def make_prompt(target_sentence, abstract=None, examples=None):
    parts = [INSTRUCTIONS]
    if abstract is not None:
        parts.append('Use the full abstract only as context.\nExtract mentions only from the target sentence.')
    parts.extend(['Do not extract event triggers.',
                  'Copy each mention exactly as it appears in the target sentence.',
                  'Return only valid JSON:\n\n{"mentions":[{"text":"...","type":"..."}]}'])
    for index, row in enumerate(examples or [], 1):
        parts.append(f"Example {index}\nTarget sentence:\n{row['target_sentence']}\n\nOutput:\n{row['completion']}")
    if abstract is not None:
        parts.append('Abstract:\n' + abstract)
    parts.append('Target sentence:\n' + target_sentence)
    return '\n\n'.join(parts)
