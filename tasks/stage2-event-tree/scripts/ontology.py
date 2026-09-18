"""SciEvents ontology: event types, canonical role order, role -> allowed nugget types."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ONTOLOGY_PATH = PROJECT_ROOT / 'data/SciEvents/ontology.json'

NUGGET_TYPES = ('OG', 'APP', 'MOD', 'FEA', 'TAK', 'DST', 'LIM', 'STR', 'WEA', 'DEG')

TYPE_GLOSS = {
    'OG': 'research team, author group, or organization',
    'APP': 'complete method, model, algorithm, system, or approach',
    'MOD': 'component of a method or architecture',
    'FEA': 'feature or information used by an approach',
    'TAK': 'research task, objective, focus, or evaluation metric',
    'DST': 'dataset, benchmark, or training data',
    'LIM': 'condition or environmental limitation',
    'STR': 'positive evaluation or strength',
    'WEA': 'weakness, limitation, or failure',
    'DEG': 'expression of degree, quantity, or magnitude',
}

EVENT_NAMES = {  # natural-language labels (Text2Event: labels must be words, not symbols)
    'PUR': 'Purpose', 'PRP': 'Proposal', 'WKS': 'WorkSummary', 'ITT': 'Investigation',
    'RWS': 'RelatedSolution', 'RWF': 'RelatedFault', 'MDS': 'MethodDesign',
    'FIN': 'Finding', 'CMP': 'Comparison', 'FAC': 'OutcomeFact',
}
NAME_TO_CODE = {v: k for k, v in EVENT_NAMES.items()}


def event_label(code, natural=False):
    return EVENT_NAMES[code] if natural else code


EVENT_GLOSS = {
    'PUR': 'purpose or goal of the research',
    'PRP': 'proposal of a method, model, or resource',
    'WKS': 'work summary of what the paper does',
    'ITT': 'investigation or analysis of a target',
    'RWS': 'related-work solution described by the paper',
    'RWF': 'related-work fault or limitation',
    'MDS': 'method design: how a component is used or built',
    'FIN': 'finding statement introducing a fact or comparison',
    'CMP': 'comparison between two subjects',
    'FAC': 'outcome fact or result',
}


def load_ontology(path=ONTOLOGY_PATH):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if tuple(data['nugget_types']) != NUGGET_TYPES:
        raise ValueError('Unexpected nugget type inventory in ontology.json')
    return data['event_types']


ROLE_TYPES = load_ontology()  # event_type -> {role: [allowed nugget/E-types]} (insertion order = canonical role order)
EVENT_TYPES = tuple(ROLE_TYPES)
ROLE_ORDER = {event: tuple(roles) for event, roles in ROLE_TYPES.items()}


def role_index(event_type, role):
    order = ROLE_ORDER[event_type]
    return order.index(role) if role in order else len(order)


def compatible_types(event_type, role):
    """Plain nugget types accepted by a role; E-* entries mean 'trigger of another event'."""
    return tuple(t for t in ROLE_TYPES[event_type][role] if not t.startswith('E-'))


def accepts_subevent(event_type, role):
    return any(t.startswith('E-') for t in ROLE_TYPES[event_type][role])
