import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from action_language import parse_actions, render_actions, render_tree, canonical_events, load_schema
from build_action_dataset import make_row

SCHEMA = load_schema(ROOT/'task/101202_action_prompt_v1.txt')

def event(kind='PRP', text='propose', args=None, offset=1):
    return dict(event_type=kind, trigger=dict(text=text,tokens=[text],offsets=[offset]),
                arguments=args or [])

def arg(role,text,offsets):
    return dict(argument_type=role,text=text,tokens=text.split(),offsets=offsets)

def test_instances_and_repeated_roles_survive():
    events=[event(args=[arg('Content','model',[2]),arg('Content','model',[2])])]*2
    expected=canonical_events(events,SCHEMA)
    actions=render_actions(expected,SCHEMA)
    assert actions.count('NEW(PRP)')==2 and actions.count('ARG(Content,')==4
    assert parse_actions(actions,SCHEMA)==expected
    assert render_tree(expected).count('(PRP ')==2

def test_escapes_and_noncontinuous_original_word_order():
    events=[event(text='say"\\',args=[arg('Content','b a',[3,2])])]
    canonical=canonical_events(events,SCHEMA)
    assert canonical[0]['arguments'][0]['text']=='b a'
    assert parse_actions(render_actions(canonical,SCHEMA),SCHEMA)==canonical
    assert parse_actions('NEW(PRP)\nTRG("say\\nnext")\nEND\nSTOP',SCHEMA)[0]['trigger']=='say\nnext'

def test_prompt_role_order_not_ontology_order():
    assert SCHEMA['RWS']==('Subject','Target','BaseComponent','TriedComponent','Condition','Dataset')
    e=event(kind='RWS',args=[arg('BaseComponent','base',[8]),arg('Target','target',[9]),arg('Subject','subject',[10])])
    assert [a['role'] for a in canonical_events([e],SCHEMA)[0]['arguments']]==['Subject','Target','BaseComponent']

def test_stable_event_and_argument_occurrence_order():
    e=event(args=[arg('Content','late',[9]),arg('Content','earlier',[2])])
    out=canonical_events([event(kind='MDS',offset=3),e,event(kind='FIN')],SCHEMA)
    assert [r['type'] for r in out]==['PRP','FIN','MDS']
    assert [a['text'] for a in out[0]['arguments']]==['earlier','late']

def test_empty():
    assert render_actions([],SCHEMA)=='STOP'
    assert parse_actions('STOP',SCHEMA)==[]
    assert render_tree([])=='(ROOT)'

@pytest.mark.parametrize('output',[
    '', 'END\nSTOP','NEW(WRONG)\nTRG("x")\nEND\nSTOP',
    'NEW(PRP)\nTRG("x")\nARG(Aim, "y")\nEND\nSTOP',
    'NEW(PRP)\nEND\nSTOP','NEW(PRP)\nTRG("x")\nTRG("y")\nEND\nSTOP',
    'NEW(PRP)\nTRG("")\nEND\nSTOP', 'NEW(PRP)\nTRG("x")\nARG(Content, "")\nEND\nSTOP',
    'NEW(PRP)\nTRG("x")\nSTOP','NEW(PRP)\nTRG("x")\nEND',
    'STOP\nSTOP','STOP trailing','NEW(PRP)\nTRG("x")\nEND\nSTOP\nNEW(PRP)',
    'NEW(PRP)\nTRG("bad\nstring")\nEND\nSTOP',
])
def test_invalid_grammar(output):
    with pytest.raises(ValueError): parse_actions(output,SCHEMA)

def source_row(events):
    return dict(id='doc::sentence-0',document_id='doc',split='train',sentence_index=0,
                target_bounds=[0,4],target_sentence='we propose model .',
                sentences=['we propose model .'],tokens=['we','propose','model','.'],
                target_events=events,n_events=len(events),completion='old tree')

def test_prompt_ignores_gold_and_preserves_metadata():
    template=(ROOT/'task/101202_action_prompt_v1.txt').read_text()
    first=make_row(source_row([event(args=[arg('Content','model',[2])])]),template,SCHEMA)
    second=make_row(source_row([]),template,SCHEMA)
    assert first['prompt']==second['prompt']
    assert '[S1] we propose model .' in first['prompt']
    assert '{ABSTRACT}' not in first['prompt'] and '{TARGET_SENTENCE}' not in first['prompt']
    assert first['target_events']==source_row([event(args=[arg('Content','model',[2])])])['target_events']
    assert '(Trigger ' not in first['compact_tree']

@pytest.mark.parametrize('events',[
    [event(offset=5)], [event(args=[arg('Content','invented',[2])])],
])
def test_source_offsets_and_text_must_match(events):
    with pytest.raises(ValueError): make_row(source_row(events),(ROOT/'task/101202_action_prompt_v1.txt').read_text(),SCHEMA)

def test_token_audit_detects_source_changes(tmp_path,monkeypatch):
    import types
    from build_action_dataset import token_audit
    source=tmp_path/'tokenizer_config.json';source.write_text('{}')
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            source.write_text('{"changed":true}')
            return [1,2] if len(messages)==1 else [1,2,3]
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    with pytest.raises(ValueError,match='Tokenizer.*changed'):
        token_audit({'train':[{'id':'test','prompt':'input','completion':'STOP'}]},tmp_path)

def test_token_audit_checks_assistant_prefix(tmp_path,monkeypatch):
    import types
    from build_action_dataset import token_audit
    (tmp_path/'tokenizer_config.json').write_text('{}')
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return [1,2] if len(messages)==1 else [9,2,3]
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer())))
    with pytest.raises(ValueError,match='prefix'):
        token_audit({'train':[{'id':'test','prompt':'input','completion':'STOP'}]},tmp_path)

def test_cli_build_preserves_split_sorted_ids_and_refuses_overwrite(tmp_path):
    import subprocess,json
    source=tmp_path/'source';source.mkdir()
    for split in ('train','dev','test'):
        r=source_row([event(args=[arg('Content','model',[2])])]);r.update(split=split,id=f'{split}::sentence-0')
        (source/f'{split}.jsonl').write_text(json.dumps(r)+'\n')
    (source/'manifest.json').write_text('{}')
    ids=tmp_path/'ordered.txt';ids.write_text('train::sentence-0\n')
    output=tmp_path/'output'
    command=[sys.executable,str(ROOT/'scripts/build_action_dataset.py'),'--source-dir',str(source),
             '--ordered-ids',str(ids),'--output',str(output),'--sample-count','1']
    first=subprocess.run(command,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    train=json.loads((output/'train.jsonl').read_text())
    sorted_train=json.loads((output/'train-sorted.jsonl').read_text())
    assert train==sorted_train
    assert json.loads((output/'dev.jsonl').read_text())['split']=='dev'
    before={p.name:p.read_bytes() for p in output.iterdir()}
    second=subprocess.run(command,capture_output=True,text=True)
    assert second.returncode!=0 and 'Output exists' in second.stderr
    assert before=={p.name:p.read_bytes() for p in output.iterdir()}
