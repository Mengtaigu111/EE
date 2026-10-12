import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tasks/action-experiment'))
sys.path.insert(0,str(ROOT/'scripts'))
from action_evaluation import reward_action,score_predictions


def e(kind='PRP',trigger='propose',args=None):
    return dict(event_type=kind,trigger={'tokens':trigger.split(),'text':trigger,'offsets':[1]},arguments=args or [])

def row(events):
    return dict(id='d::sentence-0',document_id='d',sentence_index=0,split='dev',target_events=events,
                target_sentence='we propose model .',sentences=['we propose model .'],
                tokens=['we','propose','model','.'],target_bounds=[0,4],prompt='input')


def test_presence_same_trigger_multiple_types_perfect():
    raw='NEW(PRP)\nTRG("propose")\nEND\nNEW(MDS)\nTRG("propose")\nEND\nSTOP'
    result=reward_action(raw,row([e(),e('MDS')]))
    assert result['TC_f1']==result['TI_f1']==result['type_count_f1']==result['reward']==1


def test_duplicate_presence_counts_reward_separately():
    raw='NEW(PRP)\nTRG("propose")\nEND\nNEW(PRP)\nTRG("propose")\nEND\nSTOP'
    result=reward_action(raw,row([e()]))
    assert result['TC_f1']==1 and result['type_count_f1']==pytest.approx(2/3)
    assert result['reward']==pytest.approx(.9+.1*2/3)


def test_empty_and_bad_empty_outputs():
    assert reward_action('STOP',row([]))['reward']==1
    assert reward_action('STOP',row([e()]))['reward']==0
    assert reward_action('garbage',row([]))['reward']==0
    assert reward_action('NEW(PRP)\nTRG("x")\nEND\nSTOP',row([]))['reward']==0


def test_unlocatable_trigger_retained_as_false_positive():
    raw='NEW(PRP)\nTRG("invented")\nEND\nSTOP'
    result=reward_action(raw,row([]))
    assert result['predicted_events']==1 and result['reward']==0
    scored=score_predictions([row([])],[dict(id='d::sentence-0',raw_output=raw)])
    assert scored['counts']['TC']['fp']==1


def test_parse_failure_is_not_false_empty_in_corpus():
    scored=score_predictions([row([])],[dict(id='d::sentence-0',raw_output='garbage')])
    assert scored['diagnostics']['parse_errors']==1 and scored['counts']['TC']['fp']==1


def test_preserve_wrong_unlocatable_arguments():
    raw='NEW(PRP)\nTRG("propose")\nARG(Content, "invented")\nEND\nSTOP'
    scored=score_predictions([row([e()])],[dict(id='d::sentence-0',raw_output=raw)])
    assert scored['metrics']['TC_f1']==1 and scored['counts']['AC']['fp']==1


def test_argument_reward_is_intentionally_deferred():
    gold=e(args=[dict(argument_type='Content',tokens=['model'],text='model',offsets=[2])])
    assert reward_action('NEW(PRP)\nTRG("propose")\nEND\nSTOP',row([gold]))['reward']==1


def test_prediction_coverage_must_be_exact():
    with pytest.raises(ValueError):score_predictions([row([])],[])
    with pytest.raises(ValueError):score_predictions([row([])],[dict(id='wrong',raw_output='STOP')])


def test_resident_evaluation_is_greedy_keeps_raw_tokens_and_restores_modes(tmp_path):
    import types,json,torch
    from action_evaluation import evaluate_model
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
            self.generation_config=types.SimpleNamespace(eos_token_id=[2])
        def generate(self,**kwargs):
            assert kwargs['use_model_defaults'] is False
            assert kwargs['generation_config'].do_sample is False
            assert kwargs['generation_config'].max_new_tokens==768
            ids=kwargs['input_ids']
            return torch.cat([ids,torch.tensor([[3,2,2]]).repeat(ids.shape[0],1)],dim=1)
    class Tokenizer:
        padding_side='right';pad_token_id=2;eos_token_id=2;bos_token_id=1
        def apply_chat_template(self,*args,**kwargs):return 'input'
        def __call__(self,prompts,**kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids':torch.tensor([[1,4]]).repeat(len(prompts),1),
                    'attention_mask':torch.ones(len(prompts),2,dtype=torch.long)}
        def decode(self,*args,**kwargs):return 'STOP'
    model=Model();tokenizer=Tokenizer();model.train()
    result=evaluate_model(model,tokenizer,[row([])],tmp_path/'evaluation')
    prediction=json.loads((tmp_path/'evaluation/predictions.jsonl').read_text())
    assert prediction['completion_token_ids']==[3,2] and prediction['truncated'] is False
    assert result['diagnostics']['parse_errors']==0 and result['counts']['TC']['fp']==0
    assert model.training is True and tokenizer.padding_side=='right'
