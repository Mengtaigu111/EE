import importlib.util
import sys
from pathlib import Path
import tempfile
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def core():
    assert (ROOT/'grpo_core.py').exists(), 'GRPO implementation is missing'
    import grpo_core
    return grpo_core

def test_group_advantages_ties_and_sample_std():
    c=core()
    assert torch.equal(c.group_advantages(torch.tensor([1.,1.,1.,1.])),torch.zeros(4))
    a=c.group_advantages(torch.tensor([0.,1.,2.,3.]))
    assert torch.allclose(a.mean(),torch.tensor(0.))
    assert abs(a.std().item()-1)<1e-3

def test_sequence_normalization_excludes_padding_and_has_gradient():
    c=core()
    x=torch.zeros(2,3,requires_grad=True)
    old=torch.zeros_like(x); ref=torch.zeros_like(x)
    mask=torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool)
    loss,stats=c.grpo_loss(x,old,ref,mask,torch.tensor([1.,-1.]))
    loss.backward()
    assert torch.allclose(x.grad,torch.tensor([[-.5,0,0],[1/6,1/6,1/6]]))
    assert stats['kl']==0

def test_clipping_and_reference_penalty():
    c=core()
    x=torch.tensor([[1.]],requires_grad=True)
    loss,_=c.grpo_loss(x,torch.zeros_like(x),x.detach(),torch.ones_like(x,dtype=torch.bool),torch.ones(1),beta=0)
    assert abs(loss.item()+1.2)<1e-6
    loss.backward(); assert x.grad.item()==0
    loss,stats=c.grpo_loss(torch.zeros(1,1),torch.zeros(1,1),torch.ones(1,1),torch.ones(1,1,dtype=torch.bool),torch.zeros(1))
    assert loss.item()>0 and stats['kl']>0

def test_mask_keeps_first_eos_excludes_following_pad_even_if_pad_is_eos():
    c=core()
    m=c.completion_mask(torch.tensor([[7,2,2,2],[7,8,9,10]]),[2])
    assert m.tolist()==[[True,True,False,False],[True,True,True,True]]

def test_frozen_real_peft_reference_survives_switch_and_update():
    c=core()
    from transformers import GPT2Config,GPT2LMHeadModel
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(42)
    cfg=GPT2Config(vocab_size=32,n_positions=24,n_embd=16,n_layer=1,n_head=2,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)
    base=GPT2LMHeadModel(cfg)
    state={k:v.clone() for k,v in base.state_dict().items()}
    adapter=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,lora_alpha=4,lora_dropout=.1,target_modules=['c_attn']))
    with torch.no_grad():
        for n,p in adapter.named_parameters():
            if 'lora_B' in n: p.normal_(0,.1)
    with tempfile.TemporaryDirectory() as tmp:
        adapter.save_pretrained(tmp)
        base=GPT2LMHeadModel(cfg);base.load_state_dict(state)
        model=c.initialize_dual_adapters(base,tmp)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        prompt=torch.tensor([[3,4]]);answer=torch.tensor([[5,6,2]]);mask=torch.ones_like(answer,dtype=torch.bool)
        model.eval();c.activate_adapter(model,'policy')
        before=c.completion_logprobs(model,prompt,answer,mask).detach()
        c.activate_adapter(model,'reference')
        ref_before=c.completion_logprobs(model,prompt,answer,mask).detach()
        assert torch.allclose(before,ref_before)
        c.activate_adapter(model,'policy')
        opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
        model.train()
        (-c.completion_logprobs(model,prompt,answer,mask).mean()).backward();opt.step()
        c.activate_adapter(model,'reference')
        assert not any(p.requires_grad for p in model.parameters())
        ref_after=c.completion_logprobs(model,prompt,answer,mask).detach()
        assert all(p.grad is None for n,p in model.named_parameters() if '.reference.' in n)
        assert torch.equal(ref_before,ref_after)
        c.activate_adapter(model,'policy')
        after=c.completion_logprobs(model,prompt,answer,mask).detach()
        assert not torch.allclose(before,after)
        assert all('.policy.' in n for n,p in model.named_parameters() if p.requires_grad)

def test_exported_named_policy_adapter_reloads_same_distribution():
    c=core()
    from transformers import GPT2Config,GPT2LMHeadModel
    from peft import LoraConfig,get_peft_model,PeftModel
    torch.manual_seed(9)
    cfg=GPT2Config(vocab_size=24,n_positions=16,n_embd=8,n_layer=1,n_head=1,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)
    base=GPT2LMHeadModel(cfg);state={k:v.clone() for k,v in base.state_dict().items()}
    initial=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,target_modules=['c_attn']))
    class Tokenizer:
        def save_pretrained(self,path):pass
    with tempfile.TemporaryDirectory() as tmp:
        initial.save_pretrained(Path(tmp)/'initial')
        base=GPT2LMHeadModel(cfg);base.load_state_dict(state)
        model=c.initialize_dual_adapters(base,Path(tmp)/'initial')
        with torch.no_grad():
            for n,p in model.named_parameters():
                if 'lora_B.policy' in n:p.normal_(0,.2)
        model.eval();prompt=torch.tensor([[3,4]]);answer=torch.tensor([[5,2]]);mask=torch.ones_like(answer,dtype=torch.bool)
        before=c.completion_logprobs(model,prompt,answer,mask).detach()
        exported=c.save_policy(model,Tokenizer(),Path(tmp)/'checkpoint')
        assert (exported/'adapter_model.safetensors').is_file()
        base=GPT2LMHeadModel(cfg);base.load_state_dict(state)
        reloaded=PeftModel.from_pretrained(base,exported).eval()
        assert torch.allclose(before,c.completion_logprobs(reloaded,prompt,answer,mask))

def test_sft_source_requires_selected_completed_non_smoke_adapter():
    core()
    from train_grpo import check_sft_source
    import json,hashlib
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);adapter=root/'best-adapter';adapter.mkdir()
        weights=adapter/'adapter_model.safetensors';weights.write_bytes(b'weights')
        config=adapter/'adapter_config.json';config.write_text('{}')
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        read=lambda p:json.loads(Path(p).read_text())
        metrics={'stage':'action_sft','status':'completed','smoke':False,'best_adapter_sha256':sha(weights),'best_adapter_config_sha256':sha(config)}
        path=root/'train_metrics.json';path.write_text(json.dumps(metrics))
        assert check_sft_source(adapter,read,sha,False)==path
        metrics['smoke']=True;path.write_text(json.dumps(metrics))
        with pytest.raises(ValueError,match='smoke'):check_sft_source(adapter,read,sha,False)
        metrics['smoke']=False;metrics['best_adapter_sha256']='bad';path.write_text(json.dumps(metrics))
        with pytest.raises(ValueError,match='selected'):check_sft_source(adapter,read,sha,False)

def test_real_g4_grpo_update_with_checkpointing_keeps_sft_reference():
    c=core()
    from transformers import GPT2Config,GPT2LMHeadModel
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(12)
    cfg=GPT2Config(vocab_size=24,n_positions=16,n_embd=8,n_layer=1,n_head=1,resid_pdrop=0,embd_pdrop=0,attn_pdrop=0)
    base=GPT2LMHeadModel(cfg);state={k:v.clone() for k,v in base.state_dict().items()}
    initial=get_peft_model(base,LoraConfig(task_type='CAUSAL_LM',r=2,target_modules=['c_attn']))
    with tempfile.TemporaryDirectory() as tmp:
        initial.save_pretrained(tmp)
        base=GPT2LMHeadModel(cfg);base.load_state_dict(state)
        model=c.initialize_dual_adapters(base,tmp)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        prompt=torch.tensor([[3,4]]);answers=[torch.tensor([[i,2]]) for i in [5,6,7,8]]
        masks=[torch.ones_like(a,dtype=torch.bool) for a in answers]
        model.eval();c.activate_adapter(model,'policy')
        old=[c.completion_logprobs(model,prompt,a,m).detach() for a,m in zip(answers,masks)]
        c.activate_adapter(model,'reference')
        ref=[c.completion_logprobs(model,prompt,a,m).detach() for a,m in zip(answers,masks)]
        assert all(torch.equal(o,r) for o,r in zip(old,ref))
        c.activate_adapter(model,'policy');model.train()
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.2)
        advantages=c.group_advantages(torch.tensor([0.,.2,.8,1.]))
        for i,(answer,mask) in enumerate(zip(answers,masks)):
            current=c.completion_logprobs(model,prompt,answer,mask)
            loss,_=c.grpo_loss(current,old[i],ref[i],mask,advantages[i:i+1])
            (loss/4).backward()
        optimizer.step()
        c.activate_adapter(model,'reference');model.eval()
        assert all(torch.equal(r,c.completion_logprobs(model,prompt,a,m).detach()) for r,a,m in zip(ref,answers,masks))
        c.activate_adapter(model,'policy')
        assert any(not torch.allclose(o,c.completion_logprobs(model,prompt,a,m)) for o,a,m in zip(old,answers,masks))

def test_grpo_generate_explicitly_disables_model_generation_defaults():
    # Guard the production call: a clean config alone is insufficient in HF 4.52.3.
    import ast
    tree=ast.parse((ROOT/'train_grpo.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='generate']
    assert calls, 'Missing GRPO sampling call'
    for call in calls:
        assert any(k.arg is None and isinstance(k.value,ast.Call) and isinstance(k.value.func,ast.Name)
                   and k.value.func.id=='generation_arguments' for k in call.keywords), 'Sampling must disable model-default backfill'

def test_behavior_logprobs_use_actual_generation_scores_and_sampled_tokens():
    c=core()
    from types import SimpleNamespace
    generated=SimpleNamespace(sequences=torch.tensor([[7,8,2,1]]),
        scores=(torch.tensor([[0.,1.,2.]]),torch.tensor([[3.,2.,1.]])))
    assert hasattr(c,'behavior_logprobs'), 'Actual sampling probability extraction is missing'
    mask=torch.tensor([[True,True]])
    actual=c.behavior_logprobs(generated,2,mask)
    expected=torch.stack([generated.scores[0].log_softmax(-1)[0,2],
                          generated.scores[1].log_softmax(-1)[0,1]]).unsqueeze(0)
    assert torch.allclose(actual,expected)
    assert not actual.requires_grad
    with pytest.raises(ValueError,match='shape'):
        c.behavior_logprobs(generated,2,torch.ones(1,3,dtype=torch.bool))
