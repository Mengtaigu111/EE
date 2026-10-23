"""Numerical and autograd equivalence for the memory-bounded arithmetic path."""
import copy
from pathlib import Path
import sys
import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model

sys.path.insert(0,str(Path(__file__).resolve().parent))
from stable_arithmetic import install_stable_arithmetic,completion_logprobs,ARITHMETIC_CONFIGURATION


def tiny():
    torch.manual_seed(42)
    config=Qwen3Config(vocab_size=48,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=128,attention_dropout=0.,use_sliding_window=False)
    config._attn_implementation='sdpa'
    config.base_model_tp_plan={};config.base_model_pp_plan={}
    base=Qwen3ForCausalLM(config).bfloat16()
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']),adapter_name='policy')
    # Nonzero B is necessary for an informative adapter-gradient comparison.
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'lora_B' in name:p.normal_(std=.01)
    return model


class StableArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_install_keeps_parameter_identity_dtype_and_values(self):
        model=tiny();before={n:(id(p),p.dtype,p.detach().clone(),p.requires_grad) for n,p in model.named_parameters()}
        install_stable_arithmetic(model);install_stable_arithmetic(model)
        for n,p in model.named_parameters():
            identity,dtype,value,grad=before[n]
            self.assertEqual(id(p),identity);self.assertEqual(p.dtype,dtype)
            self.assertEqual(p.requires_grad,grad);self.assertTrue(torch.equal(p,value))
        with torch.no_grad():out=model(input_ids=torch.tensor([[1,2,3]]),use_cache=False).logits
        self.assertEqual(out.dtype,torch.float32)
        self.assertEqual(ARITHMETIC_CONFIGURATION['attention_backend'],'math')

    def test_chunked_probability_and_gradient_match_full_fp32_reference(self):
        model=tiny();install_stable_arithmetic(model)
        reference=copy.deepcopy(model)
        # The second model has identical values and arithmetic; full head scores
        # are the independent reference for chunk/projection checkpointing.
        prompt=torch.tensor([[1,2,3,4],[5,6,7,8]])
        ids=torch.tensor([[9,10,11,12,13],[14,15,16,17,18]])
        mask=torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
        for m in (model,reference):
            m.train();m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});m.enable_input_require_grads()
        score=completion_logprobs(model,prompt,ids,mask,token_chunk=2)
        full=torch.cat([prompt,ids],-1);attn=torch.cat([torch.ones_like(prompt),mask.long()],-1)
        logits=reference(input_ids=full,attention_mask=attn,use_cache=False).logits[:,prompt.shape[1]-1:prompt.shape[1]-1+ids.shape[1]]
        expected=logits.gather(-1,ids.unsqueeze(-1)).squeeze(-1)-logits.logsumexp(-1)
        torch.testing.assert_close(score,expected,atol=2e-6,rtol=2e-6)
        weights=torch.tensor([[1.,-.5,.3,.1,-.2],[.5,.1,-.2,0.,0.]])
        (score*weights*mask).sum().backward();(expected*weights*mask).sum().backward()
        refs=dict(reference.named_parameters());nonzero=0
        for n,p in model.named_parameters():
            q=refs[n]
            if not p.requires_grad:self.assertIsNone(p.grad);continue
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=2e-4)
            nonzero+=int(bool(p.grad.abs().sum()>0))
        self.assertGreater(nonzero,0)

    def test_real_cached_scoring_matches_full_scoring(self):
        model=tiny();install_stable_arithmetic(model);model.eval()
        prompt=torch.tensor([[1,2,3,4]]);ids=torch.tensor([[5,6,7,8,9]]);mask=torch.ones_like(ids,dtype=torch.bool)
        with torch.no_grad():
            full=completion_logprobs(model,prompt,ids,mask,token_chunk=2);past=None;values=[]
            for i in range(ids.shape[1]):
                out=model(input_ids=prompt if i==0 else ids[:,i-1:i],attention_mask=torch.ones((1,prompt.shape[1]+i),dtype=torch.long),past_key_values=past,use_cache=True,logits_to_keep=1)
                v=out.logits[:,-1];values.append(v.gather(-1,ids[:,i:i+1]).squeeze(-1)-v.logsumexp(-1));past=out.past_key_values
            cached=torch.stack(values,-1)
        torch.testing.assert_close(full,cached,atol=2e-5,rtol=2e-5)

    def test_reject_invalid_token_shapes_and_chunks(self):
        model=tiny();install_stable_arithmetic(model)
        with self.assertRaises(ValueError):completion_logprobs(model,torch.tensor([[1,2]]),torch.tensor([[3]]),torch.tensor([[True]]),token_chunk=0)
        with self.assertRaises(ValueError):completion_logprobs(model,torch.tensor([[1,2]]),torch.tensor([[3]]),torch.tensor([[True,False]]))

    def test_reject_unsupported_head_adapter(self):
        model=tiny();model.peft_config['policy'].modules_to_save=['lm_head']
        with self.assertRaises(ValueError):install_stable_arithmetic(model)

    def test_reject_trainable_reference_adapter(self):
        model=tiny();model.add_adapter('reference',copy.deepcopy(model.peft_config['policy']))
        for n,p in model.named_parameters():
            if '.reference.' in n:p.data=p.data.float();p.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'policy LoRA'):install_stable_arithmetic(model)

    def test_reject_storage_dtype_that_misreports_protocol(self):
        model=tiny();model.get_base_model().model.layers[0].input_layernorm.weight.data=model.get_base_model().model.layers[0].input_layernorm.weight.data.float()
        with self.assertRaisesRegex(ValueError,'BF16'):install_stable_arithmetic(model)

    def test_cpu_materialization_preserves_parameters_and_tied_identity(self):
        from stable_arithmetic import materialize_cpu_parameters
        model=tiny()
        model.get_base_model().lm_head.weight=model.get_base_model().model.embed_tokens.weight
        before={n:(id(p),p.data_ptr(),p.detach().clone()) for n,p in model.named_parameters()}
        size=materialize_cpu_parameters(model)
        self.assertEqual(size,sum(p.numel()*p.element_size() for p in model.parameters()))
        for n,p in model.named_parameters():
            self.assertEqual(id(p),before[n][0]);self.assertNotEqual(p.data_ptr(),before[n][1])
            self.assertTrue(torch.equal(p,before[n][2]));self.assertEqual(p.device.type,'cpu')
        self.assertIs(model.get_base_model().lm_head.weight,model.get_base_model().model.embed_tokens.weight)


if __name__=='__main__':unittest.main()
