"""Exercise the actual dual-adapter export/import recovery path on a tiny CPU LM."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,get_peft_model
from grpo_core import initialize_dual_adapters,save_policy,completion_logprobs,activate_adapter
from checkpoint_state import save_checkpoint,load_checkpoint
from recovery_runtime import import_policy
from train_grpo_recoverable import adapter_digest

class TokenizerStub:
    def save_pretrained(self,path):
        Path(path,'tokenizer_config.json').write_text('{}')

class PeftRecoveryTest(unittest.TestCase):
    def test_only_actor_restored_reference_and_next_update_match(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
            attention_dropout=0.0,tie_word_embeddings=True)
        config.base_model_tp_plan={};config.base_model_pp_plan={}
        base=Qwen3ForCausalLM(config)
        pristine=copy.deepcopy(base)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sft=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj'],task_type='CAUSAL_LM'))
            sft.save_pretrained(root/'sft')
            def factory():
                m=initialize_dual_adapters(copy.deepcopy(pristine),root/'sft')
                return m,torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
            model,opt=factory()
            initial_digest=adapter_digest(model,'policy')
            reference_digest=adapter_digest(model,'reference')
            reference={n:p.detach().clone() for n,p in model.named_parameters() if '.reference.' in n}
            prompt=torch.tensor([[1,2]]);ids=torch.tensor([[3,4]]);mask=torch.ones_like(ids,dtype=torch.bool)
            def update(m,o):
                activate_adapter(m,'policy');(-completion_logprobs(m,prompt,ids,mask).mean()).backward();o.step();o.zero_grad()
            update(model,opt)
            self.assertNotEqual(adapter_digest(model,'policy'),initial_digest)
            self.assertEqual(adapter_digest(model,'reference'),reference_digest)
            point=save_checkpoint(root/'cp','saved',model,opt,
                {'step':1,'initial_policy_parameter_sha256':initial_digest},
                export_policy=lambda m,p:save_policy(m,TokenizerStub(),p))
            restored,newopt=factory()
            payload=load_checkpoint(point,restored,newopt,load_policy=lambda m,p:import_policy(m,p/'policy'))
            self.assertEqual(payload['step'],1)
            self.assertEqual(payload['initial_policy_parameter_sha256'],initial_digest)
            self.assertEqual(adapter_digest(restored,'policy'),adapter_digest(model,'policy'))
            update(model,opt);update(restored,newopt)
            for (n,p),(n2,q) in zip(model.named_parameters(),restored.named_parameters()):
                self.assertEqual(n,n2)
                torch.testing.assert_close(p,q,atol=0,rtol=0)
                if n in reference:torch.testing.assert_close(q,reference[n],atol=0,rtol=0)
if __name__=='__main__':unittest.main()
