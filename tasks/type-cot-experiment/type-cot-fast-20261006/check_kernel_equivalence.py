"""GPU gate: loss and LoRA gradient equivalence, token weighting, native inference."""
import argparse
import json
import os
from pathlib import Path
import sys

from kernel_support import apply_kernel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if 'TMUX' not in os.environ:
        raise ValueError('GPU equivalence check requires tmux')
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from peft import LoraConfig,get_peft_model
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA bf16 required')
    torch.manual_seed(42)
    config=Qwen3Config(vocab_size=257,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,
        attention_dropout=0.,use_cache=False,eos_token_id=256,pad_token_id=0)
    config.base_model_tp_plan={};config.base_model_pp_plan={}
    model=Qwen3ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16)
    model=get_peft_model(model,LoraConfig(task_type='CAUSAL_LM',r=4,lora_alpha=8,
        lora_dropout=0.,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    # Nonzero B ensures gradients through both LoRA factors are checked.
    with torch.no_grad():
        for name,param in model.named_parameters():
            if 'lora_B' in name:
                param.normal_(mean=0.,std=.01)
    rows=[]
    for length,prompt in [(9,3),(15,7)]:
        inputs=torch.randint(1,256,(1,length),device='cuda')
        labels=inputs.clone();labels[:,:prompt]=-100
        rows.append(dict(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels))
    denominator=sum(int((row['labels']!=-100).sum()) for row in rows)
    weights={name:param.detach().clone() for name,param in model.named_parameters()}

    def loss_and_grads():
        model.train();model.zero_grad(set_to_none=True);total=0.
        for row in rows:
            loss=model(**row,num_items_in_batch=torch.tensor(denominator,device='cuda')).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            total+=float(loss.detach());loss.backward()
        torch.cuda.synchronize()
        grads={name:param.grad.detach().float().clone() for name,param in model.named_parameters()
               if param.requires_grad and param.grad is not None}
        return total,grads

    native_loss,native_grads=loss_and_grads()
    model.eval()
    with torch.no_grad():
        native_logits=[model(input_ids=row['input_ids'],attention_mask=row['attention_mask']).logits.detach().clone()
                       for row in rows]
        native_generation=model.generate(input_ids=rows[0]['input_ids'],max_new_tokens=5,do_sample=False,use_cache=False)
    metadata=apply_kernel('liger')
    liger_loss,liger_grads=loss_and_grads()
    if set(native_grads)!=set(liger_grads):raise ValueError('Trainable gradient inventory differs')
    torch.testing.assert_close(torch.tensor(native_loss),torch.tensor(liger_loss),rtol=.005,atol=1e-5)
    gradients=[]
    for name in native_grads:
        reference=native_grads[name];actual=liger_grads[name]
        error=float((reference-actual).norm());norm=float(reference.norm())
        if error>.01*norm+1e-6:
            raise ValueError(f'Gradient norm mismatch {name}: error={error},reference={norm}')
        gradients.append(dict(name=name,reference_norm=norm,error_norm=error,relative_error=error/max(norm,1e-12)))
    if any(not torch.equal(weights[name],param) for name,param in model.named_parameters()):
        raise ValueError('Equivalence check changed weights')
    model.eval()
    with torch.no_grad():
        for row,reference in zip(rows,native_logits):
            actual=model(input_ids=row['input_ids'],attention_mask=row['attention_mask']).logits
            if not torch.equal(reference,actual):raise ValueError('labels=None eval logits changed')
        generation=model.generate(input_ids=rows[0]['input_ids'],max_new_tokens=5,do_sample=False,use_cache=False)
        if not torch.equal(native_generation,generation):raise ValueError('Greedy generation changed')
    report=dict(status='passed',dtype='bf16',dropout=0,num_items_in_batch=denominator,
        native_loss=native_loss,liger_loss=liger_loss,trainable_gradients=gradients,
        labels_none_eval_logits_exact=True,greedy_generation_exact=True,kernel_metadata=metadata)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
