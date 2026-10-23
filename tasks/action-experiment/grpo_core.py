"""Explicit original GRPO objective and frozen SFT-reference LoRA handling.

Objective: https://huggingface.co/docs/trl/grpo_trainer#loss-types
Adapter activation: https://huggingface.co/docs/peft/package_reference/peft_model
"""
import torch


def group_advantages(rewards, epsilon=1e-4):
    if rewards.ndim != 1 or rewards.numel() < 2 or not torch.isfinite(rewards).all():
        raise ValueError('Rewards must be a finite group of at least two')
    return (rewards-rewards.mean())/(rewards.std(correction=1)+epsilon)


def grpo_loss(current, old, reference, mask, advantages, clip=.2, beta=.01, field_mask=None):
    if not (current.shape == old.shape == reference.shape == mask.shape):
        raise ValueError('Log probabilities and mask must have equal shapes')
    if advantages.shape != (current.shape[0],) or (mask.sum(-1)==0).any():
        raise ValueError('Each completion needs an advantage and at least one token')
    ratio=(current-old.detach()).exp()
    delta=reference.detach()-current
    kl=delta.exp()-delta-1
    a=advantages.detach().unsqueeze(-1)
    surrogate=torch.minimum(ratio*a,ratio.clamp(1-clip,1+clip)*a)
    if field_mask is not None:
        if field_mask.shape != current.shape or not torch.isfinite(field_mask).all() or not ((field_mask==0)|(field_mask==1)).all():
            raise ValueError('Field mask must be binary, finite and match token log probabilities')
        surrogate=surrogate*field_mask.detach()
    loss=((beta*kl-surrogate)*mask).sum(-1)/mask.sum(-1)
    if not torch.isfinite(loss).all():
        raise ValueError('Nonfinite GRPO objective')
    return loss.mean(),{'kl':float(((kl.detach()*mask).sum(-1)/mask.sum(-1)).mean()),
                         'clip_fraction':float((((ratio.detach()-1).abs()>clip)*mask).sum()/mask.sum())}


def completion_mask(ids, eos_ids):
    eos=torch.zeros_like(ids,dtype=torch.bool)
    for token in eos_ids:
        eos |= ids.eq(token)
    # Include the first EOS, exclude subsequent padding (including pad == EOS).
    return (eos.cumsum(-1)-eos.long()).eq(0)


def activate_adapter(model, name):
    if name not in ('policy','reference'):
        raise ValueError('Only policy/reference adapters are allowed')
    model.set_adapter(name)
    for parameter_name,parameter in model.named_parameters():
        parameter.requires_grad_(name=='policy' and '.policy.' in parameter_name and 'lora_' in parameter_name)


def initialize_dual_adapters(base, adapter_path):
    from peft import PeftModel
    model=PeftModel.from_pretrained(base,str(adapter_path),adapter_name='policy',is_trainable=True)
    if model.peft_config['policy'].modules_to_save:
        raise ValueError('GRPO currently requires pure LoRA without modules_to_save')
    model.load_adapter(str(adapter_path),adapter_name='reference',is_trainable=False)
    # Eval/train mode must not change either probability distribution.
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):
            module.p=0.
    activate_adapter(model,'policy')
    return model


def completion_logprobs(model, prompt_ids, ids, mask, token_chunk=16):
    if prompt_ids.shape[0]!=ids.shape[0] or ids.shape!=mask.shape or prompt_ids.shape[1]<1:
        raise ValueError('Invalid prompt/completion shapes')
    inputs=torch.cat([prompt_ids,ids],dim=-1)
    attention=torch.cat([torch.ones_like(prompt_ids),mask.long()],dim=-1)
    logits=model(input_ids=inputs,attention_mask=attention,use_cache=False).logits
    start=prompt_ids.shape[1]-1
    logits=logits[:,start:start+ids.shape[1]]
    values=[]
    for offset in range(0,ids.shape[1],token_chunk):
        chunk=logits[:,offset:offset+token_chunk].float()
        labels=ids[:,offset:offset+token_chunk]
        values.append(chunk.gather(-1,labels.unsqueeze(-1)).squeeze(-1)-torch.logsumexp(chunk,dim=-1))
    return torch.cat(values,dim=-1)


def save_policy(model,tokenizer,path):
    """PEFT named exports are nested; return the actual loadable adapter directory."""
    from pathlib import Path
    activate_adapter(model,'policy')
    model.save_pretrained(str(path),selected_adapters=['policy'],safe_serialization=True)
    actual=Path(path)/'policy'
    tokenizer.save_pretrained(str(actual))
    return actual


def behavior_logprobs(generation_output,prompt_length,mask):
    """Probabilities from the actual cached sampling pass, without retokenization.

    The caller must use temperature 1, no top-k/p filtering and no processors.
    Consume the returned scores immediately, then delete the generation output.
    """
    ids=generation_output.sequences[:,prompt_length:]
    scores=generation_output.scores
    if ids.shape!=mask.shape or len(scores)!=ids.shape[1]:
        raise ValueError('Generation scores, sampled tokens and mask shape differ')
    values=[]
    for position,score in enumerate(scores):
        logits=score.float()
        values.append(logits.gather(-1,ids[:,position:position+1]).squeeze(-1)-torch.logsumexp(logits,-1))
    return torch.stack(values,dim=-1).detach()
