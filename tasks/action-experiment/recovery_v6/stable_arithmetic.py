"""Consistent FP32 Qwen3 arithmetic with unchanged BF16 parameter storage.

Vanilla LoRA only. Per-instance forwards preserve parameter identity/dtypes,
including recovery signatures and AdamW state. No global nn.Linear patching.
"""
import copy
from types import MethodType

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.utils.checkpoint import checkpoint

ARITHMETIC_CONFIGURATION=dict(
    protocol='fp32-body-bf16-storage-v1',base_parameter_storage='bfloat16',
    adapter_parameter_storage='float32',activations='float32',
    linear_arithmetic='float32',attention_backend='math',allow_tf32=False,
    completion_projection='checkpointed_fp32_chunks',token_chunk=16)


def materialize_cpu_parameters(model):
    """Read mmap-backed weights fully before acquiring scarce GPU memory.

    This retains Parameter objects, dtypes, ties and values. Only their CPU
    storage is materialized; neither optimizer state nor CUDA is touched.
    """
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('Materialization must precede GPU transfer')
    size=0
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.data=parameter.detach().clone(memory_format=torch.contiguous_format)
            size+=parameter.numel()*parameter.element_size()
    return size


def initialize_dual_adapters_cpu(base,adapter_path):
    """Prepare both adapters without PEFT's implicit temporary CUDA staging."""
    from peft import PeftModel
    from grpo_core import activate_adapter
    model=PeftModel.from_pretrained(base,str(adapter_path),adapter_name='policy',
        is_trainable=True,torch_device='cpu')
    if model.peft_config['policy'].modules_to_save:
        raise ValueError('GRPO requires pure LoRA without modules_to_save')
    model.load_adapter(str(adapter_path),adapter_name='reference',
        is_trainable=False,torch_device='cpu')
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    activate_adapter(model,'policy')
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('Adapter preparation must remain on CPU')
    print('Materializing CPU weights before GPU admission',flush=True)
    materialize_cpu_parameters(model)
    return model


def _linear_forward(self,inputs):
    # The cast must stay inside decoder/head checkpoint regions, so backward
    # recomputes each frozen FP32 weight instead of retaining a model-wide copy.
    with torch.autocast(device_type=inputs.device.type,enabled=False):
        return F.linear(inputs.float(),self.weight.float(),
            self.bias.float() if self.bias is not None else None)


def _embedding_forward(self,*args,**kwargs):
    return self._stable_original_forward(*args,**kwargs).float()


def _attention_forward(self,*args,**kwargs):
    # This context is inside attention.forward, so gradient-checkpoint backward
    # recomputation uses the same backend as the original training forward.
    with sdpa_kernel(SDPBackend.MATH):
        return self._stable_original_forward(*args,**kwargs)


def install_stable_arithmetic(model):
    """Install validated arithmetic without altering any parameter or RNG state."""
    existing=getattr(model,'_stable_arithmetic_configuration',None)
    if existing is not None:
        if existing!=ARITHMETIC_CONFIGURATION:raise ValueError('Arithmetic protocol already differs')
        return copy.deepcopy(existing)
    base=model.get_base_model()
    if base.config.model_type!='qwen3':raise ValueError('Stable arithmetic supports Qwen3 only')
    for config in model.peft_config.values():
        if (getattr(config,'is_prompt_learning',False) or config.modules_to_save
                or getattr(config,'use_dora',False) or getattr(config,'alora_invocation_tokens',None)):
            raise ValueError('Stable arithmetic requires vanilla LoRA without saved modules')
    embedding=model.get_input_embeddings();head=model.get_output_embeddings()
    if not isinstance(embedding,torch.nn.Embedding) or not isinstance(head,torch.nn.Linear):
        raise ValueError('Embedding/head adapters are unsupported')
    for name,parameter in model.named_parameters():
        if 'lora_' in name:
            if parameter.dtype!=torch.float32:raise ValueError('LoRA parameters must stay FP32')
            if parameter.requires_grad and '.policy.' not in name:
                raise ValueError('Only policy LoRA may be trainable')
        elif parameter.requires_grad:raise ValueError('Only policy LoRA may be trainable')
        elif parameter.dtype!=torch.bfloat16:raise ValueError('Base parameters must retain BF16 storage')
    if any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules()):
        raise ValueError('Probability consistency requires dropout disabled')
    torch.backends.cuda.matmul.allow_tf32=False
    for module in model.modules():
        if isinstance(module,torch.nn.Linear):
            module.forward=MethodType(_linear_forward,module)
    embedding._stable_original_forward=embedding.forward
    embedding.forward=MethodType(_embedding_forward,embedding)
    for layer in base.model.layers:
        attention=layer.self_attn
        attention._stable_original_forward=attention.forward
        attention.forward=MethodType(_attention_forward,attention)
    model._stable_arithmetic_configuration=copy.deepcopy(ARITHMETIC_CONFIGURATION)
    return copy.deepcopy(ARITHMETIC_CONFIGURATION)


def completion_logprobs(model,prompt_ids,ids,mask,token_chunk=16):
    """Score only completion positions, checkpointing vocab projection+logsumexp.

    Vanilla LoRA is already injected into the body modules; bypassing the causal
    LM wrapper avoids projecting every prompt position into the full vocabulary.
    """
    if (token_chunk<1 or prompt_ids.ndim!=2 or ids.ndim!=2
            or mask.shape!=ids.shape or prompt_ids.shape[0]!=ids.shape[0]
            or prompt_ids.shape[1]<1 or ids.shape[1]<1):
        raise ValueError('Invalid prompt/completion shapes or token chunk')
    if getattr(model,'_stable_arithmetic_configuration',None)!=ARITHMETIC_CONFIGURATION:
        raise ValueError('Install stable arithmetic before scoring')
    inputs=torch.cat([prompt_ids,ids],-1)
    attention=torch.cat([torch.ones_like(prompt_ids),mask.long()],-1)
    body=model.get_base_model().model
    hidden=body(input_ids=inputs,attention_mask=attention,use_cache=False,return_dict=True).last_hidden_state
    start=prompt_ids.shape[1]-1
    hidden=hidden[:,start:start+ids.shape[1]]
    head=model.get_output_embeddings()

    def score(chunk,labels):
        values=head(chunk)
        return values.gather(-1,labels.unsqueeze(-1)).squeeze(-1)-values.logsumexp(-1)

    values=[]
    for offset in range(0,ids.shape[1],token_chunk):
        chunk=hidden[:,offset:offset+token_chunk];labels=ids[:,offset:offset+token_chunk]
        if torch.is_grad_enabled() and chunk.requires_grad:
            values.append(checkpoint(score,chunk,labels,use_reentrant=False,preserve_rng_state=False))
        else:values.append(score(chunk,labels))
    return torch.cat(values,-1)
