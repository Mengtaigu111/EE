import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]))
from sampling import audit_sampling_prefix


def records(n,steps,seed):
    import torch
    result=[]
    for epoch in range(4):
        indices=list(range(n)) if epoch==0 else torch.randperm(n,generator=torch.Generator().manual_seed(seed+epoch)).tolist()
        for ordinal,index in enumerate(indices):
            step=epoch*((n+15)//16)+ordinal//16
            if step>=steps:return result
            result.append(dict(epoch=epoch,index=index,id=str(index),microbatch=len(result),optimizer_step_before=step))
    return result


@pytest.mark.parametrize('size,steps',[(28,4),(12828,802),(28,3),(32,3)])
def test_seed52_course_tail_and_partial_epoch(size,steps):
    result=audit_sampling_prefix(records(size,steps,52),[str(i) for i in range(size)],global_step=steps,seed=52)
    assert result['passed']


def test_seed42_second_epoch_rejected():
    with pytest.raises(ValueError,match='permutation'):
        audit_sampling_prefix(records(28,4,42),[str(i) for i in range(28)],global_step=4,seed=52)


def test_cpu_real_native_trainer_seed52_cot_and_tree(tmp_path):
    import json
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM,TrainingArguments
    from train_cot import CoTTrainer,ShuffledSampler
    from sorted_first_sampler import SortedFirstTrainer,IndexedDataset,ObservedCollator

    rows=[]
    for i in range(28):
        tokens=[1]+[2+i%8]*(2+i%4)+[11]
        rows.append(dict(input_ids=tokens,attention_mask=[1]*len(tokens),labels=[-100,-100]+tokens[2:]))
    def collate(batch):
        return {key:torch.tensor([row[key] for row in batch]) for key in ('input_ids','attention_mask','labels')}
    for phase,trainer_class,steps in [('cot',CoTTrainer,2),('tree',SortedFirstTrainer,4)]:
        torch.manual_seed(52)
        config=Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=2,head_dim=8,attention_dropout=0.,use_cache=False)
        config.base_model_tp_plan={};config.base_model_pp_plan={}
        model=Qwen3ForCausalLM(config)
        args=TrainingArguments(output_dir=str(tmp_path/phase),use_cpu=True,per_device_train_batch_size=1,
            gradient_accumulation_steps=16,max_steps=steps,num_train_epochs=2,learning_rate=1e-4,
            lr_scheduler_type='constant',save_strategy='no',eval_strategy='no',logging_strategy='no',
            report_to='none',remove_unused_columns=False,disable_tqdm=True,seed=52,data_seed=52)
        path=tmp_path/f'{phase}-order.jsonl'
        trainer=trainer_class(model=model,args=args,train_dataset=IndexedDataset(rows,[str(i) for i in range(28)]),
            data_collator=ObservedCollator(collate),observation_path=path,sampling_seed=52)
        assert trainer.model_accepts_loss_kwargs
        trainer.train()
        observed=[json.loads(line) for line in path.read_text().splitlines()]
        if phase=='tree':
            audit_sampling_prefix(observed,[str(i) for i in range(28)],global_step=4,seed=52)
        else:
            assert [(r['epoch'],r['index']) for r in observed]==list(ShuffledSampler(28,52))
            assert all(r['optimizer_step_before']==j//16 and r['microbatch']==j for j,r in enumerate(observed))
