import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]))
from sampling import audit_sampling_prefix, validate_microbatch, observation_records


def records_for(n, microbatch, steps):
    import torch
    rows=[]; batches=0
    for epoch in range(4):
        indices=list(range(n)) if epoch==0 else torch.randperm(n,generator=torch.Generator().manual_seed(42+epoch)).tolist()
        for start in range(0,n,microbatch):
            update=epoch*((n+15)//16)+start//16
            if update>=steps:
                return rows
            meta=[dict(epoch=epoch,index=i,id=str(i)) for i in indices[start:start+microbatch]]
            rows.extend(observation_records(meta,batches,update));batches+=1
    return rows


@pytest.mark.parametrize('batch',[1,2])
@pytest.mark.parametrize('n,steps',[(32,4),(28,4),(12828,802),(28,3)])
def test_sample_sequence_update_groups_and_tail(batch,n,steps):
    result=audit_sampling_prefix(records_for(n,batch,steps),[str(i) for i in range(n)],global_step=steps,microbatch=batch)
    assert result['passed']
    assert result['observed_rows']==sum(result['epoch_counts'])


@pytest.mark.parametrize('field,value',[('ordinal_in_batch',1),('microbatch',1),('optimizer_step_before',1),('index',3),('id','bad')])
def test_reject_corrupt_observed_rows(field,value):
    rows=records_for(28,2,2);rows[0][field]=value
    with pytest.raises(ValueError):
        audit_sampling_prefix(rows,[str(i) for i in range(28)],global_step=2,microbatch=2)


@pytest.mark.parametrize('value',[0,3,True,1.5,'2'])
def test_invalid_microbatch(value):
    with pytest.raises(ValueError):validate_microbatch(value)


def test_metadata_not_mutated_and_rows_share_actual_microbatch():
    meta=[dict(epoch=0,index=0,id='a'),dict(epoch=0,index=1,id='b')]
    rows=observation_records(meta,4,0)
    assert [r['ordinal_in_batch'] for r in rows]==[0,1]
    assert [r['microbatch'] for r in rows]==[4,4]
    assert 'ordinal_in_batch' not in meta[0]


def test_cpu_real_trainer_batch2_matches_batch1_token_normalization_and_tail(tmp_path):
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments
    from sorted_first_sampler import SortedFirstTrainer, IndexedDataset, ObservedCollator
    from sampling import MicrobatchObservationMixin

    class ProbeTrainer(MicrobatchObservationMixin, SortedFirstTrainer):
        pass

    def collate(rows):
        length=max(len(row['input_ids']) for row in rows)
        return {key:torch.tensor([row[key]+[pad]*(length-len(row[key])) for row in rows])
                for key,pad in [('input_ids',0),('labels',-100),('attention_mask',0)]}

    rows=[]
    for i in range(28):
        tokens=[1]+[2+i%8]*(2+i%4)+[11]
        rows.append(dict(input_ids=tokens,attention_mask=[1]*len(tokens),labels=[-100,-100]+tokens[2:]))
    states=[]
    for batch in (1,2):
        torch.manual_seed(42)
        config=Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            attention_dropout=0.,use_cache=False)
        config.base_model_tp_plan={};config.base_model_pp_plan={}
        model=Qwen3ForCausalLM(config)
        args=TrainingArguments(output_dir=str(tmp_path/f'b{batch}'),use_cpu=True,
            per_device_train_batch_size=batch,gradient_accumulation_steps=16//batch,
            max_steps=4,num_train_epochs=2,learning_rate=1e-4,lr_scheduler_type='constant',
            save_strategy='no',eval_strategy='no',logging_strategy='no',report_to='none',
            remove_unused_columns=False,disable_tqdm=True,seed=42,data_seed=42)
        trainer=ProbeTrainer(model=model,args=args,train_dataset=IndexedDataset(rows,[str(i) for i in range(28)]),
            data_collator=ObservedCollator(collate),observation_path=tmp_path/f'order{batch}.jsonl')
        assert trainer.model_accepts_loss_kwargs, 'Token denominator must reach Qwen3 loss'
        trainer.train()
        import json
        records=[json.loads(line) for line in (tmp_path/f'order{batch}.jsonl').read_text().splitlines()]
        audit_sampling_prefix(records,[str(i) for i in range(28)],global_step=4,microbatch=batch)
        states.append({k:v.detach().clone() for k,v in model.state_dict().items()})
    assert states[0].keys()==states[1].keys()
    for key in states[0]:
        torch.testing.assert_close(states[0][key],states[1][key],rtol=2e-5,atol=2e-6,msg=key)
