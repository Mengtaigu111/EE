import json
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tasks/action-experiment'))
from action_common import MAX_NEW_TOKENS,MAX_LENGTH,verify_locks,sha256,prepare_run


def test_generation_cap():
    assert MAX_NEW_TOKENS==768 and MAX_LENGTH==4096
    from train_sft import parser
    from action_evaluation import evaluate_model
    args=parser().parse_args(['--output','/tmp/new-action-cap-run'])
    assert args.max_new_tokens==MAX_NEW_TOKENS
    assert evaluate_model.__kwdefaults__['max_new_tokens']==MAX_NEW_TOKENS


def test_content_lock_rejects_replacement(tmp_path):
    file=tmp_path/'a';file.write_text('original')
    lock={str(file):sha256(file)};verify_locks(lock)
    file.write_text('changed')
    with pytest.raises(ValueError):verify_locks(lock)


def test_prepare_refuses_existing_directory_without_touching_it(tmp_path):
    output=tmp_path/'run';output.mkdir();(output/'note').write_text('keep')
    with pytest.raises((ValueError,FileExistsError)):
        prepare_run(output,'action_sft',tmp_path/'adapter',{})
    assert (output/'note').read_text()=='keep'


def test_grpo_base_handoff_rejects_changed_shard(tmp_path):
    from action_common import verify_sft_handoff
    model=tmp_path/'model';model.mkdir()
    shard=model/'model-00001.safetensors';shard.write_bytes(b'base')
    run=tmp_path/'sft';run.mkdir();adapter=run/'best-adapter';adapter.mkdir()
    base_locks={str(shard):sha256(shard)}
    (run/'provenance.json').write_text(json.dumps({'model_path':str(model),'input_sha256':base_locks}))
    verify_sft_handoff(adapter,model,base_locks,{})
    shard.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SFT.*base'):
        verify_sft_handoff(adapter,model,{str(shard):sha256(shard)},{})


def test_grpo_base_handoff_requires_complete_sft_inventory(tmp_path):
    from action_common import verify_sft_handoff
    model=tmp_path/'model';model.mkdir();shard=model/'weight';shard.write_bytes(b'base')
    run=tmp_path/'sft';run.mkdir();adapter=run/'best-adapter';adapter.mkdir()
    (run/'provenance.json').write_text(json.dumps({'model_path':str(model),'input_sha256':{}}))
    with pytest.raises(ValueError,match='SFT.*base'):
        verify_sft_handoff(adapter,model,{str(shard):sha256(shard)},{})


def test_qwen_defaults_cannot_override_sampling_or_greedy():
    from action_common import generation_arguments
    from transformers import GPT2Config,GPT2LMHeadModel,GenerationConfig
    model=GPT2LMHeadModel(GPT2Config(vocab_size=16,n_embd=8,n_layer=1,n_head=1,n_positions=8))
    model.generation_config=GenerationConfig(do_sample=True,temperature=.6,top_p=.95,top_k=20,
                                             transformers_version='4.52.3')
    for sampling in (True,False):
        cfg=GenerationConfig(do_sample=sampling,temperature=1.,top_p=1.,top_k=0)
        resolved,_=model._prepare_generation_config(**generation_arguments(cfg))
        assert resolved.do_sample==sampling
        assert resolved.temperature==1 and resolved.top_p==1 and resolved.top_k==0


def test_training_sampling_auditor_is_source_locked():
    from action_common import source_files
    assert ROOT/'tasks/type-cot-experiment/type-cot-goldorder-20261009/sampling.py' in source_files()
