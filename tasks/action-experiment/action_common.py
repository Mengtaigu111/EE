"""Shared Action run defaults, exclusive outputs and complete content provenance."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_DIR = HERE/'data/action-v2-goldorder-20261012'
MODEL_PATH = '/opt/models/Qwen3-8B'
COT_ADAPTER = ROOT/'tasks/type-cot-experiment/runs/C-CoT1-NoRE-GoldOrder-42-20261009/cot/final-adapter'
MAX_NEW_TOKENS = 768
MAX_LENGTH = 4096
DEV_STEPS = [401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208]


def generation_arguments(config):
    # Transformers >=4.50 otherwise replaces explicit global-default values
    # (temperature=1, top_p=1, do_sample=False) with the Qwen model defaults.
    return {'generation_config':config,'use_model_defaults':False}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def append_jsonl(path, value):
    with Path(path).open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def verify_locks(locks):
    for name,expected in locks.items():
        if sha256(name) != expected:
            raise ValueError(f'Content lock changed: {name}')


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def verify_data(data_dir=DATA_DIR):
    data_dir=Path(data_dir).resolve()
    manifest=read_json(data_dir/'manifest.json')
    if manifest.get('format') != 'scievents-action-v1':
        raise ValueError('Expected audited Action dataset')
    if not all(manifest.get(k) for k in ('all_rows_roundtrip_verified','gold_text_offsets_verified','sorted1_exact_ids','prompt_context_only')):
        raise ValueError('Incomplete Action dataset verification')
    verify_locks(manifest['input_sha256'])
    verify_locks(manifest['output_sha256'])
    for name in ('train.jsonl','dev.jsonl','test.jsonl','train-sorted.jsonl','ordered-ids.txt','prompt_template.txt'):
        path=str((data_dir/name).resolve())
        if path not in manifest['output_sha256']:
            raise ValueError(f'Unfrozen dataset input: {path}')
    audit=read_json(data_dir/'token-audit.json')
    if not audit.get('assistant_prefix_verified') or audit.get('enable_thinking') is not False:
        raise ValueError('Expected real untruncated tokenizer audit')
    verify_locks(audit['tokenizer_inputs_sha256'])
    return manifest


def source_files():
    files=list(HERE.glob('*.py'))+list((HERE/'tests').glob('*.py'))
    files += [ROOT/'scripts/action_language.py',ROOT/'scripts/score_tree_presence.py',
              ROOT/'scripts/build_action_dataset.py',
              ROOT/'tasks/type-cot-experiment/type-cot-goldorder-20261009/config.py',
              ROOT/'tasks/type-cot-experiment/type-cot-goldorder-20261009/sampling.py',
              ROOT/'tasks/stage2-event-tree/scripts/model_utils.py']
    files += list((ROOT/'tasks/stage2-event-tree/curriculum').glob('*.py'))
    return sorted(set(files))


def verify_sft_handoff(adapter,model_path,base_locks,data_locks):
    source=read_json(Path(adapter).parent/'provenance.json')
    if Path(source['model_path']).resolve() != Path(model_path).resolve():
        raise ValueError('SFT reference base model path differs')
    inherited=source['input_sha256']
    if any(inherited.get(name) != value for name,value in base_locks.items()):
        raise ValueError('SFT reference base contents/inventory differ')
    if any(inherited.get(name) != value for name,value in data_locks.items()):
        raise ValueError('SFT reference data/scoring inputs differ')


def prepare_run(output,stage,init_adapter,configuration,model_path=MODEL_PATH,allow_smoke=False):
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output exists: {output}')
    if stage not in ('action_sft','action_grpo'):
        raise ValueError('Unknown Action stage')
    data_dir=Path(configuration.get('data_dir',DATA_DIR)).resolve()
    data=verify_data(data_dir)
    audit=read_json(data_dir/'token-audit.json')
    if configuration.get('max_len',MAX_LENGTH) < audit['lengths']['training_tokens']['p100']:
        raise ValueError('Training context limit would truncate gold targets')
    if not allow_smoke and configuration.get('max_new_tokens',MAX_NEW_TOKENS) < audit['lengths']['completion_tokens']['p100']:
        raise ValueError('Generation cap below longest audited gold completion')
    adapter=Path(init_adapter).resolve()
    for filename in ('adapter_config.json','adapter_model.safetensors'):
        if not (adapter/filename).is_file():raise FileNotFoundError(adapter/filename)
    # Reuse the repaired base-model inventory; no weak index-only lock.
    base_config=load_module('_action_complete_base_lock',ROOT/'tasks/type-cot-experiment/type-cot-goldorder-20261009/config.py')
    paths=base_config.model_source_files(model_path)
    base_locks={str(p.resolve()):sha256(p) for p in paths}
    if stage=='action_grpo':
        # A current hash alone cannot establish that the SFT stage used this base.
        handoff_data={str((data_dir/name).resolve()):data['output_sha256'][str((data_dir/name).resolve())]
                      for name in ('train.jsonl','dev.jsonl','test.jsonl','prompt_template.txt')}
        handoff_data.update({str(p.resolve()):sha256(p) for p in
                            (ROOT/'scripts/action_language.py',ROOT/'scripts/score_tree_presence.py',HERE/'action_evaluation.py')})
        verify_sft_handoff(adapter,model_path,base_locks,handoff_data)
    paths += list(adapter.glob('*.json'))+[adapter/'adapter_model.safetensors']
    paths += [p for p in (adapter.parent/'train_metrics.json',adapter.parent/'config.json',
                         adapter.parent/'configuration.json') if p.is_file()]
    paths += source_files()+[data_dir/'manifest.json']
    locks={**data['input_sha256'],**data['output_sha256'],
           **{str(p.resolve()):base_locks.get(str(p.resolve())) or sha256(p) for p in paths}}
    verify_locks(locks)
    output.mkdir(parents=True,exist_ok=False)
    for path in source_files():
        target=output/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha256(target)!=locks[str(path.resolve())]:
            raise ValueError(f'Source snapshot changed during freeze: {path}')
    verify_locks(locks)
    evidence=dict(format='action-run-provenance-v1',stage=stage,model_path=str(Path(model_path).resolve()),
                  init_adapter=str(adapter),data_dir=str(data_dir),input_sha256=locks,
                  enable_thinking=False,legal_action_mask=False,score_mode='hierarchical text presence',
                  control_metric='dev TC_f1, then TI_f1; exact ties retain earlier checkpoint',
                  smoke=bool(allow_smoke))
    write_json(output/'configuration.json',{**configuration,'stage':stage})
    write_json(output/'provenance.json',evidence)
    return evidence
