"""Frozen configuration for fresh one-epoch CoT followed by dense-dev Sorted1."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
CURRICULUM = PROJECT/'tasks/stage2-event-tree/curriculum'
sys.path.insert(0, str(CURRICULUM))
from online_dev_config import read, jsonl, save, sha, require, verify_locks
from online_dev_config import validate_manifest as validate_tree_reference

EVAL_STEPS = [401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208]
COT_DATA = HERE.parent/'data/type-cot-recognition-user-final-20261005'
COT_PROMPT = HERE.parent/'prompts/user-final-20261005/prompt_template.txt'
REFERENCE = CURRICULUM/'sorted-first-online-dev-20261005/manifest.json'
SPEC = dict(experiment='TypeCoT1-AppendixNoCopy-Sorted1-DenseOnlineDev', seed=42,
            model_path='/opt/models/Qwen3-8B', cot_init_adapter=None,
            cot_steps=802, cot_epochs=1, cot_warmup_steps=25,
            cot_scheduler='cosine', cot_learning_rate=.0001,
            cot_max_new_tokens=1125, tree_steps=3208, tree_epochs=4,
            tree_warmup_steps=97, tree_eval_steps=EVAL_STEPS,
            tree_max_new_tokens=319, control_metric='official offset dev AC_f1',
            test_controls_training=False, cot_selection='fixed final checkpoint',
            cot_data_dir=str(COT_DATA), cot_prompt_file=str(COT_PROMPT))


def audit_alignment(m):
    tree = m['tree_reference']
    for split in ('train','dev','test'):
        cot_rows = jsonl(Path(m['cot_data_dir'])/f'{split}.jsonl')
        tree_rows = jsonl(tree['train_file'] if split == 'train' else Path(tree['data_dir'])/f'{split}.jsonl')
        cot = {r['id']:r for r in cot_rows}
        other = {r['id']:r for r in tree_rows}
        require(len(cot)==len(cot_rows) and len(other)==len(tree_rows) and cot.keys()==other.keys(),
                f'{split} stage IDs differ')
        for key, row in cot.items():
            require(row['split']==split, f'Wrong CoT split: {key}')
            for field in ('target_events','target_sentence','sentences','sentence_index','document_id'):
                require(row[field]==other[key][field], f'Stage gold/context differs: {key} {field}')
    return True


def build_manifest():
    tree = read(REFERENCE)
    locks = {**tree['frozen_hashes'], str(REFERENCE):sha(REFERENCE),
             str(COT_PROMPT):sha(COT_PROMPT)}
    for name in ('train.jsonl','dev.jsonl','test.jsonl','manifest.json','verification.json'):
        path = COT_DATA/name
        locks[str(path)] = sha(path)
    for name in ('config.json','tokenizer_config.json','model.safetensors.index.json'):
        path = Path(SPEC['model_path'])/name
        locks[str(path)] = sha(path)
    m = {**SPEC, 'tree_reference':tree, 'frozen_hashes':locks}
    validate_manifest(m)
    return m


def validate_manifest(m):
    for name,value in SPEC.items():
        require(m.get(name)==value, f'Frozen CoT/tree recipe mismatch: {name}')
    require(m['tree_reference']==read(REFERENCE), 'Tree reference differs')
    validate_tree_reference(m['tree_reference'])
    required = {**m['tree_reference']['frozen_hashes'], str(REFERENCE):sha(REFERENCE),
                str(COT_PROMPT):sha(COT_PROMPT)}
    for name in ('train.jsonl','dev.jsonl','test.jsonl','manifest.json','verification.json'):
        path = COT_DATA/name
        required[str(path)] = sha(path)
    for name in ('config.json','tokenizer_config.json','model.safetensors.index.json'):
        path = Path(SPEC['model_path'])/name
        required[str(path)] = sha(path)
    require(m['frozen_hashes']==required, 'Frozen lock inventory differs')
    verify_locks(m['frozen_hashes'])
    audit_alignment(m)


def source_files():
    files = list(HERE.glob('*.py')) + list(CURRICULUM.glob('*.py'))
    for directory in (PROJECT/'tasks/stage2-event-tree/scripts',
                      HERE.parent/'scripts',
                      PROJECT/'tasks/stage2-event-tree/scorer-repair-20261004'):
        files += list(directory.glob('*.py'))
    # Fixed scorer lives in a separate frozen module directory.
    files += list((CURRICULUM.parent/'evaluation-v2-20261003/scripts').glob('*.py'))
    return sorted(set(files))
