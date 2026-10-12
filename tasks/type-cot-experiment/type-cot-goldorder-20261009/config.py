"""Independent one-epoch CoT recipe with corrected gold-order tree supervision."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[2]
CURRICULUM=PROJECT/'tasks/stage2-event-tree/curriculum'
sys.path.insert(0,str(CURRICULUM))
from online_dev_config import read,jsonl,save,sha,require,verify_locks
EVAL_STEPS=[401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208]
COT_DATA=HERE.parent/'data/type-cot-recognition-user-final-20261005'
TREE_DATA=CURRICULUM.parent/'data/sentence-c-appendix-no-copy-gold-order-20261008'
SORTED=CURRICULUM/'sorted-first-gold-order-20261008'
REFERENCE=CURRICULUM/'sorted-first-online-dev-20261005/manifest.json'
COT_PROMPT=HERE.parent/'prompts/user-final-20261005/prompt_template.txt'
SPEC = dict(experiment='TypeCoT1-GoldOrder-AppendixNoCopy-Sorted1-DenseOnlineDev',
            model_path='/opt/models/Qwen3-8B', cot_init_adapter=None, cot_dev_diagnostic=False,
            cot_steps=802, cot_epochs=1, cot_warmup_steps=25,
            cot_scheduler='cosine', cot_learning_rate=.0001,
            cot_max_new_tokens=1125, tree_steps=3208, tree_epochs=4,
            tree_warmup_steps=97, tree_eval_steps=EVAL_STEPS,
            tree_max_new_tokens=319, control_metric='official offset dev AC_f1',
            test_controls_training=False, cot_selection='fixed final checkpoint',
            cot_data_dir=str(COT_DATA), cot_prompt_file=str(COT_PROMPT))

def build_tree_reference(seed):
    # Reuse numerical recipe only; old manifest's data/code locks are deliberately excluded.
    old=read(REFERENCE)
    tree={k:v for k,v in old.items() if k not in ('frozen_hashes','reference_run')}
    tree.update(experiment=experiment_name(seed),seed=seed,init_adapter=None,
        checkpoint_steps=EVAL_STEPS,eval_steps=None,train_file=str(SORTED/'train-sorted.jsonl'),
        source_train_file=str(TREE_DATA/'train.jsonl'),ordered_ids_file=str(SORTED/'ordered-ids.txt'),
        prompt_file=str(SORTED/'prompt.md'),sorted_manifest=str(SORTED/'manifest.json'),
        data_dir=str(TREE_DATA),document_data_dir=str(CURRICULUM.parent/'data/none'))
    paths=[TREE_DATA/f'{split}.jsonl' for split in ('train','dev','test')]
    paths += [TREE_DATA/'manifest.json',SORTED/'manifest.json',SORTED/'train-sorted.jsonl',
              SORTED/'ordered-ids.txt',SORTED/'prompt.md']
    paths += [CURRICULUM.parent/'data/none'/f'{split}.jsonl' for split in ('dev','test')]
    paths += [PROJECT/'data/SciEvents'/f'{split}.json' for split in ('train','dev','test')]
    tree['frozen_hashes']={str(p):sha(p) for p in paths}
    return tree


def audit_alignment(m):
    tree=m['tree_reference']
    for split in ('train','dev','test'):
        cot=jsonl(Path(m['cot_data_dir'])/f'{split}.jsonl')
        rows=jsonl(tree['train_file'] if split=='train' else Path(tree['data_dir'])/f'{split}.jsonl')
        by_id={r['id']:r for r in rows}
        require(len(cot)==len(rows)==len(by_id),f'{split} duplicate/count mismatch')
        require({r['id'] for r in cot}==set(by_id),f'{split} IDs differ')
        for row in cot:
            other=by_id[row['id']]
            for field in ('target_events','target_sentence','sentences','sentence_index','document_id'):
                require(row[field]==other[field],f'Gold/context differ {row["id"]}: {field}')
    rows=jsonl(tree['train_file'])
    require([r['id'] for r in rows]==Path(tree['ordered_ids_file']).read_text().splitlines(), 'Frozen Sorted1 ID order differs')
    for split in ('dev','test'):
        docs=jsonl(Path(tree['document_data_dir'])/f'{split}.jsonl')
        raw={d['doc_id']:d for d in read(PROJECT/'data/SciEvents'/f'{split}.json')}
        require(len(docs)==len(raw),f'{split} raw document count differs')
        for d in docs:
            require(d['gold_events']==raw[d['document_id']]['events'], 'Raw gold_events changed')
    correction=read(TREE_DATA/'manifest.json')
    require(all(correction['verification'].values()), 'Gold-order correction not verified')
    verify_locks(correction['outputs_sha256'])
    return True


ALLOWED_SEEDS=(42,66,58)

def experiment_name(seed):
    require(type(seed) is int and seed in ALLOWED_SEEDS, 'Only planned seeds 42,66,58 are allowed')
    return f"{SPEC['experiment']}-Seed{seed}"

def model_source_files(model_path):
    """Include actual sharded weights and tokenizer/generation inputs, not just metadata."""
    model = Path(model_path)
    required = [model/name for name in
                ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json')]
    weight_map = read(required[-1]).get('weight_map')
    require(isinstance(weight_map, dict) and weight_map, 'Missing/nonempty model weight_map required')
    shards = set()
    for name in weight_map.values():
        require(isinstance(name, str) and name, 'Invalid model shard filename')
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Model shard must be relative to model directory')
        shard = model/relative
        require(shard.is_file(), f'Missing model shard: {shard}')
        shards.add(shard)
    optional = [model/name for name in
                ('tokenizer.json', 'tokenizer.model', 'vocab.json', 'merges.txt',
                 'added_tokens.json', 'special_tokens_map.json', 'generation_config.json',
                 'chat_template.jinja') if (model/name).is_file()]
    return sorted(set(required + optional) | shards)

def data_locks(seed):
    locks=dict(build_tree_reference(seed)['frozen_hashes'])
    paths=[REFERENCE,COT_PROMPT]
    paths += [COT_DATA/name for name in ('train.jsonl','dev.jsonl','test.jsonl','manifest.json','verification.json')]
    paths += model_source_files(SPEC['model_path'])
    locks.update({str(p):sha(p) for p in paths})
    return locks

def build_manifest(seed=42):
    m={**SPEC,'experiment':experiment_name(seed),'seed':seed,
       'tree_reference':build_tree_reference(seed),'frozen_hashes':data_locks(seed)}
    validate_manifest(m)
    return m

def validate_manifest(m):
    seed=m.get('seed');name=experiment_name(seed)
    for key,value in SPEC.items():
        require(m.get(key)==(name if key=='experiment' else value),f'Frozen recipe mismatch: {key}')
    require(m['tree_reference']==build_tree_reference(seed),'Corrected tree reference differs')
    # Recompute every content hash once; equality checks both inventory and contents.
    # Historical incomplete manifests must be re-frozen for new runs, never upgraded here.
    require(m['frozen_hashes']==data_locks(seed),'Frozen lock inventory/content differs; create a new manifest')
    audit_alignment(m)

def source_files():
    files=list(HERE.glob('*.py'))+list((HERE/'tests').glob('*.py'))+list(CURRICULUM.glob('*.py'))
    for directory in (CURRICULUM.parent/'scripts',HERE.parent/'scripts',
        CURRICULUM.parent/'evaluation-v2-20261003/scripts',CURRICULUM.parent/'token-evaluation-20261003',
        CURRICULUM.parent/'scorer-repair-20261004'):
        files+=list(directory.glob('*.py'))
    files += [PROJECT/'utils.py',PROJECT/'eval_complex_quick_f1.py',
              PROJECT/'scripts/score_event_trees.py',PROJECT/'scripts/score_tree_presence.py']
    return sorted(set(files))
