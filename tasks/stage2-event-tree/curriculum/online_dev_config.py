"""Immutable inputs and recipe for the new Sorted1 online-dev experiment."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
PROJECT = STAGE.parents[1]
SORTED = HERE / 'sorted-first-20261003/manifest.json'
REFERENCE = STAGE / 'runs/C-CL-Mean-AppendixNoCopy-Sorted1-42-20261003'
CONFIG_DIR = HERE / 'sorted-first-online-dev-20261005'
PROMPT_HASH = '2bcc66773a71904973ae4f85b865e60eddfacf4230515618576d3217e6776b41'
SPEC = {
    'experiment': 'C-CL-Mean-AppendixNoCopy-Sorted1-OnlineDev',
    'model_path': '/opt/models/Qwen3-8B', 'init_adapter': None, 'seed': 42,
    'total_optimizer_steps': 3208, 'eval_steps': 401, 'warmup_steps': 97,
    'checkpoint_steps': list(range(401, 3209, 401)), 'epochs': 4,
    'microbatch': 1, 'gradient_accumulation': 16, 'max_len': 4096,
    'r': 16, 'alpha': 32, 'dropout': .05, 'learning_rate': .0001,
    'warmup_ratio': .03, 'scheduler': 'linear-warmup-plateau',
    'min_delta': .001, 'reduce_patience': 3, 'stop_patience': 6, 'lr_factor': .5,
    'targets': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    'precision': 'bf16', 'gradient_checkpointing': True, 'enable_thinking': False,
    'batch_size': 4, 'max_new_tokens': 319, 'do_sample': False,
    'train_count': 12828, 'dev_count': 1593, 'test_count': 1627,
    'prompt_sha256': PROMPT_HASH, 'control_metric': 'official offset dev AC_f1',
    'evaluation_version': 'evaluation-v2-20261003',
    'adam_beta1': .9, 'adam_beta2': .999, 'adam_epsilon': 1e-8, 'weight_decay': 0.,
}


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_locks(locks):
    for name, expected in locks.items():
        require(sha(name) == expected, f'Frozen input/source changed: {name}')


def build_manifest():
    old = read(SORTED)
    run = read(REFERENCE / 'run.json')
    locks = {**old['source_hashes'], **old['output_hashes'], **run['source_code_hashes'],
             str(SORTED): sha(SORTED), str(REFERENCE / 'run.json'): sha(REFERENCE / 'run.json'),
             str(REFERENCE / 'training/config.json'): sha(REFERENCE / 'training/config.json')}
    data_dir = Path(old['data_dir'])
    for split in ('dev', 'test'):
        for p in (data_dir / f'{split}.jsonl', STAGE / 'data/none' / f'{split}.jsonl',
                  PROJECT / 'data/SciEvents' / f'{split}.json'):
            locks[str(p)] = sha(p)
    verify_locks(locks)
    result = {**SPEC, 'reference_run': str(REFERENCE), 'sorted_manifest': str(SORTED),
              'train_file': old['sorted_train_file'], 'source_train_file': old['source_train_file'],
              'ordered_ids_file': old['ordered_ids_file'], 'prompt_file': old['prompt_template_file'],
              'data_dir': str(data_dir), 'document_data_dir': str(STAGE / 'data/none'),
              'frozen_hashes': locks,
              'sampling': 'epoch0 frozen sorted; epochs1-3 randperm(seed+epoch)',
              'loss': 'assistant tokens only', 'resume_from_checkpoint': None}
    validate_manifest(result)
    return result


def validate_manifest(m):
    for key, value in SPEC.items():
        require(m.get(key) == value, f'Frozen recipe mismatch: {key}')
    require(m.get('resume_from_checkpoint') is None, 'Initial experiment must start fresh')
    old = read(SORTED)
    for new_key, old_key in [('train_file', 'sorted_train_file'),
                             ('source_train_file', 'source_train_file'),
                             ('ordered_ids_file', 'ordered_ids_file'),
                             ('prompt_file', 'prompt_template_file'), ('data_dir', 'data_dir')]:
        require(m.get(new_key) == old[old_key], f'Wrong frozen path: {new_key}')
    require(m.get('document_data_dir') == str(STAGE / 'data/none'), 'Wrong document gold')
    run = read(REFERENCE/'run.json')
    required_locks = {**old['source_hashes'], **old['output_hashes'], **run['source_code_hashes']}
    for name, digest in required_locks.items():
        require(m['frozen_hashes'].get(name) == digest, f'Missing/incorrect frozen hash: {name}')
    required_paths = [SORTED, REFERENCE/'run.json', REFERENCE/'training/config.json']
    for split in ('dev', 'test'):
        required_paths += [Path(m['data_dir'])/f'{split}.jsonl', STAGE/'data/none'/f'{split}.jsonl',
                           PROJECT/'data/SciEvents'/f'{split}.json']
    require(all(str(p) in m['frozen_hashes'] for p in required_paths), 'Missing frozen input hashes')
    verify_locks(m['frozen_hashes'])
    require(sha(m['prompt_file']) == PROMPT_HASH, 'Original prompt hash differs')
    rows = Path(m['train_file']).read_bytes().splitlines(keepends=True)
    source = {json.loads(line)['id']: line for line in
              Path(m['source_train_file']).read_bytes().splitlines(keepends=True)}
    ids = [json.loads(line)['id'] for line in rows]
    require(len(ids) == len(set(ids)) == len(source) == 12828, 'Train count/unique IDs differ')
    require(ids == Path(m['ordered_ids_file']).read_text().splitlines(), 'Frozen order differs')
    require(all(source[json.loads(line)['id']] == line for line in rows), 'Sample contents changed')
    for split in ('dev', 'test'):
        require(len(jsonl(Path(m['data_dir']) / f'{split}.jsonl')) == SPEC[f'{split}_count'],
                f'{split} coverage differs')


def source_files():
    names = ['online_dev_config.py', 'dev_feedback.py', 'online_dev_evaluation.py',
             'online_dev_trainer.py', 'train_sorted_first_online_dev.py',
             'run_sorted_first_online_dev.py', 'verify_sorted_first_online_dev.py']
    files = [HERE / name for name in names]
    repair = STAGE / 'scorer-repair-20261004'
    files += [repair / 'instance_scorer.py']
    return files
