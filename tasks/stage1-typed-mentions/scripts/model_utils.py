"""Shared untruncated Qwen tokenization and experiment provenance."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

DEFAULT_MODEL = '/opt/models/Qwen3-8B'


def load_model_config(model_path):
    """Return an explicit single-device config for this non-sharded run."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    # transformers 4.52.3 may have no tensor-parallel registry with torch 2.4;
    # empty plans keep the model on the one CUDA device used by this experiment.
    config.base_model_tp_plan = {}
    config.base_model_pp_plan = {}
    return config


def read_examples(path):
    examples = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(examples, list) or len(examples) != 4:
        raise ValueError('E0 requires exactly 4 fixed train examples')
    if any(row.get('split') != 'train' for row in examples):
        raise ValueError('Few-shot examples must come from train')
    if len({row['id'] for row in examples}) != 4:
        raise ValueError('Few-shot examples must have distinct IDs')
    return examples


def read_rows(path, limit=None):
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(model, data):
    packages = {}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    model = Path(model).resolve()
    return {'python': platform.python_version(), 'packages': packages,
            'data_path': str(Path(data).resolve()), 'data_sha256': file_hash(data),
            'model_path': str(model),
            'script_hashes': {path.name: file_hash(path) for path in
                              Path(__file__).resolve().parent.glob('*.py')},
            'model_files': {name: file_hash(model / name) for name in
                            ('config.json', 'tokenizer_config.json', 'model.safetensors.index.json')
                            if (model / name).exists()}}


def prompt_tokens(prompt, tokenizer):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, truncation=False)


def encode_training_row(row, tokenizer, max_len=None):
    messages = [{'role': 'user', 'content': row['prompt']},
                {'role': 'assistant', 'content': row['completion']}]
    prefix = prompt_tokens(row['prompt'], tokenizer)
    tokens = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=False, enable_thinking=False, truncation=False)
    if tokens[:len(prefix)] != prefix:
        raise ValueError(f"{row['id']}: assistant prompt token prefix mismatch")
    if len(tokens) <= len(prefix):
        raise ValueError(f"{row['id']}: no assistant target tokens")
    if max_len is not None and len(tokens) > max_len:
        raise ValueError(f"{row['id']}: {len(tokens)} tokens exceed max_len={max_len}; truncation forbidden")
    return {'input_ids': tokens, 'attention_mask': [1] * len(tokens),
            'labels': [-100] * len(prefix) + tokens[len(prefix):]}


def audit_rows(rows, tokenizer, max_len):
    lengths, target_lengths, overlength = [], [], []
    for row in rows:
        encoded = encode_training_row(row, tokenizer)
        size = len(encoded['input_ids'])
        lengths.append(size)
        target_lengths.append(sum(label != -100 for label in encoded['labels']))
        if size > max_len:
            overlength.append({'id': row['id'], 'tokens': size})
    return {'count': len(rows), 'max_len': max_len, 'max_tokens': max(lengths),
            'mean_tokens': sum(lengths) / len(lengths),
            'max_assistant_tokens': max(target_lengths),
            'overlength_count': len(overlength), 'overlength': overlength,
            'prefix_verified': True, 'truncation': False}


class AssistantCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        width = max(len(row['input_ids']) for row in rows)
        result = {}
        for key, pad in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]:
            result[key] = torch.tensor([row[key] + [pad] * (width - len(row[key]))
                                        for row in rows], dtype=torch.long)
        return result
