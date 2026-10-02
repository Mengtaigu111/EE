#!/usr/bin/env python3
"""Create a fresh-start easy-two-epoch schedule without rebuilding frozen data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / 'user-v2-20261002/mean_curriculum_manifest.json'
DEFAULT_OUTPUT = HERE / 'user-v2-easy2-20261002'
EXPECTED_PROMPT_SHA256 = '8e6d508af5e9de15e001a8dba1da979d45353f334ef365d94edd87aa5254f7a2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(stage1_count, stage2_count, effective_batch, total_steps):
    if min(stage1_count, stage2_count, effective_batch, total_steps) <= 0:
        raise ValueError('counts, batch and total steps must be positive')
    per_epoch = (stage1_count + effective_batch - 1) // effective_batch
    stage1_steps = 2 * per_epoch
    remaining = total_steps - stage1_steps
    if remaining <= 0:
        raise ValueError('total budget must leave positive stage2 steps')
    return per_epoch, stage1_steps, remaining


def build(output_dir=DEFAULT_OUTPUT, manifest_path=DEFAULT_MANIFEST):
    output_dir, manifest_path = Path(output_dir).resolve(), Path(manifest_path).resolve()
    old_bytes = manifest_path.read_bytes()
    parent = json.loads(old_bytes)
    frozen_hashes = {**parent['source_hashes'], **parent['output_hashes']}
    for path, expected in frozen_hashes.items():
        if sha(path) != expected:
            raise ValueError(f'frozen input hash differs: {path}')
    config = parent['baseline_training_config']
    effective_batch = config['microbatch'] * config['gradient_accumulation']
    if effective_batch != 16 or parent['baseline_total_optimizer_steps'] != 4010:
        raise ValueError('unexpected baseline batch or total optimizer budget')
    counts = (parent['stage1_total_count'], parent['stage2_total_count'])
    if counts != (6504, 12828) or parent['base_model'] != '/opt/models/Qwen3-8B':
        raise ValueError('unexpected frozen data counts or raw base model')
    per_epoch, stage1_steps, stage2_steps = budget(*counts, effective_batch, 4010)
    prompt = Path(parent['prompt_template_file']).read_bytes()
    if hashlib.sha256(prompt).hexdigest() != EXPECTED_PROMPT_SHA256:
        raise ValueError('approved prompt hash differs')
    if Path(parent['prompt_source']).read_bytes() != prompt:
        raise ValueError('approved source prompt differs from frozen template')
    manifest = copy.deepcopy(parent)
    manifest.update(
        experiment='C-CL-Mean-UserV2-Easy2',
        parent_manifest=str(manifest_path),
        parent_manifest_sha256=hashlib.sha256(old_bytes).hexdigest(),
        preserved_parent_manifest=str(output_dir/'parent_manifest.json'),
        stage1_epoch_target=2,
        effective_batch_size=effective_batch,
        stage1_steps_per_epoch=per_epoch,
        stage1_optimizer_steps=stage1_steps,
        stage2_optimizer_steps=stage2_steps,
        stage2_effective_epochs=stage2_steps / ((counts[1]+effective_batch-1)//effective_batch),
        initialization='fresh_base_model', init_adapter=None, resume_from_checkpoint=None,
        prompt_template_file=str(output_dir/'prompt.md'),
        checkpoint_global_steps=[802,814,1604,2406,3208,4010],
        schedule={
            'total_optimizer_steps':4010, 'stage1_optimizer_steps':stage1_steps,
            'stage2_optimizer_steps':stage2_steps, 'stage1_epoch_target':2,
            'stage2_resume_global_step':814, 'stage2_full_state_resume':True,
            'stage2_ignore_data_skip':True, 'lr_horizon_steps':4010,
        },
    )
    snapshots = {output_dir/'prompt.md':prompt, output_dir/'parent_manifest.json':old_bytes}
    manifest['source_hashes'] = dict(frozen_hashes)
    manifest['source_hashes'].update({str(manifest_path):sha(manifest_path), str(Path(__file__).resolve()):sha(__file__)})
    manifest['output_hashes'] = {str(path):hashlib.sha256(content).hexdigest() for path,content in snapshots.items()}
    manifest['verification'].update(frozen_data_reused_byte_identical=True, exact_two_easy_epochs=True,
                                     unchanged_total_budget=True, fresh_base_initialization=True,
                                     preserved_parent_manifest=True)
    manifest_bytes = (json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    outputs = {**snapshots, output_dir/'mean_curriculum_manifest.json':manifest_bytes}
    for path, content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f'refusing to overwrite different derived file: {path}')
    for path, content in outputs.items():
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                stream.write(content)
    if manifest_path.read_bytes() != old_bytes:
        raise ValueError('parent manifest changed during build')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = build(args.output_dir,args.manifest)
    print(json.dumps({key:result[key] for key in ('experiment','stage1_epoch_target','stage1_optimizer_steps','stage2_optimizer_steps','baseline_total_optimizer_steps','prompt_sha256')},indent=2))
