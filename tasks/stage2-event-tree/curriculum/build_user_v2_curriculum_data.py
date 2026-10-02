#!/usr/bin/env python3
"""Inject the exact approved user prompt, retaining frozen mean curriculum IDs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
DEFAULT_MANIFEST = HERE / 'mean_curriculum_manifest.json'
DEFAULT_PROMPT = PROJECT / 'task/prompt/1002-event-forest-v2.md'
DEFAULT_OUTPUT = HERE / 'user-v2-20261002'
DEFAULT_DATA = HERE.parent / 'data/sentence-c-user-v2-20261002'
ABSTRACT_PLACEHOLDER = '[S1] <sentence 1>\n[S2] <sentence 2>\n...'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save_bytes(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'refusing to overwrite different derived file: {path}')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content)


def encode_rows(rows):
    return ''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode('utf-8')


def convert_row(row, template):
    """Recover frozen prompt sentences and validate their token boundaries."""
    try:
        abstract, target = row['prompt'].rsplit('\nFull Abstract:\n',1)[1].split('\n\nTARGET SENTENCE ',1)
        target_match = re.fullmatch(r'\[S(\d+)\]:\n(.*?)\n\nEvent forest:\n?', target, re.DOTALL)
        sentences = []
        for i,line in enumerate(abstract.splitlines(),1):
            match = re.fullmatch(r'\[S(\d+)\] (.*)',line)
            if not match or int(match[1]) != i:
                raise ValueError('invalid frozen sentence labels')
            sentences.append(match[2])
        index = row['sentence_index']
        if not sentences or not 0 <= index < len(sentences):
            raise ValueError('sentence index outside frozen abstract')
        if not target_match or int(target_match[1]) != index+1 or target_match[2] != sentences[index]:
            raise ValueError('frozen target sentence mismatch')
        lengths = [len(s.split()) for s in sentences]
        bounds = [sum(lengths[:index]),sum(lengths[:index+1])]
        if ' '.join(sentences).split() != row['tokens'] or row['target_bounds'] != bounds:
            raise ValueError('sentence tokens/bounds do not tile frozen document')
        if row['sentences'] != sentences or row['target_sentence'] != sentences[index] or row['abstract'] != ' '.join(sentences):
            raise ValueError('stored sentence metadata differs from frozen prompt')
        if any(template.count(p) != 1 for p in (ABSTRACT_PLACEHOLDER,'[S{k}]','<target sentence>')):
            raise ValueError('approved template placeholders must occur once')
        result = dict(row)
        result['prompt'] = template.replace(ABSTRACT_PLACEHOLDER,abstract).replace('[S{k}]',f'[S{index+1}]').replace('<target sentence>',sentences[index])
        result['mode'] = 'sentence-user-v2'
        return result
    except (IndexError,KeyError,TypeError) as exc:
        raise ValueError(f'invalid frozen prompt metadata: {row.get("id")}') from exc


def build(output_dir=DEFAULT_OUTPUT, data_dir=DEFAULT_DATA, manifest_path=DEFAULT_MANIFEST, prompt_path=DEFAULT_PROMPT):
    output_dir,data_dir,manifest_path,prompt_path = [Path(p).resolve() for p in (output_dir,data_dir,manifest_path,prompt_path)]
    original = json.loads(manifest_path.read_text())
    frozen_dir = manifest_path.parent
    template_bytes = prompt_path.read_bytes()
    template = template_bytes.decode('utf-8')
    for path,expected in {**original['source_hashes'],**original['output_hashes']}.items():
        if sha(path) != expected:
            raise ValueError(f'frozen source hash differs: {path}')
    sources = {'train':Path(original['source_train_file']),'dev':Path(original['dev_data']),'test':Path(original['test_data'])}
    generated = {}
    for split,path in sources.items():
        source = read_rows(path)
        converted = [convert_row(row,template) for row in source]
        if len({r['id'] for r in converted}) != len(converted):
            raise ValueError(f'duplicate {split} IDs')
        generated[split] = converted
    train = generated['train']
    ids = {name:(frozen_dir/f'mean_curriculum_{name}_ids.txt').read_text().splitlines() for name in ('stage1','stage2','root_stage1')}
    if ids['stage2'] != [r['id'] for r in train] or ids['root_stage1'] != original['root_stage1_ids']:
        raise ValueError('frozen full train / root ID order mismatch')
    by_id = {r['id']:r for r in train}
    stage1 = [by_id[i] for i in ids['stage1']]
    if len(stage1) != original['stage1_total_count'] or len(train) != original['total_train'] or len(set(ids['stage1'])) != len(stage1):
        raise ValueError('frozen curriculum counts mismatch')
    if [r['id'] for r in read_rows(original['stage1_data'])] != ids['stage1']:
        raise ValueError('frozen stage1 row ID order mismatch')
    if sum(r['n_events']==0 for r in stage1) != original['root_stage1_count'] or {r['id'] for r in stage1 if not r['n_events']} != set(ids['root_stage1']):
        raise ValueError('frozen ROOT membership mismatch')
    outputs = {output_dir/'prompt.md':template_bytes}
    for split,rows in generated.items():
        outputs[data_dir/f'{split}.jsonl'] = encode_rows(rows)
    outputs[output_dir/'mean_curriculum_stage1.jsonl'] = encode_rows(stage1)
    outputs[output_dir/'mean_curriculum_stage2.jsonl'] = outputs[data_dir/'train.jsonl']
    copy_names = [f'mean_curriculum_{name}_ids.txt' for name in ids]
    copy_names += [f'{control}_control_{suffix}' for control in ('event_count','random_half') for suffix in ('assignments.csv','config.json')]
    for name in copy_names:
        outputs[output_dir/name] = (frozen_dir/name).read_bytes()
    # Check all protected destinations before making any writes.
    for path,content in outputs.items():
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f'refusing to overwrite different derived file: {path}')
    for path,content in outputs.items():
        save_bytes(path,content)
    for split,path in sources.items():
        for old,new in zip(read_rows(path),read_rows(data_dir/f'{split}.jsonl')):
            if {k:v for k,v in old.items() if k not in ('mode','prompt')} != {k:v for k,v in new.items() if k not in ('mode','prompt')}:
                raise ValueError('gold or metadata payload changed')
    if (output_dir/'mean_curriculum_stage2.jsonl').read_bytes() != (data_dir/'train.jsonl').read_bytes():
        raise ValueError('stage2 differs from new fulltrain')
    if prompt_path.read_bytes() != template_bytes:
        raise ValueError('approved prompt changed during build')
    manifest = copy.deepcopy(original)
    manifest.update(experiment='C-CL-Mean-UserV2',data_dir=str(data_dir),source_train_file=str(data_dir/'train.jsonl'),
                    stage1_data=str(output_dir/'mean_curriculum_stage1.jsonl'),stage2_data=str(output_dir/'mean_curriculum_stage2.jsonl'),
                    dev_data=str(data_dir/'dev.jsonl'),test_data=str(data_dir/'test.jsonl'),
                    prompt_source=str(prompt_path),prompt_path=str(prompt_path),prompt_template_file=str(output_dir/'prompt.md'),prompt_sha256=sha(prompt_path),
                    parent_manifest=str(manifest_path),mode='sentence-user-v2',
                    comparison_limitation='Comparison against AppendixNoCopy baseline changes both prompt and curriculum; it does not isolate curriculum effects.',
                    root_sampling='exact frozen root_stage1_ids order from parent manifest; no resampling',
                    controls_note='Copied historical configurations and assignments for provenance only; auto_train=false, no control training.')
    manifest['source_hashes'].update({str(p):sha(p) for p in [manifest_path,prompt_path,Path(__file__).resolve(),*outputs]})
    manifest['output_hashes'] = {str(p):sha(p) for p in outputs}
    manifest['verification'] = {'byte_preserving_stage1':True,'byte_identical_stage2':True,
                                'unique_complete_assignment_join':True,'fixed_threshold_and_gold_counts':True,
                                'exact_user_template':True,'immutable_nonprompt_payload':True,
                                'frozen_stage_and_root_id_order':True,'template_file_unchanged':True}
    save_bytes(output_dir/'mean_curriculum_manifest.json',(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--prompt',type=Path,default=DEFAULT_PROMPT)
    args = parser.parse_args()
    result = build(args.output_dir,args.data_dir,args.manifest,args.prompt)
    print(json.dumps({k:result[k] for k in ('experiment','data_dir','stage1_total_count','stage2_total_count','prompt_sha256','baseline_total_optimizer_steps')},indent=2))
