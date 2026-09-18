"""Greedy event-forest generation for one dataset file, with explicit provenance."""
import argparse
import json
import os
import time
from pathlib import Path

from linearize import parse_forest
from model_utils import (DEFAULT_MODEL, file_hash, load_model_config, prompt_tokens,
                         provenance, read_rows, write_json)

MAX_NEW_TOKENS = 768


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    rows = read_rows(args.data, args.limit)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting bf16 is required')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {**provenance(args.model, args.data), 'seed': args.seed, 'mode': rows[0].get('mode'),
              'batch_size': args.batch_size, 'limit': args.limit, 'do_sample': False,
              'max_new_tokens': MAX_NEW_TOKENS, 'enable_thinking': False, 'precision': 'bf16',
              'adapter': str(args.adapter.resolve()),
              'adapter_sha256': file_hash(args.adapter / 'adapter_model.safetensors')}
    write_json(Path(str(args.output) + '.config.json'), config)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=load_model_config(args.model),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    started, truncated = time.time(), 0
    with args.output.open('x', encoding='utf-8') as stream, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            encoded = [{'input_ids': prompt_tokens(row['prompt'], tokenizer)} for row in batch]
            if max(len(e['input_ids']) for e in encoded) + MAX_NEW_TOKENS > model.config.max_position_embeddings:
                raise ValueError('Inference prompt exceeds model context; truncation forbidden')
            inputs = tokenizer.pad(encoded, padding=True, return_tensors='pt').to('cuda')
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None)
            new_tokens = generated[:, inputs['input_ids'].shape[1]:]
            outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, raw, ids in zip(batch, outputs, new_tokens):
                hit_limit = bool((ids != tokenizer.pad_token_id).sum().item() >= MAX_NEW_TOKENS
                                 and tokenizer.eos_token_id not in ids.tolist())
                truncated += hit_limit
                events, diag = parse_forest(raw)
                record = {k: row[k] for k in ('id', 'document_id', 'split', 'mode')}
                record.update(raw_output=raw, parsed_events=events, parse_diagnostics=diag,
                              invalid=diag['invalid'], hit_max_new_tokens=hit_limit)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({'completed': start + len(batch), 'total': len(rows),
                              'seconds': time.time() - started, 'hit_max_new_tokens': truncated}), flush=True)
    write_json(Path(str(args.output) + '.complete.json'),
               {'status': 'completed', 'count': len(rows), 'wall_seconds': time.time() - started,
                'hit_max_new_tokens': truncated, 'output_sha256': file_hash(args.output)})


if __name__ == '__main__':
    main()
