"""Stream greedy typed-mention generations with explicit provenance."""
import argparse
import json
import os
import time
from pathlib import Path

from model_utils import (DEFAULT_MODEL, file_hash, load_model_config,
                         prompt_tokens, provenance, read_examples, read_rows,
                         write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--few-shot', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    if args.adapter and args.few_shot:
        parser.error('formal LoRA inference must not use few-shot examples')
    if args.output.exists():
        raise FileExistsError(args.output)
    # Formal adapter inference is memory-light enough for batch 4 on the 24 GiB
    # card. Keep E0/few-shot at the requested batch size for reproducibility.
    if args.adapter and args.batch_size == 1:
        args.batch_size = 4
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from align_mentions import parse_output
    rows = read_rows(args.data, args.limit)
    examples = read_examples(args.few_shot) if args.few_shot else None
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU supporting bf16 is required')
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {**provenance(args.model, args.data), 'seed': args.seed,
              'batch_size': args.batch_size, 'batch_size_auto': bool(args.adapter), 'limit': args.limit,
              'do_sample': False, 'max_new_tokens': 400, 'enable_thinking': False,
              'precision': 'bf16', 'adapter': str(args.adapter.resolve()) if args.adapter else None,
              'few_shot': str(args.few_shot.resolve()) if args.few_shot else None,
              'few_shot_sha256': file_hash(args.few_shot) if args.few_shot else None,
              'adapter_sha256': file_hash(args.adapter / 'adapter_model.safetensors') if args.adapter else None}
    write_json(Path(str(args.output) + '.config.json'), config)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=load_model_config(args.model),
        local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    started = time.time()
    with args.output.open('x', encoding='utf-8') as stream, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            prompts = []
            for row in batch_rows:
                if examples is None:
                    prompts.append(row['prompt'])
                else:
                    from prompts import make_prompt
                    prompts.append(make_prompt(row['target_sentence'], row['abstract'], examples))
            encoded = [{'input_ids': prompt_tokens(prompt, tokenizer)} for prompt in prompts]
            if max(len(row['input_ids']) for row in encoded) + 400 > model.config.max_position_embeddings:
                raise ValueError('Inference prompt exceeds model context; truncation forbidden')
            inputs = tokenizer.pad(encoded, padding=True, return_tensors='pt').to('cuda')
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=400,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                temperature=None, top_p=None, top_k=None)
            outputs = tokenizer.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            for row, raw in zip(batch_rows, outputs):
                mentions, invalid = parse_output(raw)
                stream.write(json.dumps({**row, 'raw_output': raw,
                    'parsed_mentions': mentions, 'invalid': invalid}, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps({'completed': start + len(batch_rows), 'total': len(rows),
                              'seconds': time.time() - started}), flush=True)
    write_json(Path(str(args.output) + '.complete.json'),
               {'status': 'completed', 'count': len(rows), 'wall_seconds': time.time() - started,
                'output_sha256': file_hash(args.output)})


if __name__ == '__main__':
    main()
