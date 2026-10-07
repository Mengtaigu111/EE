"""Greedy Qwen3-8B generation for Type-COT rows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE2_SCRIPTS = PROJECT_ROOT / "tasks/stage2-event-tree/scripts"
if str(STAGE2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(STAGE2_SCRIPTS))

from model_utils import DEFAULT_MODEL, load_model_config, prompt_tokens, read_rows, write_json  # noqa: E402
from parser import parse_type_cot_output  # noqa: E402


def predict(data_path: Path, output_path: Path, model_path: str, adapter_path: Path,
            batch_size: int, seed: int, max_new_tokens: int) -> None:
    if output_path.exists():
        raise FileExistsError(output_path)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    rows = read_rows(data_path)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU supporting bf16 is required")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=load_model_config(model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    hit_limit_total = 0
    with output_path.open("x", encoding="utf-8") as stream, torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = [{"input_ids": prompt_tokens(row["prompt"], tokenizer)} for row in batch]
            if max(len(item["input_ids"]) for item in encoded) + max_new_tokens > model.config.max_position_embeddings:
                raise ValueError("inference prompt exceeds model context; truncation forbidden")
            inputs = tokenizer.pad(encoded, padding=True, return_tensors="pt").to("cuda")
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, raw, ids in zip(batch, outputs, new_tokens):
                hit_limit = bool(
                    (ids != tokenizer.pad_token_id).sum().item() >= max_new_tokens
                    and tokenizer.eos_token_id not in ids.tolist()
                )
                hit_limit_total += int(hit_limit)
                record = {key: row[key] for key in ("id", "document_id", "split", "mode") if key in row}
                record.update(raw_output=raw, parse=parse_type_cot_output(raw), hit_max_new_tokens=hit_limit)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
    write_json(Path(str(output_path) + ".complete.json"), {
        "status": "completed",
        "count": len(rows),
        "hit_max_new_tokens": hit_limit_total,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=1125)
    args = parser.parse_args()
    predict(args.data, args.output, args.model, args.adapter, args.batch_size, args.seed, args.max_new_tokens)


if __name__ == "__main__":
    main()
