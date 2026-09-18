"""Reward-model input construction, length audits, and score computation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def build_reward_input(abstract: str, forest_context: str, candidate_event: str) -> str:
    return (f'Abstract:\n{abstract}\n\n'
            f'Forest context:\n{forest_context}\n\n'
            f'Candidate event:\n{candidate_event}')


def reward_score(p_exists: float, p_type_ok: float, event_score_hat: float) -> float:
    return float(p_exists) * float(p_type_ok) * (0.5 + 0.5 * float(event_score_hat))


def audit_lengths(tokenizer: Any, texts: Iterable[str], max_len: int,
                  strict: bool = True) -> dict[str, Any]:
    texts = list(texts)
    if max_len < 1:
        raise ValueError('max_len must be positive')
    encoded = tokenizer(texts, add_special_tokens=False, truncation=False)
    lengths = [len(ids) for ids in encoded['input_ids']]
    overlength = [{'index': index, 'tokens': length}
                  for index, length in enumerate(lengths) if length > max_len]
    result = {'count': len(lengths), 'max_tokens': max(lengths, default=0),
              'mean_tokens': sum(lengths) / len(lengths) if lengths else 0.0,
              'max_len': max_len, 'overlength_count': len(overlength),
              'overlength': overlength, 'truncation': False}
    if strict and overlength:
        raise ValueError(f'{len(overlength)} inputs exceed max_len={max_len}; truncation forbidden')
    return result


def score_model_outputs(outputs: dict[str, Any]) -> list[float]:
    import torch

    exists = torch.sigmoid(outputs['exists_logits']).detach().cpu().tolist()
    type_ok = torch.sigmoid(outputs['type_logits']).detach().cpu().tolist()
    event_score = outputs['event_score'].detach().cpu().tolist()
    return [reward_score(a, b, c) for a, b, c in zip(exists, type_ok, event_score)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-len', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    from train_reward import MultiHeadRewardModel, collate_reward_rows, load_backbone, tokenize_reward_rows
    from common import read_jsonl

    if not torch.cuda.is_available():
        raise RuntimeError('reward scoring requires CUDA')
    source_rows = read_jsonl(args.data)
    tokenizer_path = args.model_dir / 'tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else args.backbone,
                                              local_files_only=True)
    token_rows, audit = tokenize_reward_rows(source_rows, tokenizer, args.max_len, strict=True)
    _, backbone = load_backbone(args.backbone)
    model = MultiHeadRewardModel(backbone)
    model.load_state_dict(torch.load(args.model_dir / 'model.pt', map_location='cpu'))
    model.to('cuda').eval()
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream, torch.inference_mode():
        for start in range(0, len(token_rows), args.batch_size):
            batch = collate_reward_rows(token_rows[start:start + args.batch_size], tokenizer.pad_token_id)
            inputs = {key: value.to('cuda') if isinstance(value, torch.Tensor) else value
                      for key, value in batch.items() if key != 'labels'}
            predictions = model(inputs['input_ids'], inputs['attention_mask'])
            scores = score_model_outputs(predictions)
            exists = torch.sigmoid(predictions['exists_logits']).cpu().tolist()
            type_ok = torch.sigmoid(predictions['type_logits']).cpu().tolist()
            event_score = predictions['event_score'].cpu().tolist()
            for row, score, p_exists, p_type, p_event in zip(source_rows[start:start + args.batch_size],
                                                              scores, exists, type_ok, event_score):
                result = dict(row, p_exists=p_exists, p_type_ok=p_type,
                              event_score_hat=p_event, reward_score=score)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
    audit_path = output.with_suffix('.audit.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
