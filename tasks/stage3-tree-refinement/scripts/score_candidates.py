"""Reward-model input construction, length audits, and score computation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _escape(text: str) -> str:
    return str(text).replace('\\', '\\\\').replace('"', '\\"')


def candidate_to_sexpr(candidate: dict[str, Any]) -> str:
    trigger = candidate.get('trigger_text', candidate.get('trigger', ''))
    args = ' '.join(f'({argument["role"]} "{_escape(argument.get("text", ""))}")'
                    for argument in candidate.get('args', []))
    return f'({candidate["type"]} (Trigger "{_escape(trigger)}")' + (f' {args}' if args else '') + ')'


def build_forest_context(forest: Iterable[dict[str, Any]], target: dict[str, Any]) -> str:
    forest = list(forest)
    result = []
    replaced = False
    for event in forest:
        same = event is target
        if not same:
            try:
                from common import event_key
                same = event_key(event) == event_key(target)
            except (KeyError, ValueError):
                same = False
        if same and not replaced:
            result.append(f'({event["type"]} [THIS])')
            replaced = True
        else:
            result.append(candidate_to_sexpr(event))
    if not replaced:
        result.append(f'({target["type"]} [THIS])')
    return ' '.join(result)


def build_reward_input(abstract: str, forest_context: str, candidate_event: str) -> str:
    return (f'Abstract:\n{abstract}\n\n'
            f'Forest context:\n{forest_context}\n\n'
            f'Candidate event:\n{candidate_event}')


def build_reward_row(data_row: dict[str, Any], greedy_forest: Iterable[dict[str, Any]],
                    candidate: dict[str, Any]) -> dict[str, Any]:
    label = candidate.get('label', {})
    text = build_reward_input(
        data_row['abstract'], build_forest_context(greedy_forest, candidate),
        candidate_to_sexpr(candidate),
    )
    return {
        'id': f'{data_row["id"]}:{repr(candidate.get("event_key", candidate.get("type")))}',
        'document_id': data_row['id'], 'text': text,
        'exists': int(label.get('exists', 0)), 'type_ok': int(label.get('type_ok', 0)),
        'event_score': float(label.get('event_score', 0.0)),
        'source': candidate.get('source', 'real'),
    }


def reward_score(p_exists: float, p_type_ok: float, event_score_hat: float) -> float:
    return float(p_exists) * float(p_type_ok) * (0.5 + 0.5 * float(event_score_hat))


def audit_lengths(tokenizer: Any, texts: Iterable[str], max_len: int,
                  strict: bool = True) -> dict[str, Any]:
    texts = list(texts)
    if max_len < 1:
        raise ValueError('max_len must be positive')
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
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


def binary_auc(labels: Iterable[float], scores: Iterable[float]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positives = sum(label > 0.5 for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return float('nan')
    rank_sum = sum(index for index, (_, label) in enumerate(pairs, 1) if label > 0.5)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def spearman_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    left, right = list(left), list(right)
    if len(left) != len(right) or len(left) < 2:
        return float('nan')

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and values[order[end]] == values[order[position]]:
                end += 1
            rank = (position + 1 + end) / 2
            for index in order[position:end]:
                result[index] = rank
            position = end
        return result

    a, b = ranks(left), ranks(right)
    mean_a, mean_b = sum(a) / len(a), sum(b) / len(b)
    numerator = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom_a = sum((x - mean_a) ** 2 for x in a) ** 0.5
    denom_b = sum((y - mean_b) ** 2 for y in b) ** 0.5
    return numerator / (denom_a * denom_b) if denom_a and denom_b else float('nan')


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


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
    all_exists, all_exists_scores = [], []
    all_type, all_type_scores = [], []
    all_event, all_event_scores = [], []
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
            all_exists.extend(token_rows[index]['labels']['exists'] for index in range(start, start + len(scores)))
            all_exists_scores.extend(exists)
            all_type.extend(token_rows[index]['labels']['type_ok'] for index in range(start, start + len(scores)))
            all_type_scores.extend(type_ok)
            all_event.extend(token_rows[index]['labels']['event_score'] for index in range(start, start + len(scores)))
            all_event_scores.extend(event_score)
            for row, score, p_exists, p_type, p_event in zip(source_rows[start:start + args.batch_size],
                                                              scores, exists, type_ok, event_score):
                result = dict(row, p_exists=p_exists, p_type_ok=p_type,
                              event_score_hat=p_event, reward_score=score)
                stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
    audit_path = output.with_suffix('.audit.json')
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    metrics = {
        'exists_auc': finite_or_none(binary_auc(all_exists, all_exists_scores)),
        'type_ok_auc_on_exists': finite_or_none(binary_auc(
            [label for label, exists in zip(all_type, all_exists) if exists > 0.5],
            [score for score, exists in zip(all_type_scores, all_exists) if exists > 0.5])),
        'event_score_spearman_on_type_ok': finite_or_none(spearman_correlation(
            [label for label, type_ok in zip(all_event, all_type) if type_ok > 0.5],
            [score for score, type_ok in zip(all_event_scores, all_type) if type_ok > 0.5])),
    }
    output.with_suffix('.metrics.json').write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
