"""Multi-head whole-tree reward model used by Stage-3 R1/R0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from common import atomic_write_json, read_jsonl
from score_candidates import LengthAuditError, audit_lengths, format_reward_text


def _backbone_dtype(backbone: nn.Module) -> torch.dtype:
    for parameter in backbone.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _peak_memory_metrics(device: str) -> dict[str, float | int]:
    if not device.startswith('cuda') or not torch.cuda.is_available():
        return {'peak_memory_bytes': 0, 'peak_memory_gib': 0.0}
    torch.cuda.synchronize(torch.device(device))
    peak_bytes = int(torch.cuda.max_memory_allocated(torch.device(device)))
    return {'peak_memory_bytes': peak_bytes,
            'peak_memory_gib': peak_bytes / (1024 ** 3)}


def _resolve_pooling(backbone: nn.Module, pooling: str) -> str:
    if pooling in ('last', 'qwen'):
        return 'last'
    if pooling in ('cls', 'roberta'):
        return 'cls'
    if pooling != 'auto':
        raise ValueError("pooling must be one of 'auto', 'last', or 'cls'")
    model_type = str(getattr(getattr(backbone, 'config', None), 'model_type', '')).lower()
    return 'cls' if model_type in {'roberta', 'xlm-roberta'} else 'last'


class MultiHeadRewardModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int | None = None,
                 pooling: str = 'auto'):
        super().__init__()
        self.backbone = backbone
        self.pooling = _resolve_pooling(backbone, pooling)
        config = getattr(backbone, 'config', None)
        hidden_size = hidden_size or getattr(config, 'hidden_size', None) or getattr(config, 'n_embd', None)
        if hidden_size is None:
            raise ValueError('backbone config has no hidden size')
        self.exists_head = nn.Linear(hidden_size, 1)
        self.type_head = nn.Linear(hidden_size, 1)
        self.event_head = nn.Linear(hidden_size, 1)
        head_dtype = _backbone_dtype(backbone)
        self.exists_head.to(dtype=head_dtype)
        self.type_head.to(dtype=head_dtype)
        self.event_head.to(dtype=head_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError('input_ids and attention_mask must have the same 2-D shape')
        if not torch.all(attention_mask.sum(dim=1) > 0):
            raise ValueError('every input must contain at least one non-padding token')
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = getattr(output, 'last_hidden_state', None)
        if hidden is None:
            states = getattr(output, 'hidden_states', None)
            if not states:
                raise ValueError('backbone did not return hidden states')
            hidden = states[-1]
        if hidden.ndim != 3 or hidden.shape[:2] != input_ids.shape:
            raise ValueError('backbone hidden states have an incompatible shape')
        if self.pooling == 'cls':
            positions = attention_mask.argmax(dim=1)
        else:
            positions = attention_mask.shape[1] - 1 - attention_mask.flip(dims=(1,)).argmax(dim=1)
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch, positions]
        return {
            'exists_logits': self.exists_head(pooled).squeeze(-1),
            'type_logits': self.type_head(pooled).squeeze(-1),
            'event_score': torch.sigmoid(self.event_head(pooled).squeeze(-1)),
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype)
    denominator = mask.sum()
    if denominator.item() == 0:
        return values.sum() * 0.0
    return (values * mask).sum() / denominator


def multitask_loss(outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    exists_target = labels['exists'].to(outputs['exists_logits'].dtype)
    type_target = labels['type_ok'].to(outputs['type_logits'].dtype)
    event_target = labels['event_score'].to(outputs['event_score'].dtype)
    exists_loss = F.binary_cross_entropy_with_logits(
        outputs['exists_logits'], exists_target)
    type_values = F.binary_cross_entropy_with_logits(
        outputs['type_logits'], type_target, reduction='none')
    type_loss = _masked_mean(type_values, exists_target > 0.5)
    event_values = F.mse_loss(outputs['event_score'], event_target, reduction='none')
    event_loss = _masked_mean(event_values, (exists_target > 0.5) & (type_target > 0.5))
    return {'loss': exists_loss + type_loss + event_loss,
            'exists_loss': exists_loss, 'type_loss': type_loss, 'event_loss': event_loss}


def tokenize_reward_rows(rows: list[dict[str, Any]], tokenizer: Any, max_len: int,
                         strict: bool = True, backend: str = 'qwen') -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = [format_reward_text(row['text'], backend=backend) for row in rows]
    audit = audit_lengths(tokenizer, texts, max_len=max_len, strict=strict)
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
    if len(encoded['input_ids']) != len(rows):
        raise ValueError('tokenizer returned a different number of rows')
    token_rows = []
    for row, input_ids in zip(rows, encoded['input_ids']):
        token_rows.append({
            'input_ids': list(input_ids),
            'labels': {
                'exists': float(row['exists']),
                'type_ok': float(row['type_ok']),
                'event_score': float(row['event_score']),
            },
        })
    return token_rows, audit


def collate_reward_rows(rows: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    if not rows:
        raise ValueError('collate requires at least one row')
    if pad_token_id is None:
        raise ValueError('pad_token_id must not be None')
    if any(not row['input_ids'] for row in rows):
        raise ValueError('every row must contain at least one input token')
    width = max(len(row['input_ids']) for row in rows)
    input_ids = [row['input_ids'] + [pad_token_id] * (width - len(row['input_ids'])) for row in rows]
    attention = [[1] * len(row['input_ids']) + [0] * (width - len(row['input_ids'])) for row in rows]
    labels = {key: torch.tensor([row['labels'][key] for row in rows], dtype=torch.float32)
              for key in ('exists', 'type_ok', 'event_score')}
    return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention, dtype=torch.long), 'labels': labels}


def load_backbone(model_path: str, dtype: torch.dtype = torch.float32) -> tuple[Any, Any]:
    """Load tokenizer and encoder backbone.

    Training keeps fp32 master weights and runs the forward pass under bf16
    autocast; pure-bf16 parameters would lose most 2e-5-scale AdamW updates.
    Scoring may pass ``dtype=torch.bfloat16`` explicitly.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    backbone = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype,
                                         attn_implementation='sdpa')
    return tokenizer, backbone


def train_rows(rows: list[dict[str, Any]], model: MultiHeadRewardModel, tokenizer: Any,
               device: str, epochs: int, learning_rate: float, grad_accumulation: int,
               seed: int, microbatch_size: int = 1) -> dict[str, Any]:
    if not rows:
        raise ValueError('training requires at least one row')
    if epochs < 1:
        raise ValueError('epochs must be positive')
    if learning_rate <= 0:
        raise ValueError('learning_rate must be positive')
    if grad_accumulation < 1:
        raise ValueError('grad_accumulation must be positive')
    if microbatch_size < 1:
        raise ValueError('microbatch_size must be positive')
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    effective_batch_size = microbatch_size * grad_accumulation
    updates_per_epoch = (len(rows) + effective_batch_size - 1) // effective_batch_size
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_updates = max(1, int(total_updates * 0.03))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates - warmup_updates))
    started = time.time()
    update = 0
    microbatches = 0
    totals = {'loss': 0.0, 'exists_loss': 0.0, 'type_loss': 0.0, 'event_loss': 0.0}
    sample_count = 0
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        for window_start in range(0, len(order), effective_batch_size):
            window_end = min(window_start + effective_batch_size, len(order))
            window_count = window_end - window_start
            optimizer.zero_grad(set_to_none=True)
            for batch_start in range(window_start, window_end, microbatch_size):
                batch_indices = order[batch_start:batch_start + microbatch_size]
                batch = [rows[index] for index in batch_indices]
                inputs = collate_reward_rows(batch, tokenizer.pad_token_id)
                inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                          for key, value in inputs.items()}
                with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',
                                    dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                    outputs = model(inputs['input_ids'], inputs['attention_mask'])
                    losses = multitask_loss(
                        outputs, {key: value.to(device) for key, value in inputs['labels'].items()})
                    loss = losses['loss']
                (loss * (len(batch) / window_count)).backward()
                for key in totals:
                    totals[key] += float(losses[key].detach().cpu()) * len(batch)
                sample_count += len(batch)
                microbatches += 1
            optimizer.step()
            update += 1
            if update <= warmup_updates:
                scale = update / warmup_updates
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate * scale
            else:
                scheduler.step()
    samples = max(1, sample_count)
    return {key: value / samples for key, value in totals.items()} | {
        'epochs': epochs, 'updates': update, 'microbatches': microbatches,
        'microbatch_size': microbatch_size, 'grad_accumulation': grad_accumulation,
        'effective_batch_size': effective_batch_size,
        'wall_seconds': time.time() - started, 'status': 'completed',
        **_peak_memory_metrics(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=('qwen', 'roberta'), default='qwen')
    parser.add_argument('--max-len', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--grad-accumulation', type=int, default=32)
    parser.add_argument('--microbatch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.max_len < 1:
        raise ValueError('--max-len must be positive')
    if args.microbatch_size < 1:
        raise ValueError('--microbatch-size must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('reward training requires CUDA')
    if args.backend == 'roberta':
        args.learning_rate = 1e-5
        args.epochs = 3
    rows = read_jsonl(args.data)
    tokenizer, backbone = load_backbone(args.model, dtype=torch.float32)
    try:
        token_rows, audit = tokenize_reward_rows(rows, tokenizer, args.max_len,
                                                 strict=True, backend=args.backend)
    except LengthAuditError as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output / 'audit.json', exc.audit)
        raise
    model = MultiHeadRewardModel(backbone, pooling=args.backend)
    metrics = train_rows(token_rows, model, tokenizer, 'cuda', args.epochs, args.learning_rate,
                         args.grad_accumulation, args.seed, microbatch_size=args.microbatch_size)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), args.output / 'model.pt')
    tokenizer.save_pretrained(args.output / 'tokenizer')
    atomic_write_json(args.output / 'audit.json', audit)
    atomic_write_json(args.output / 'metrics.json', metrics)
    atomic_write_json(args.output / 'config.json', {
        'model': args.model, 'backend': args.backend, 'max_len': args.max_len,
        'pooling': model.pooling,
        'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'grad_accumulation': args.grad_accumulation,
        'microbatch_size': args.microbatch_size,
        'effective_batch_size': args.grad_accumulation * args.microbatch_size,
        'seed': args.seed, 'master_weights': 'fp32', 'autocast': 'bf16',
    })
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == '__main__':
    main()
