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
from score_candidates import audit_lengths


class MultiHeadRewardModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone, 'config', None)
        hidden_size = hidden_size or getattr(config, 'hidden_size', None) or getattr(config, 'n_embd', None)
        if hidden_size is None:
            raise ValueError('backbone config has no hidden size')
        self.exists_head = nn.Linear(hidden_size, 1)
        self.type_head = nn.Linear(hidden_size, 1)
        self.event_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = getattr(output, 'last_hidden_state', None)
        if hidden is None:
            states = getattr(output, 'hidden_states', None)
            if not states:
                raise ValueError('backbone did not return hidden states')
            hidden = states[-1]
        last_positions = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[batch, last_positions]
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
    event_loss = _masked_mean(event_values, type_target > 0.5)
    return {'loss': exists_loss + type_loss + event_loss,
            'exists_loss': exists_loss, 'type_loss': type_loss, 'event_loss': event_loss}


def tokenize_reward_rows(rows: list[dict[str, Any]], tokenizer: Any, max_len: int,
                         strict: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = [row['text'] for row in rows]
    audit = audit_lengths(tokenizer, texts, max_len=max_len, strict=strict)
    encoded = tokenizer(texts, add_special_tokens=True, truncation=False)
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
    width = max(len(row['input_ids']) for row in rows)
    input_ids = [row['input_ids'] + [pad_token_id] * (width - len(row['input_ids'])) for row in rows]
    attention = [[1] * len(row['input_ids']) + [0] * (width - len(row['input_ids'])) for row in rows]
    labels = {key: torch.tensor([row['labels'][key] for row in rows], dtype=torch.float32)
              for key in ('exists', 'type_ok', 'event_score')}
    return {'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention, dtype=torch.long), 'labels': labels}


def load_backbone(model_path: str, dtype: torch.dtype = torch.bfloat16) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    backbone = AutoModel.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype,
                                         attn_implementation='sdpa')
    return tokenizer, backbone


def train_rows(rows: list[dict[str, Any]], model: MultiHeadRewardModel, tokenizer: Any,
               device: str, epochs: int, learning_rate: float, grad_accumulation: int,
               seed: int) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_updates = max(1, (len(rows) * epochs + grad_accumulation - 1) // grad_accumulation)
    warmup_updates = max(1, int(total_updates * 0.03))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_updates - warmup_updates))
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    update = 0
    totals = {'loss': 0.0, 'exists_loss': 0.0, 'type_loss': 0.0, 'event_loss': 0.0, 'batches': 0}
    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(seed + epoch).shuffle(order)
        for batch_start in range(0, len(order), grad_accumulation):
            batch = [rows[index] for index in order[batch_start:batch_start + grad_accumulation]]
            inputs = collate_reward_rows(batch, tokenizer.pad_token_id)
            inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                      for key, value in inputs.items()}
            with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',
                                dtype=torch.bfloat16, enabled=device.startswith('cuda')):
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                losses = multitask_loss(outputs, {key: value.to(device) for key, value in inputs['labels'].items()})
                loss = losses['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update <= warmup_updates:
                scale = update / warmup_updates
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate * scale
            else:
                scheduler.step()
            for key in totals:
                if key != 'batches':
                    totals[key] += float(losses[key].detach().cpu())
            totals['batches'] += 1
    batches = max(1, totals['batches'])
    return {key: value / batches for key, value in totals.items() if key != 'batches'} | {
        'epochs': epochs, 'updates': update, 'wall_seconds': time.time() - started,
        'status': 'completed',
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
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('reward training requires CUDA')
    if args.backend == 'roberta':
        args.learning_rate = 1e-5
        args.epochs = 3
    rows = read_jsonl(args.data)
    tokenizer, backbone = load_backbone(args.model)
    token_rows, audit = tokenize_reward_rows(rows, tokenizer, args.max_len, strict=args.backend == 'qwen')
    model = MultiHeadRewardModel(backbone)
    metrics = train_rows(token_rows, model, tokenizer, 'cuda', args.epochs, args.learning_rate,
                         args.grad_accumulation, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), args.output / 'model.pt')
    tokenizer.save_pretrained(args.output / 'tokenizer')
    atomic_write_json(args.output / 'audit.json', audit)
    atomic_write_json(args.output / 'metrics.json', metrics)
    atomic_write_json(args.output / 'config.json', {
        'model': args.model, 'backend': args.backend, 'max_len': args.max_len,
        'epochs': args.epochs, 'learning_rate': args.learning_rate,
        'grad_accumulation': args.grad_accumulation, 'seed': args.seed,
    })
    print(json.dumps(metrics, ensure_ascii=False))
