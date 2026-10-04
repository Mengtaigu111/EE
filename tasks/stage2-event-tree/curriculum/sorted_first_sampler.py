"""Epoch-aware ordering and evidence from batches actually used by the model.

This deliberately inherits Sampler, not RandomSampler: Accelerate may replace
RandomSampler with SeedableRandomSampler and silently destroy the sorted pass.
Indices carry their epoch through DataLoader prefetch, rather than consulting
mutable sampler state later in the collator.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from curriculum_trainer import CurriculumTrainer


class SortedFirstSampler(Sampler):
    def __init__(self, size, seed=42):
        if size < 1:
            raise ValueError('Sampler requires a nonempty dataset')
        self.size, self.seed, self.epoch = size, seed, 0

    def __len__(self):
        return self.size

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('Epoch cannot be negative')
        self.epoch = int(epoch)

    def __iter__(self):
        epoch = self.epoch
        indices = range(self.size) if epoch == 0 else torch.randperm(
            self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
        return iter((epoch, index) for index in indices)


class IndexedDataset(Dataset):
    def __init__(self, rows, ids):
        if len(rows) != len(ids) or not rows or len(set(ids)) != len(ids):
            raise ValueError('Rows require aligned unique IDs')
        self.rows, self.ids = rows, list(ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (-1, key)
        return {**self.rows[index], '__sampling': {'epoch': epoch, 'index': index,
                                                 'id': self.ids[index]}}


class ObservedCollator:
    """Retain evidence until training_step; count speculative prefetched batches."""
    def __init__(self, collator):
        self.collator = collator
        self.microbatches = self.examples = self.assistant_tokens = 0

    def __call__(self, batch):
        self.microbatches += 1
        self.examples += len(batch)
        self.assistant_tokens += sum(sum(x != -100 for x in row['labels']) for row in batch)
        clean = [{k: v for k, v in row.items() if k != '__sampling'} for row in batch]
        result = self.collator(clean)
        result['__sampling'] = [row['__sampling'] for row in batch]
        return result


class SortedFirstTrainer(CurriculumTrainer):
    def __init__(self, *args, observation_path, sampling_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.dataloader_num_workers != 0 or self.args.dataloader_drop_last:
            raise ValueError('Observed sampling requires workers=0 and no dropped rows')
        if self.args.group_by_length:
            raise ValueError('Length grouping would violate difficulty order')
        self.sampling_seed = sampling_seed
        self.observed_microbatches = 0
        self.observation_path = Path(observation_path)
        self.observation_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents reusing evidence from a different run.
        with self.observation_path.open('x', encoding='utf-8'):
            pass

    def _get_train_sampler(self, train_dataset=None):
        return SortedFirstSampler(len(self.train_dataset if train_dataset is None else train_dataset),
                                  self.sampling_seed)

    def training_step(self, model, inputs, num_items_in_batch=None):
        metadata = inputs.pop('__sampling')
        # Accelerator can collate a future batch before the current one is used.
        # Log only training_step entries, with epoch embedded in each sample.
        with self.observation_path.open('a', encoding='utf-8') as stream:
            for row in metadata:
                stream.write(json.dumps({**row, 'microbatch': self.observed_microbatches,
                    'optimizer_step_before': self.state.global_step}, ensure_ascii=False) + '\n')
        self.observed_microbatches += 1
        return super().training_step(model, inputs, num_items_in_batch)


def sampling_audit(records, ids, *, seed, expected_epochs, global_step,
                   total_optimizer_steps, collated_microbatches):
    """Fail closed on partial passes, duplicates, wrong IDs, or incorrect order."""
    size = len(ids)
    epochs = []
    for epoch in range(expected_epochs):
        observed = [r for r in records if r['epoch'] == epoch]
        indices = [r['index'] for r in observed]
        expected = list(range(size)) if epoch == 0 else torch.randperm(
            size, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        coverage = sorted(indices) == list(range(size))
        identity = all(0 <= r['index'] < size and r['id'] == ids[r['index']] for r in observed)
        sequential = indices == list(range(size))
        permutation = indices == expected
        unique = len(indices) == len(set(indices))
        complete = len(indices) == size and coverage and identity and unique and permutation
        if epoch > 0 and size > 1:
            complete = complete and not sequential
        epochs.append({'epoch': epoch, 'count': len(indices), 'expected_count': size,
            'complete': complete, 'sequential': sequential, 'seeded_permutation': permutation,
            'unique': unique, 'coverage': coverage, 'identity': identity,
            'order_sha256': hashlib.sha256(json.dumps([r['id'] for r in observed],
                                          ensure_ascii=False).encode()).hexdigest()})
    observed_microbatches = len({r['microbatch'] for r in records})
    extra_epochs = sorted({r['epoch'] for r in records} - set(range(expected_epochs)))
    return {'format': 'sorted-first-sampling-v1', 'seed': seed, 'rows': size,
        'expected_epochs': expected_epochs, 'global_step': global_step,
        'total_optimizer_steps': total_optimizer_steps,
        'observed_microbatches': observed_microbatches,
        'collated_microbatches': collated_microbatches,
        'prefetched_microbatches': collated_microbatches - observed_microbatches,
        'extra_epochs': extra_epochs,
        'complete': global_step == total_optimizer_steps and not extra_epochs
                    and all(e['complete'] for e in epochs)
                    and collated_microbatches >= observed_microbatches,
        'epochs': epochs,
        'prefetch_note': 'Only training_step records enter observed-order; epoch travels with sampler index. '
                         'Collator may run ahead; extra collated batches are excluded from coverage.'}
