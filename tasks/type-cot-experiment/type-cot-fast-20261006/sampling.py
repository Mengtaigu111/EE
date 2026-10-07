"""Microbatch-independent sample evidence; optimizer groups remain 16 examples."""
import json
import math
from numbers import Integral
from pathlib import Path
import sys

CURRICULUM = Path(__file__).resolve().parents[2] / 'stage2-event-tree/curriculum'
if str(CURRICULUM) not in sys.path:
    sys.path.insert(0, str(CURRICULUM))
from curriculum_trainer import CurriculumTrainer
from sorted_first_sampler import SortedFirstSampler


def validate_microbatch(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (1, 2):
        raise ValueError('Only measured microbatch 1 or 2 is allowed')
    return int(value)


def observation_records(metadata, microbatch_index, optimizer_step):
    return [{**row, 'microbatch': microbatch_index, 'ordinal_in_batch': ordinal,
             'optimizer_step_before': optimizer_step}
            for ordinal, row in enumerate(metadata)]


class MicrobatchObservationMixin:
    """Replace only inherited row logging, retaining safe finite-loss/checkpoints."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch = validate_microbatch(self.args.per_device_train_batch_size)
        if self.args.gradient_accumulation_steps != 16 // batch:
            raise ValueError('Effective train batch must remain exactly 16')
        if self._train_batch_size != batch:
            raise ValueError('Automatic batch changes are forbidden')

    def training_step(self, model, inputs, num_items_in_batch=None):
        metadata = inputs.pop('__sampling')
        batch = validate_microbatch(self.args.per_device_train_batch_size)
        if not metadata or len(metadata) > batch:
            raise ValueError('Unexpected actual training microbatch')
        records = observation_records(metadata, self.observed_microbatches, self.state.global_step)
        with self.observation_path.open('a', encoding='utf-8') as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
        self.observed_microbatches += 1
        # SortedFirstTrainer only adds the legacy one-row logging. Bypass that
        # method and retain its parent's unchanged finite loss and Trainer token
        # denominator (num_items_in_batch spans the whole accumulation group).
        return CurriculumTrainer.training_step(self, model, inputs, num_items_in_batch)


def audit_sampling_prefix(records, ids, *, global_step, microbatch=1, max_epochs=4):
    import torch
    batch = validate_microbatch(microbatch)
    n = len(ids)
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(n > 0 and len(set(ids)) == n, 'Empty/duplicated train IDs')
    updates_per_epoch = math.ceil(n/16)
    require(0 < global_step <= max_epochs*updates_per_epoch, 'Training budget outside course')
    full, partial = divmod(global_step, updates_per_epoch)
    counts = [n]*full + ([min(partial*16, n)] if partial else [])
    require(len(records) == sum(counts), 'Actual row consumption differs from optimizer steps')
    position = batch_offset = 0
    for epoch, count in enumerate(counts):
        expected = list(range(n)) if epoch == 0 else torch.randperm(
            n, generator=torch.Generator().manual_seed(42+epoch)).tolist()
        rows = records[position:position+count]
        require(all(r['epoch'] == epoch for r in rows), 'Unexpected/nonmonotonic epoch')
        require([r['index'] for r in rows] == expected[:count], 'Sampling permutation/prefix differs')
        require(all(r['id'] == ids[r['index']] for r in rows), 'Sampling ID/index mismatch')
        require(all(r['optimizer_step_before'] == epoch*updates_per_epoch+j//16
                    for j, r in enumerate(rows)), 'Sampling optimizer boundary differs')
        require(all(r['microbatch'] == batch_offset+j//batch and r['ordinal_in_batch'] == j%batch
                    for j, r in enumerate(rows)), 'Actual microbatch or ordinal differs')
        position += count
        batch_offset += math.ceil(count/batch)
    return {'passed': True, 'global_step': global_step, 'observed_rows': len(records),
            'epoch_counts': counts, 'complete_epochs': full,
            'partial_epoch_rows': counts[-1] if partial else 0,
            'microbatch': batch, 'observed_microbatches': batch_offset}
