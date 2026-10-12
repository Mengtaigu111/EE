"""Original batch-one sampling auditor with explicit seeded course."""
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_sampling_prefix(records, ids, *, global_step, accumulation=16, max_epochs=4, seed=42):
    import torch
    n = len(ids)
    require(n > 0 and len(set(ids)) == n, 'Empty/duplicated train IDs')
    updates_per_epoch = math.ceil(n / accumulation)
    require(0 < global_step <= max_epochs * updates_per_epoch, 'Training budget outside course')
    full, partial = divmod(global_step, updates_per_epoch)
    counts = [n] * full + ([min(partial * accumulation, n)] if partial else [])
    require(len(records) == sum(counts), 'Actual row consumption differs from optimizer steps')
    require([r['microbatch'] for r in records] == list(range(len(records))), 'Microbatch evidence repeats/skips')
    position = 0
    for epoch, count in enumerate(counts):
        expected = list(range(n)) if epoch == 0 else torch.randperm(
            n, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        rows = records[position:position + count]
        require(all(r['epoch'] == epoch for r in rows), 'Unexpected/nonmonotonic epoch')
        require([r['index'] for r in rows] == expected[:count], 'Sampling permutation/prefix differs')
        require(all(r['id'] == ids[r['index']] for r in rows), 'Sampling ID/index mismatch')
        # An incomplete gradient accumulation group is flushed at epoch end.
        require(all(r['optimizer_step_before'] == epoch * updates_per_epoch + j // accumulation
                    for j, r in enumerate(rows)), 'Sampling optimizer boundary differs')
        position += count
    return {'passed': True, 'global_step': global_step, 'observed_rows': len(records),
            'epoch_counts': counts, 'complete_epochs': full,
            'partial_epoch_rows': counts[-1] if partial else 0}
