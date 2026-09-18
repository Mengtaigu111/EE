"""Sequential Stage-2 runs with durable stage status (same conventions as Stage-1).

Experiments: T0 = no guidance, T1 = flat typed hints, T2 = typed hints + role->nugget-type schema.
For T1/T2 each split is predicted twice: with Stage-1 hints (main) and with gold hints (oracle).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
MODE_OF = {'smoke': 'none', 'T0': 'none', 'T1': 'flat', 'T2': 'typed', 'T3': 'natural', 'T4': 'none', 'T5': 'noschema'}
PRESTAGE = {'T4': ('sub', 2)}  # experiment -> (substructure mode, epochs) trained before the full-tree stage
MAX_LEN = {'none': 2048, 'flat': 2816, 'typed': 2816, 'natural': 2048, 'noschema': 2048, 'sub': 2048}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def run_command(command, path, state, outputs=()):
    exit_path = path.with_suffix('.exit.json')
    if exit_path.exists() and json.loads(exit_path.read_text())['exit_code'] == 0 and all(p.exists() for p in outputs):
        print(f'{now()} Reusing successful stage {path}', flush=True)
        return
    existing = [p for p in [path, exit_path, *outputs] if p.exists()]
    for output in outputs:
        existing.extend(p for p in output.parent.glob(output.name + '.*') if p not in existing)
    if existing:
        archive = path.parent / 'failed_attempts' / (path.stem + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        archive.mkdir(parents=True)
        for previous in dict.fromkeys(existing):
            previous.rename(archive / previous.name)
    state.update(status='running', stage=path.stem, command=command, updated_at=now())
    save(path.parent / 'run.json', state)
    started = time.time()
    with path.open('x') as log:
        log.write(shlex.join(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT.parents[1])
        state.update(pid=process.pid)
        save(path.parent / 'run.json', state)
        print(f'{now()} PID={process.pid} {shlex.join(command)}', flush=True)
        code = process.wait()
    save(exit_path, dict(exit_code=code, wall_seconds=time.time() - started, finished_at=now()))
    if code:
        state.update(status='failed', exit_code=code, updated_at=now())
        save(path.parent / 'run.json', state)
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--experiments', default='smoke,T0,T2,T1', help='comma list from smoke,T0..T5')
    parser.add_argument('--seeds', default='42', help='comma list; smoke always uses 42')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--no-oracle', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('GPU execution must run in tmux')
    experiments = [e.strip() for e in args.experiments.split(',') if e.strip()]
    unknown = [e for e in experiments if e not in MODE_OF]
    if unknown:
        parser.error(f'unknown experiments: {unknown}')
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    queue = args.queue_dir or ROOT / 'queues' / stamp
    if args.resume and (not args.queue_dir or not (queue / 'queue.json').exists()):
        parser.error('--resume requires an existing --queue-dir')
    queue.mkdir(parents=True, exist_ok=args.resume)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(SCRIPTS.glob('*')) if p.is_file()}
    queue_state = json.loads((queue / 'queue.json').read_text()) if args.resume else dict(started_at=now(), stamp=stamp)
    stamp = queue_state['stamp']
    queue_state.update(status='running', pid=os.getpid(), scripts=sources, updated_at=now(),
                       tmux=os.environ['TMUX'], python=sys.executable, experiments=experiments, seeds=seeds)
    save(queue / 'queue.json', queue_state)
    plan = [(e, 42) for e in experiments if e == 'smoke'] + [(e, s) for s in seeds for e in experiments if e != 'smoke']
    status, run, state = 'failed', None, {}
    try:
        for experiment, seed in plan:
            mode = MODE_OF[experiment]
            run = ROOT / 'runs' / f'{experiment}-{seed}-{stamp}'
            run.mkdir(parents=True, exist_ok=args.resume)
            state = dict(experiment=experiment, seed=seed, mode=mode, status='pending', started_at=now(), queue=str(queue))
            save(run / 'run.json', state)
            save(run / 'source_hashes.json', sources)
            data = ROOT / 'data' / mode
            init = []
            if experiment in PRESTAGE:
                pre_mode, pre_epochs = PRESTAGE[experiment]
                command = [sys.executable, str(SCRIPTS / 'train_lora.py'), '--data', str(ROOT / 'data' / pre_mode / 'train.jsonl'),
                           '--output', str(run / 'pretraining'), '--seed', str(seed), '--epochs', str(pre_epochs),
                           '--max-len', str(MAX_LEN[pre_mode])]
                run_command(command, run / 'pretrain.log', state, outputs=[run / 'pretraining'])
                init = ['--init-adapter', str(run / 'pretraining/adapter')]
            command = [sys.executable, str(SCRIPTS / 'train_lora.py'), '--data', str(data / 'train.jsonl'),
                       '--output', str(run / 'training'), '--seed', str(seed), '--epochs', str(args.epochs),
                       '--max-len', str(MAX_LEN[mode])] + init
            if experiment == 'smoke':
                command += ['--limit', '8', '--max-steps', '1']
            run_command(command, run / 'train.log', state, outputs=[run / 'training'])
            variants = [(mode, '')]
            if mode in ('flat', 'typed') and not args.no_oracle and experiment != 'smoke':
                variants.append((f'{mode}-oracle', '.oracle'))
            for split in (['dev'] if experiment == 'smoke' else ['dev', 'test']):
                for variant, suffix in variants:
                    pred = run / f'{split}{suffix}.predictions.jsonl'
                    command = [sys.executable, str(SCRIPTS / 'predict.py'), '--data', str(ROOT / 'data' / variant / f'{split}.jsonl'),
                               '--output', str(pred), '--seed', str(seed), '--adapter', str(run / 'training/adapter'), '--batch-size', '4']
                    if experiment == 'smoke':
                        command += ['--limit', '2']
                    run_command(command, run / f'{split}{suffix}.predict.log', state, outputs=[pred])
                    metrics = run / f'{split}{suffix}.metrics.json'
                    command = [sys.executable, str(SCRIPTS / 'evaluate.py'), '--data', str(ROOT / 'data' / variant / f'{split}.jsonl'),
                               '--predictions', str(pred), '--output', str(metrics), '--complex']
                    if experiment == 'smoke':
                        command += ['--allow-subset']
                    run_command(command, run / f'{split}{suffix}.evaluate.log', state, outputs=[metrics])
            state.update(status='completed', finished_at=now(), exit_code=0)
            save(run / 'run.json', state)
            subprocess.run([sys.executable, str(SCRIPTS / 'write_report.py'), '--root', str(ROOT)], check=False)
        status = 'completed'
    except BaseException as error:
        queue_state.update(error=repr(error), failed_run=str(run) if run else None, failed_stage=state.get('stage'))
        raise
    finally:
        queue_state.update(status=status, finished_at=now())
        save(queue / 'queue.json', queue_state)
        subprocess.run([sys.executable, str(SCRIPTS / 'write_report.py'), '--root', str(ROOT)], check=False)


if __name__ == '__main__':
    main()
