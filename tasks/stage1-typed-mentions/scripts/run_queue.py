"""Sequential E0, E1, E1-S and repeated E1 runs, with durable stage status."""
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


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def run_command(command, path, state, outputs=()):
    exit_path = path.with_suffix('.exit.json')
    if exit_path.exists() and json.loads(exit_path.read_text())['exit_code'] == 0 and all(p.exists() for p in outputs):
        print(f'{now()} Reusing successful stage {path}', flush=True)
        return
    existing = [p for p in [path, exit_path, *outputs] if p.exists()]
    # Keep complete failed artifacts and prediction sidecars in an immutable attempt directory.
    for output in outputs:
        existing.extend(p for p in output.parent.glob(output.name+'.*') if p not in existing)
    if existing:
        archive = path.parent/'failed_attempts'/(path.stem+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        archive.mkdir(parents=True)
        for previous in dict.fromkeys(existing):
            previous.rename(archive/previous.name)

    state.update(status='running', stage=path.stem, command=command, updated_at=now())
    save(path.parent/'run.json', state)
    started = time.time()
    with path.open('x') as log:
        log.write(shlex.join(command)+'\n'); log.flush()
        try:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parents[1])
        except OSError as error:
            state.update(status='failed', error=str(error), updated_at=now())
            save(path.parent/'run.json',state)
            raise
        state.update(pid=process.pid)
        save(path.parent/'run.json', state)
        print(f'{now()} PID={process.pid} {shlex.join(command)}', flush=True)
        code = process.wait()
    save(path.with_suffix('.exit.json'),dict(exit_code=code,wall_seconds=time.time()-started,finished_at=now()))
    if code:
        state.update(status='failed',exit_code=code,updated_at=now())
        save(path.parent/'run.json',state)
        raise subprocess.CalledProcessError(code,command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queue-dir',type=Path)
    parser.add_argument('--e0-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not os.environ.get('TMUX'):
        raise RuntimeError('GPU execution must run in tmux')
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    queue = args.queue_dir or ROOT/'queues'/stamp
    if args.resume and (not args.queue_dir or not (queue/'queue.json').exists()):
        parser.error('--resume requires an existing --queue-dir')
    queue.mkdir(parents=True,exist_ok=args.resume)
    if not (SCRIPTS/'write_report.py').exists():
        raise FileNotFoundError('write_report.py must exist before queue starts')
    sources = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(SCRIPTS.glob('*')) if p.is_file()}
    queue_state = json.loads((queue/'queue.json').read_text()) if args.resume else dict(started_at=now(), stamp=stamp)
    stamp = queue_state['stamp']
    queue_state.update(status='running',pid=os.getpid(),scripts=sources,updated_at=now(),tmux=os.environ['TMUX'],python=sys.executable)
    save(queue/'queue.json',queue_state)
    status = 'failed'
    try:
        experiments = [('E0',42,'SD_G')]
        if not args.e0_only:
            experiments += [('smoke',42,'SD_G'),('E1',42,'SD_G'),('E1-S',42,'S_G'),('E1',43,'SD_G'),('E1',44,'SD_G')]
        for experiment,seed,mode in experiments:
            run = ROOT/'runs'/f'{experiment}-{seed}-{stamp}'
            run.mkdir(parents=True,exist_ok=args.resume)
            state = dict(experiment=experiment,seed=seed,mode=mode,status='pending',started_at=now(),queue=str(queue))
            save(run/'run.json',state)
            save(run/'source_hashes.json',sources)
            data = ROOT/'data'/mode
            if experiment != 'E0':
                command = [sys.executable,str(SCRIPTS/'train_lora.py'),'--data',str(data/'train.jsonl'),
                           '--output',str(run/'training'),'--seed',str(seed),'--max-len','1024' if mode=='S_G' else '1536']
                if experiment == 'smoke':
                    command += ['--limit','16','--max-steps','1']
                run_command(command,run/'train.log',state,outputs=[run/'training'])
            for split in (['dev'] if experiment in ('E0','smoke') else ['dev','test']):
                pred = run/f'{split}.predictions.jsonl'
                command = [sys.executable,str(SCRIPTS/'predict.py'),'--data',str(data/f'{split}.jsonl'),
                           '--output',str(pred),'--seed',str(seed),
                           # Inference has separate KV-cache memory from training;
                           # batch 4 reduces model reloads/tokenizer overhead while
                           # keeping training microbatch and effective batch fixed.
                           '--batch-size','4' if experiment != 'E0' else '1']
                if experiment=='E0':
                    command += ['--few-shot',str(ROOT/'data/few_shot.json'),'--limit','32']
                else:
                    command += ['--adapter',str(run/'training/adapter')]
                    if experiment=='smoke':
                        command += ['--limit','2']
                run_command(command,run/f'{split}.predict.log',state,outputs=[pred])
                command = [sys.executable,str(SCRIPTS/'evaluate.py'),'--data',str(data/f'{split}.jsonl'),
                           '--predictions',str(pred),'--output',str(run/f'{split}.metrics.json')]
                if experiment in ('E0','smoke'):
                    command += ['--allow-subset']
                run_command(command,run/f'{split}.evaluate.log',state,outputs=[run/f'{split}.metrics.json',run/f'{split}.metrics.errors.jsonl'])
            state.update(status='completed',finished_at=now(),exit_code=0)
            save(run/'run.json',state)
            subprocess.run([sys.executable,str(SCRIPTS/'write_report.py'),'--root',str(ROOT)],check=True)
        status='completed'
    except BaseException as error:
        queue_state.update(error=repr(error),failed_run=str(run) if 'run' in locals() else None,
                           failed_stage=state.get('stage') if 'state' in locals() else None)
        raise
    finally:
        queue_state.update(status=status,finished_at=now())
        save(queue/'queue.json',queue_state)
        if (SCRIPTS/'write_report.py').exists():
            subprocess.run([sys.executable,str(SCRIPTS/'write_report.py'),'--root',str(ROOT)],check=False)

if __name__ == '__main__': main()
