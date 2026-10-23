"""Single-model A800 continuation -> B, with CLI smoke and explicit full resume."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from recovery_runtime import archive_uninitialized_run, validate_smoke

HERE=Path(__file__).resolve().parent
ACTION=HERE.parent
ROOT=ACTION.parents[1]
PYTHON='/home/myluo/miniconda3/envs/qwen3meld-run/bin/python'
SFT=ACTION/'runs/action-sft-v2-42-20261012/best-adapter'
OLD_A=ACTION/'runs/r0-v2-A-original-sft1604-patience3-42-20261016'
LAUNCH=ACTION/'launch-recovery-A800-B-v5-20261019'


def write_state(value):
    LAUNCH.mkdir(exist_ok=True)
    temp=LAUNCH/'pipeline-state.tmp'
    temp.write_text(json.dumps(dict(**value,updated_unix=time.time()),indent=2)+'\n')
    os.replace(temp,LAUNCH/'pipeline-state.json')


def stages():
    common=['--init-adapter',str(SFT),'--reward-version','r0_v2','--gradient-accumulation','4',
            '--dev-steps','400','--early-stopping-patience','3','--checkpoint-steps','25','--gpu-target-mib','21000']
    warm=['--warm-start-run',str(OLD_A),'--warm-start-step','800']
    return [
        ('smoke-A800',LAUNCH/'smoke-A800',['--smoke','--max-steps','802','--policy-update-mask','off']+warm+common),
        ('smoke-B',LAUNCH/'smoke-B',['--smoke','--max-steps','2','--policy-update-mask','field_v2']+common),
        ('A',ACTION/'runs/r0-v2-A800-recoverable-v5-42-20261019',['--max-steps','3207','--policy-update-mask','off']+warm+common),
        ('B',ACTION/'runs/r0-v2-B-recoverable-v5-42-20261019',['--max-steps','3207','--policy-update-mask','field_v2']+common),
    ]


def main():
    if 'TMUX' not in os.environ:raise RuntimeError('Use tmux for the queue')
    LAUNCH.mkdir(exist_ok=True)
    with (LAUNCH/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=json.loads((LAUNCH/'source-sha256.json').read_text())
        def verify_sources():
            for name,digest in frozen.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Recovery source changed: '+name)
        def verify_smoke(name,out):
            try:
                validate_smoke(out)
            except (ValueError,OSError,TypeError) as error:
                write_state(dict(status='failed',stage=name,output=str(out),error=str(error),later_stages_started=False))
                raise
        verify_sources()
        (LAUNCH/'commands.json').write_text(json.dumps([{ 'stage':name,'output':str(out),'argv':args} for name,out,args in stages()],indent=2))
        for name,out,args in stages():
            verify_sources()
            metrics=out/'train_metrics.json'
            if metrics.exists():
                done=json.loads(metrics.read_text())
                if done.get('status')=='completed':
                    if name.startswith('smoke-'):verify_smoke(name,out)
                    continue
            command=[PYTHON,'-u',str(HERE/'train_grpo_recoverable.py'),'--output',str(out)]+args
            if (out/'recovery-checkpoints/latest.json').exists():
                from checkpoint_state import resolve_latest
                command+=['--resume-checkpoint',str(resolve_latest(out/'recovery-checkpoints'))]
            elif out.exists():
                archive_uninitialized_run(out)
            write_state(dict(status='running',stage=name,output=str(out),controller_pid=os.getpid(),command=command))
            with (LAUNCH/f'{name}.log').open('a') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                write_state(dict(status='failed',stage=name,output=str(out),exit_code=result.returncode,
                    later_stages_started=False,resume_available=(out/'recovery-checkpoints/latest.json').exists()))
                raise SystemExit(result.returncode)
            if name.startswith('smoke-'):
                verify_smoke(name,out)
        write_state(dict(status='completed',stages=['smoke-A800','smoke-B','A','B']))

if __name__=='__main__':main()
