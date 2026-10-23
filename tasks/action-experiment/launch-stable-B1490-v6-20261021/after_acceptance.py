"""Wait for the existing GPU verifier; freeze and hand off only on acceptance."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

LAUNCH=Path(__file__).resolve().parent
sys.path.insert(0,str(LAUNCH))
import run_controller as controller
ACCEPTANCE=controller.ACCEPTANCE


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def process_start_ticks(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    return raw[raw.rindex(')')+2:].split()[19]


def verifier_alive(pid,start):
    try:
        args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
        return process_start_ticks(pid)==start and any(Path(arg).name=='verify_gpu.py' for arg in args if arg)
    except (OSError,ValueError,IndexError):return False


def freeze_sources(launch,acceptance):
    launch,acceptance=Path(launch),Path(acceptance)
    receipt_path=launch/'cpu-acceptance.json'
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('status')!='passed':raise ValueError('CPU acceptance is not complete')
    locks=dict(receipt['source_sha256'])
    for path,expected in locks.items():
        if digest(path)!=expected:raise ValueError('CPU-tested source changed: '+path)
    runtime=json.loads((acceptance/'runtime-source-sha256.json').read_text())
    locks.update(runtime)
    for path in [receipt_path,*[acceptance/name for name in ('result.json','gpu.exit','runtime-source-sha256.json')]]:
        locks[str(path.resolve())]=digest(path)
    destination=launch/'source-sha256.json'
    if destination.exists():raise FileExistsError('Deployment has already been frozen')
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(locks,indent=2)+'\n');os.replace(temporary,destination)


def execute_controller(launch):
    os.execv(controller.PYTHON,[controller.PYTHON,'-u',str(Path(launch)/'run_controller.py')])


def run_after_acceptance(launch,acceptance,verifier_pid,verifier_start,*,
        alive=verifier_alive,sleep=time.sleep,verify=controller.verify_acceptance,
        freeze=freeze_sources,execute=execute_controller):
    launch,acceptance=Path(launch),Path(acceptance)
    with (launch/'handoff.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=controller.read(launch/'pipeline-state.json') if (launch/'pipeline-state.json').exists() else {}
        if previous.get('status') in ('failed','completed') or previous.get('stage')=='B':
            raise ValueError('Existing terminal/B queue requires explicit review; no automatic retry')
        try:
            while not (acceptance/'gpu.exit').exists():
                if not alive(verifier_pid,verifier_start):
                    # The shell writes gpu.exit immediately after Python exits.
                    sleep(1)
                    if (acceptance/'gpu.exit').exists():break
                    raise RuntimeError('GPU verifier exited without an exit record')
                controller.write_state(launch,status='waiting_gpu_acceptance',stage='verification',
                    model_pid=verifier_pid,acceptance=str(acceptance),auto_restart=False,
                    next_stage='B1490_full_state_fork',controller_pid=os.getpid())
                sleep(5)
            if int((acceptance/'gpu.exit').read_text())!=0:
                raise RuntimeError('GPU acceptance failed; B was not started')
            batch=verify(acceptance)
            freeze(launch,acceptance)
            controller.write_state(launch,status='acceptance_passed',stage='verification',
                evaluation_batch_size=batch,acceptance=str(acceptance),auto_restart=False,
                next_stage='B1490_full_state_fork',controller_pid=os.getpid())
            execute(launch)
            return 0
        except BaseException as error:
            controller.write_state(launch,status='failed',stage='verification',error=repr(error),
                later_stages_started=False,auto_restart=False,controller_pid=os.getpid())
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verifier-pid',type=int,required=True)
    parser.add_argument('--verifier-start',required=True)
    args=parser.parse_args()
    if 'TMUX' not in os.environ:raise RuntimeError('Run handoff inside tmux')
    run_after_acceptance(LAUNCH,ACCEPTANCE,args.verifier_pid,args.verifier_start)
