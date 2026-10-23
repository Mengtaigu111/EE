"""Deliver an inspection request to the existing chat every two hours."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
LAUNCH = HERE.parent
ROOT = LAUNCH.parents[2]
CODEX = '/home/myluo/.local/bin/codex'
THREAD = '01a1545f-f450-7552-8cb5-c22e768ed86a'
INTERVAL_SECONDS = 7200
RETRY_SECONDS = 60
MESSAGE = '''这是用户已授权的每两小时自动巡检。请在 /home/myluo/EE/EXCEEDS 只读检查新 B 单阶段队列
tasks/action-experiment/launch-stable-B1490-v6-20261021/ 的 pipeline-state.json、pipeline.log、B.log，
首先按stage区分：若stage=verification，核查现有GPU验收进程及
tasks/action-experiment/verification-stable-v6-20261021/ 的resource-state.json、gpu.log、result.json、gpu.exit，
以及本队列handoff进程。此阶段新B目录和B.log尚不存在属于正常状态；只允许现有一个本项目验收模型，
不启动第二个。foreign GPU作业使其waiting是正常等待，不把其他用户进程算成本项目重复模型。
验收全部通过且来源冻结后，handoff才会串行启动B；验收失败只停队列，不自动重试。
若stage=B，再检查下述新run：
以及新 run tasks/action-experiment/runs/r0-v2-B-stable-v6-from1490-42-20261021/ 的训练进度、dev指标、
最新日志、failure.json、configuration.json、precision-transition.json 和最新完整恢复点。
新 B 是从旧 B1490 完整状态分叉的 v6 高精度数值协议，保留优化器/RNG/早停及历史最优step1200；
继承的旧BF16 dev指标不得冒充新协议重新测得的指标。核对新run来源锁及checkpoint manifest文件哈希。
同时核验旧 A run tasks/action-experiment/runs/r0-v2-A800-recoverable-v5-42-20261019/ 的 A1775恢复点，
旧 B run tasks/action-experiment/runs/r0-v2-B-recoverable-v5-42-20261019/ 的 B1490恢复点仍完整，旧 A/B没有重启。
核验 A下test-step800-after-A1775-20261021、test-step1200-after-A1775-20261021 输出完整性及指标仍在。
核对tmux/进程与GPU实际占用：当前串行阶段至多一个本项目模型进程，等待GPU时允许零个CUDA模型；监控调度器只能有一个实例。
旧A1775和旧B1490的probability guard故障是已知历史事件，不重复当作新增异常；不恢复A，不重启旧B，不改阈值。
本次只读核验生产文件；仅将检查摘要写入本新队列monitoring/inspection-state.json供下次比较。
结合该文件与本chat前次结果区分新增变化。保持已定数据、checkpoint、概率保护和数值协议，不启动额外模型，
不自动重启任何故障。只有新增异常、阶段完成、整个队列完成或确需用户操作时才通知用户；
状态无变化或仅正常训练进度变化时保持安静，不重复状态。pipeline completed时做最终完整性检查；
投递调度器将在成功投递该最终检查后退出。'''


def migrate_deadline(old_state_path, new_state_path, *, clock=time.time):
    """Deployment-only helper: preserve the old deadline after stopping its PID.

    This never starts/stops a scheduler and never enqueues a chat message.
    The caller must stop the old instance before starting the new one.
    """
    old = json.loads(Path(old_state_path).read_text())
    new_state_path = Path(new_state_path)
    if new_state_path.exists():
        raise FileExistsError('New monitor state already exists')
    now = clock()
    state = dict(status='scheduled', created_unix=now, updated_unix=now,
        next_check_unix=old.get('next_check_unix') or now + INTERVAL_SECONDS,
        last_attempt_unix=old.get('last_attempt_unix'),
        last_enqueued_unix=old.get('last_enqueued_unix'), returncode=None, stderr='',
        attempts=0, migrated_from=str(Path(old_state_path).resolve()),
        previous_attempts=old.get('attempts', 0))
    new_state_path.parent.mkdir(parents=True, exist_ok=True)
    write_state(new_state_path, state)
    return state


@contextmanager
def single_instance(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def enqueue_check():
    return subprocess.run(
        [CODEX, 'queue', '--thread', THREAD, '--message', MESSAGE],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False)


def write_state(path, state):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pipeline_completed(path):
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(value, dict) and value.get('status') == 'completed'


def run_scheduler(state_path, pipeline_path, *, enqueue=enqueue_check,
                  clock=time.time, sleep=time.sleep, steps=None):
    """Resume persisted deadlines; steps bounds delivery attempts for CPU tests."""
    state_path = Path(state_path)
    with single_instance(state_path.with_suffix('.lock')):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get('status') == 'completed':
                return
        else:
            now = clock()
            state = dict(status='scheduled', created_unix=now, updated_unix=now,
                         next_check_unix=now + INTERVAL_SECONDS,
                         last_attempt_unix=None, last_enqueued_unix=None,
                         returncode=None, stderr='', attempts=0)
        state['scheduler_pid'] = os.getpid()
        write_state(state_path, state)
        attempted = 0
        while steps is None or attempted < steps:
            remaining = state['next_check_unix'] - clock()
            if remaining > 0:
                sleep(min(remaining, 60))
                continue
            started = clock()
            try:
                result = enqueue()
                code = result.returncode
                stderr = result.stderr or ''
            except (OSError, subprocess.TimeoutExpired) as error:
                code = None
                stderr = f'{type(error).__name__}: {error}'
            finished = clock()
            success = code == 0
            completed = success and pipeline_completed(pipeline_path)
            state.update(status='completed' if completed else 'scheduled' if success else 'delivery_failed',
                         updated_unix=finished, last_attempt_unix=started,
                         returncode=code, stderr=stderr[-2048:],
                         next_check_unix=None if completed else finished + (INTERVAL_SECONDS if success else RETRY_SECONDS),
                         attempts=state['attempts'] + 1)
            if success:
                state['last_enqueued_unix'] = finished
            write_state(state_path, state)
            attempted += 1
            if completed:
                return


if __name__ == '__main__':
    run_scheduler(HERE / 'state.json', LAUNCH / 'pipeline-state.json')
