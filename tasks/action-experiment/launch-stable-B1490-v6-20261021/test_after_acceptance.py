import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('stable_after',Path(__file__).with_name('after_acceptance.py'))
after=importlib.util.module_from_spec(spec);spec.loader.exec_module(after)


def test_waits_for_existing_verifier_then_starts_once(tmp_path):
    launch=tmp_path/'launch';launch.mkdir();acceptance=tmp_path/'accept';acceptance.mkdir()
    events=[]
    def sleep(_):
        events.append('wait');(acceptance/'gpu.exit').write_text('0\n')
    result=after.run_after_acceptance(launch,acceptance,123,'start',
        alive=lambda *_:True,sleep=sleep,verify=lambda _:4,
        freeze=lambda *_:events.append('freeze'),execute=lambda *_:events.append('controller'))
    assert result==0 and events==['wait','freeze','controller']


def test_failed_acceptance_never_starts_or_freezes(tmp_path):
    launch=tmp_path/'launch';launch.mkdir();acceptance=tmp_path/'accept';acceptance.mkdir()
    (acceptance/'gpu.exit').write_text('1\n')
    with pytest.raises(RuntimeError,match='GPU acceptance failed'):
        after.run_after_acceptance(launch,acceptance,123,'start',
            freeze=lambda *_:pytest.fail('must not freeze'),execute=lambda *_:pytest.fail('must not launch'))
    state=json.loads((launch/'pipeline-state.json').read_text())
    assert state['status']=='failed' and state['stage']=='verification'


def test_dead_verifier_without_exit_is_failure_not_retry(tmp_path):
    launch=tmp_path/'launch';launch.mkdir();acceptance=tmp_path/'accept';acceptance.mkdir()
    with pytest.raises(RuntimeError,match='exited without'):
        after.run_after_acceptance(launch,acceptance,123,'start',alive=lambda *_:False,
            execute=lambda *_:pytest.fail('must not start'))


def test_changed_sources_cannot_start_b(tmp_path):
    launch=tmp_path/'launch';launch.mkdir();acceptance=tmp_path/'accept';acceptance.mkdir()
    (acceptance/'gpu.exit').write_text('0\n')
    def verify(_):raise ValueError('changed runtime source')
    with pytest.raises(ValueError,match='runtime'):
        after.run_after_acceptance(launch,acceptance,123,'start',verify=verify,
            freeze=lambda *_:pytest.fail('must not freeze'),execute=lambda *_:pytest.fail('must not start'))


def test_freeze_requires_cpu_receipt_matching_current_sources(tmp_path):
    launch=tmp_path/'launch';launch.mkdir();acceptance=tmp_path/'accept';acceptance.mkdir()
    source=tmp_path/'code.py';source.write_text('old')
    receipt=launch/'cpu-acceptance.json'
    receipt.write_text(json.dumps({'status':'passed','source_sha256':{str(source):after.digest(source)}}))
    source.write_text('new')
    with pytest.raises(ValueError,match='CPU-tested source changed'):
        after.freeze_sources(launch,acceptance)
