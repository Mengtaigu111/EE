import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent


def runner():
    file=HERE/'run_user_v2_curriculum.py'
    assert file.exists(), 'separate UserV2 runner missing'
    spec=importlib.util.spec_from_file_location('user_v2_runner',file)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_curriculum_budget_and_dev_selection():
    r=runner()
    assert r.verify_budget(4010,2005,4010)==(2005,2005)
    assert r.checkpoint_steps(4010,2005,802)==[802,1604,2005,2406,3208,4010]
    assert r.select_checkpoint([{'step':802,'AC_f1':.4},{'step':2406,'AC_f1':.4}])['step']==2406
    with pytest.raises(ValueError): r.verify_budget(4010,2005,4011)


def test_new_report_labels_changed_prompt_and_uses_own_directory(tmp_path):
    r=runner()
    manifest=r.read(HERE/'mean_curriculum_manifest.json')
    manifest['prompt_template_file']=str(HERE.parents[2]/'task/prompt/1002-event-forest-v2.md')
    manifest['prompt_sha256']=r.sha(manifest['prompt_template_file'])
    manifest['comparison_limitation']='prompt and curriculum both changed'
    baseline=r.read(manifest['baseline_run_file'])
    best=next(x for x in baseline['dev_checkpoints'] if x['checkpoint']==baseline['best_checkpoint'])
    for phase,step in [(1,2005),(2,4010)]:
        folder=tmp_path/f'stage{phase}';folder.mkdir()
        (folder/'train_metrics.json').write_text(json.dumps({'global_step':step,'phase_train_loss':.1,
            'first_logged_loss':.2,'last_logged_loss':.1,'collated_examples':32080,'effective_collated_passes':2.5}))
    (tmp_path/'stage2/resume_evidence.json').write_text(json.dumps({'optimizer_exact':True}))
    state=dict(baseline_run=manifest['baseline_run_file'],manifest=str(tmp_path/'manifest.json'),
        stage1_dev_metrics=best['metrics'],final_dev_metrics=best['metrics'],best_dev_metrics=best['metrics'],
        best_checkpoint=best['checkpoint'],best_step=2406,dev_checkpoints=[{}]*6,test_official=baseline['test_official'])
    r.report(tmp_path,state,manifest,publish=False)
    text=(tmp_path/'mean_curriculum_training_report.md').read_text()
    assert 'UserV2' in text and 'prompt 与 curriculum 同时变化' in text
    assert 'Stage2 JSONL 与 baseline train 字节一致' not in text
    assert '40.9561' in text and 'ΔAC=+0.0000' in text
