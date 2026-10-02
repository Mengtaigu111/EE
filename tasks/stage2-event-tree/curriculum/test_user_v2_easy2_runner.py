import importlib.util
import json
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent


def module():
    file=HERE/'run_user_v2_easy2_curriculum.py'
    assert file.exists(),'new Easy2 runner missing'
    spec=importlib.util.spec_from_file_location('easy2_runner',file)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_unequal_budget_and_exact_epoch_boundary():
    r=module()
    assert r.verify_budget(4010,814,4010)==(814,3196)
    assert r.checkpoint_steps(4010,814,802)==[802,814,1604,2406,3208,4010]
    m={'stage1_total_count':6504,'stage1_optimizer_steps':814,'stage2_optimizer_steps':3196,'stage1_epoch_target':2}
    cfg={'microbatch':1,'gradient_accumulation':16}
    assert r.validate_schedule(m,cfg,4010)==814
    with pytest.raises(ValueError):r.validate_schedule({**m,'stage1_optimizer_steps':2005},cfg,4010)
    with pytest.raises(ValueError):r.verify_budget(4010,814,4009)
    with pytest.raises(ValueError):r.verify_budget(4010,4010,4010)


def test_smoke_budget_is_unequal_too():
    assert module().smoke_schedule()==(6,2,2)


def test_report_budget_and_variant_are_accurate(tmp_path):
    r=module();manifest=r.read(HERE/'user-v2-20261002/mean_curriculum_manifest.json')
    baseline=r.read(manifest['baseline_run_file']);best=next(x for x in baseline['dev_checkpoints'] if x['checkpoint']==baseline['best_checkpoint'])
    for phase,step in [(1,814),(2,4010)]:
        out=tmp_path/f'stage{phase}';out.mkdir()
        (out/'train_metrics.json').write_text(json.dumps({'global_step':step,'phase_train_loss':.1,'first_logged_loss':.1,
            'last_logged_loss':.1,'collated_examples':13008 if phase==1 else 51124,'effective_collated_passes':2 if phase==1 else 3.985}))
    (tmp_path/'stage2/resume_evidence.json').write_text(json.dumps({'optimizer_exact':True}))
    state=dict(baseline_run=manifest['baseline_run_file'],manifest=str(tmp_path/'manifest.json'),stage1_dev_metrics=best['metrics'],
        final_dev_metrics=best['metrics'],best_dev_metrics=best['metrics'],best_checkpoint=best['checkpoint'],best_step=2406,
        dev_checkpoints=[{}]*6,test_official=baseline['test_official'])
    r.report(tmp_path,state,manifest,publish=False)
    text=(tmp_path/'mean_curriculum_training_report.md').read_text()
    assert 'Easy2' in text and 'Stage1=814; Stage2=3196' in text
    assert 'prompt 与 curriculum 同时变化' in text
