import importlib.util
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent


def module():
    path = HERE / 'run_c_mean_curriculum.py'
    assert path.exists(), 'curriculum runner not implemented'
    spec = importlib.util.spec_from_file_location('curriculum_runner', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_checkpoint_selection_uses_dev_ac_later_step_ties():
    results = [dict(step=802, AC_f1=.4), dict(step=2005, AC_f1=.41),
               dict(step=4010, AC_f1=.40), dict(step=2406, AC_f1=.41)]
    assert module().select_checkpoint(results)['step'] == 2406


def test_budget_verification_rejects_weights_only_or_extra_steps():
    mod = module()
    assert mod.verify_budget(4010, 2005, 4010) == (2005, 2005)
    with pytest.raises(ValueError):
        mod.verify_budget(4010, 2005, 6015)
    with pytest.raises(ValueError):
        mod.verify_budget(4010, 2004, 4010)


def test_checkpoint_cadence_matches_baseline_and_includes_boundaries():
    assert module().checkpoint_steps(4010, 2005, 802) == [802, 1604, 2005, 2406, 3208, 4010]


def test_complete_report_renders_against_actual_manifest(tmp_path):
    import json
    mod = module()
    manifest = mod.read(HERE / 'mean_curriculum_manifest.json')
    baseline = mod.read(manifest['baseline_run_file'])
    best = next(x for x in baseline['dev_checkpoints'] if x['checkpoint'] == baseline['best_checkpoint'])
    for phase, step in ((1, 2005), (2, 4010)):
        folder = tmp_path / f'stage{phase}'
        folder.mkdir()
        (folder / 'train_metrics.json').write_text(json.dumps({
            'global_step': step, 'phase_train_loss': .1, 'first_logged_loss': .2,
            'last_logged_loss': .1, 'collated_examples': 32080, 'effective_collated_passes': 2.5}))
    (tmp_path / 'stage2/resume_evidence.json').write_text(json.dumps({'optimizer_exact': True}))
    state = dict(baseline_run=manifest['baseline_run_file'], stage1_dev_metrics=best['metrics'],
        final_dev_metrics=best['metrics'], best_dev_metrics=best['metrics'],
        best_checkpoint=best['checkpoint'], best_step=2406,
        dev_checkpoints=[{}]*6, test_official=baseline['test_official'])
    mod.report(tmp_path, state, manifest, publish=False)
    output = (tmp_path / 'mean_curriculum_training_report.md').read_text()
    assert '40.9561' in output
    assert '16.557' in output
    assert 'ΔAC=+0.0000' in output
