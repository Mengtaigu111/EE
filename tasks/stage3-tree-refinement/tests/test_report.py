import json


def test_report_reads_only_json_artifacts_and_states_v1_omissions(tmp_path):
    from write_report import render_report

    (tmp_path / 'metrics.json').write_text(json.dumps({'AC_f1': 0.4}), encoding='utf-8')
    (tmp_path / 'notes.txt').write_text('not an input', encoding='utf-8')
    report = render_report(tmp_path)
    assert 'metrics.json' in report
    assert 'notes.txt' not in report
    assert 'R2 Tree Refiner' in report


def test_report_collects_artifact_metadata_and_renders_required_sections(tmp_path):
    from write_report import collect_artifacts, render_report

    (tmp_path / 'd1-dev.summary.json').write_text(json.dumps({
        'split': 'dev', 'trigger_recall': 0.4, 'diversity': 3,
    }), encoding='utf-8')
    (tmp_path / 'r0.tau.metrics.json').write_text(json.dumps({
        'tau': 0.7, 'AC_f1': 0.42,
    }), encoding='utf-8')
    artifacts = collect_artifacts(tmp_path)
    assert artifacts[0]['path'] == 'd1-dev.summary.json'
    assert len(artifacts[0]['sha256']) == 64
    assert artifacts[0]['json']['trigger_recall'] == 0.4

    report = render_report(tmp_path)
    for heading in (
        '## D1', '## D2', '## F0/F1', '## Reward model',
        '## R1 / R0 / R01 / RB', '## Tau scan', '## Complex test',
        '## Baselines and comparison', '## v1 omissions',
    ):
        assert heading in report
    assert 'd1-dev.summary.json' in report
    assert 'r0.tau.metrics.json' in report


def test_report_rejects_invalid_json_artifacts(tmp_path):
    from write_report import collect_artifacts

    (tmp_path / 'broken.json').write_text('{', encoding='utf-8')
    try:
        collect_artifacts(tmp_path)
    except ValueError as exc:
        assert 'invalid JSON artifact' in str(exc)
    else:
        raise AssertionError('expected invalid JSON failure')
