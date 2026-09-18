import json


def test_report_reads_only_json_artifacts_and_states_v1_omissions(tmp_path):
    from write_report import render_report

    (tmp_path / 'metrics.json').write_text(json.dumps({'AC_f1': 0.4}), encoding='utf-8')
    (tmp_path / 'notes.txt').write_text('not an input', encoding='utf-8')
    report = render_report(tmp_path)
    assert 'metrics.json' in report
    assert 'notes.txt' not in report
    assert 'R2 Tree Refiner' in report
