import importlib.util
from pathlib import Path

HERE = Path(__file__).parents[1]


def module():
    spec = importlib.util.spec_from_file_location('cot_online_config', HERE/'config.py')
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_schedule_preserves_original_and_inserts_midpoints():
    m = module()
    assert m.EVAL_STEPS == [401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208]
    assert set(range(401,3209,401)).issubset(m.EVAL_STEPS)


def test_manifest_uses_fresh_cot_and_no_historical_init():
    m = module()
    value = m.build_manifest()
    assert value['cot_steps'] == 802
    assert value['cot_warmup_steps'] == 25
    assert value['cot_init_adapter'] is None
    assert value['tree_reference']['init_adapter'] is None
    assert value['tree_eval_steps'] == m.EVAL_STEPS
    assert 'user-final-20261005' in value['cot_data_dir']
