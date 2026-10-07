import importlib.util
from pathlib import Path


def test_cot_first_epoch_is_random_complete_reproducible():
    path=Path(__file__).parents[1]/'train_cot.py'
    spec=importlib.util.spec_from_file_location('cot_stage_training',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a=list(module.ShuffledSampler(32,42))
    b=list(module.ShuffledSampler(32,42))
    assert a==b
    assert sorted(i for e,i in a)==list(range(32))
    assert [i for e,i in a]!=list(range(32))
    assert all(e==0 for e,i in a)
