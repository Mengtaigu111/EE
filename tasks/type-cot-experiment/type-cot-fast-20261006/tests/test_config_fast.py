import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]

def load_config():
    spec=importlib.util.spec_from_file_location('fast_config',HERE/'config.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_phase_batch_validation_and_accumulation():
    c=load_config()
    assert c.phase_accumulation(1)==16
    assert c.phase_accumulation(2)==8
    for x in [0,3,4,True,'2',2.0]:
        with pytest.raises(ValueError):c.phase_accumulation(x)

def test_manifest_preserves_gold_course_and_allows_phase_specific_batch():
    c=load_config();m=c.build_manifest(cot_microbatch=1,tree_microbatch=2)
    assert m['cot_microbatch']==1 and m['tree_microbatch']==2
    assert m['cot_dev_diagnostic'] is False
    assert m['tree_reference']['microbatch']==1
    assert m['tree_eval_steps']==[401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208]
    c.validate_manifest(m)
    m['cot_microbatch']=4
    with pytest.raises(ValueError):c.validate_manifest(m)


def test_kernel_selection_and_dependency_source_locks():
    c=load_config()
    m=c.build_manifest(training_kernel='liger')
    assert m['training_kernel']=='liger'
    assert any('vendor/liger_kernel/' in str(p) for p in c.source_files())
    c.validate_manifest(m)
    m['training_kernel']='anything'
    with pytest.raises(ValueError):c.validate_manifest(m)
