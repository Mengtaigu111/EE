import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]

def load_config():
    spec=importlib.util.spec_from_file_location('seed52_config',HERE/'config.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_seed52_and_original_recipe():
    c=load_config();m=c.build_manifest()
    assert m['seed']==52
    assert m['cot_dev_diagnostic'] is False
    assert m['tree_reference']['seed']==42
    assert m['tree_reference']['microbatch']==1 and m['tree_reference']['gradient_accumulation']==16
    assert m['tree_eval_steps']==[401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807,3007,3208]
    assert not any('vendor' in str(p) or 'type-cot-fast' in str(p) for p in c.source_files())
    c.validate_manifest(m)
    m['seed']=42
    with pytest.raises(ValueError):c.validate_manifest(m)
