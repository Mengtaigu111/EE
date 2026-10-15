import importlib.util
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]
def load_config():
    spec=importlib.util.spec_from_file_location("goldorder_config",HERE/"config.py")
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);return c
@pytest.mark.parametrize("seed",[42,66,58])
def test_seed_recipe_and_corrected_gold(seed):
    c=load_config();m=c.build_manifest(seed)
    assert m["seed"]==m["tree_reference"]["seed"]==seed
    assert m["experiment"].endswith(f"Seed{seed}")
    assert "gold-order-20261008" in m["tree_reference"]["data_dir"]
    assert "sorted-first-gold-order-20261008" in m["tree_reference"]["train_file"]
    assert m["cot_steps"]==802 and m["cot_warmup_steps"]==25
    assert m["tree_steps"]==3208 and len(m["tree_eval_steps"])==13
    assert not any("re-positive" in p for p in m["frozen_hashes"])
    c.validate_manifest(m)
    m["tree_reference"]["seed"]=52
    with pytest.raises(ValueError):c.validate_manifest(m)
def test_reject_unplanned_seed():
    with pytest.raises(ValueError):load_config().build_manifest(52)
def test_presence_in_full_pipeline():
    text=(HERE/"run.py").read_text()
    assert "score_tree_presence.py" in text and "text_presence" in text
    assert "from run import test_diagnostics" in (HERE/"verify.py").read_text()

def test_full_diagnostics_preserves_historical_modes():
    import sys
    sys.path.insert(0,str(HERE))
    import run
    c=load_config();m=c.build_manifest(42)
    prior=c.read(c.PROJECT/'tasks/type-cot-experiment/runs/C-CoT-REPositive-Once-GoldOrder-42-20261008/run.json')
    result=run.test_diagnostics(m['tree_reference'],prior['test_prediction'],prior['test_merged'])
    assert {'official_offset','official_raw_token','instance_offset','instance_raw_token','text_presence'} <= set(result)
    assert len(result['text_presence']['metrics'])==18
    assert result['text_presence']==prior['test_scores']['text_presence']
