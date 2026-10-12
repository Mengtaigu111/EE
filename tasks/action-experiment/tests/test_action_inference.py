import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tasks/action-experiment'))
from evaluate_adapter import selected_adapter
from action_common import sha256


def test_only_completed_selected_adapter_allowed(tmp_path):
    run=tmp_path/'sft';run.mkdir();adapter=run/'best-adapter';adapter.mkdir()
    weight=adapter/'adapter_model.safetensors';weight.write_bytes(b'selected')
    config=adapter/'adapter_config.json';config.write_text('{}')
    metrics={'status':'completed','stage':'action_sft','smoke':False,'best_adapter_sha256':sha256(weight),'best_adapter_config_sha256':sha256(config)}
    (run/'train_metrics.json').write_text(json.dumps(metrics))
    assert selected_adapter(run)==adapter
    weight.write_bytes(b'changed')
    with pytest.raises(ValueError):selected_adapter(run)


def test_smoke_cannot_be_reported_as_formal(tmp_path):
    adapter=tmp_path/'best-adapter';adapter.mkdir()
    weight=adapter/'adapter_model.safetensors';weight.write_bytes(b'weights')
    config=adapter/'adapter_config.json';config.write_text('{}')
    (tmp_path/'train_metrics.json').write_text(json.dumps({'status':'completed','stage':'action_grpo',
          'smoke':True,'best_adapter_sha256':sha256(weight),'best_adapter_config_sha256':sha256(config)}))
    with pytest.raises(ValueError,match='smoke'):selected_adapter(tmp_path)
    assert selected_adapter(tmp_path,allow_smoke=True)==adapter


def test_selected_lora_config_cannot_change_with_same_weights(tmp_path):
    from train_grpo import check_sft_source
    from action_common import read_json
    adapter=tmp_path/'best-adapter';adapter.mkdir()
    weights=adapter/'adapter_model.safetensors';weights.write_bytes(b'weights')
    config=adapter/'adapter_config.json';config.write_text('{"lora_alpha":32}')
    (tmp_path/'train_metrics.json').write_text(json.dumps({'status':'completed','stage':'action_sft',
        'smoke':False,'best_adapter_sha256':sha256(weights),'best_adapter_config_sha256':sha256(config)}))
    config.write_text('{"lora_alpha":64}')
    with pytest.raises(ValueError,match='config'):selected_adapter(tmp_path)
    with pytest.raises(ValueError,match='config'):check_sft_source(adapter,read_json,sha256,False)
