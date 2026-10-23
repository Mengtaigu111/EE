"""R0 must not change raw-prediction evaluation or the historical codec."""
import hashlib
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))


def test_protected_source_bytes_are_unchanged():
    evidence=HERE/'verification-r0-20261013/immutable-before.json'
    if not evidence.exists():pytest.skip('Local pre-change inventory unavailable')
    root=HERE.parents[1]
    for path,digest in json.loads(evidence.read_text()).items():
        # R0-v2 explicitly extends the training loss; its off-path is covered
        # by numerical loss/gradient equivalence tests instead of byte identity.
        if path=='tasks/action-experiment/grpo_core.py':continue
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path


def test_frozen_1627_raw_predictions_keep_identical_presence_metrics():
    from action_evaluation import score_predictions
    from action_common import read_rows
    predictions=HERE/'runs/action-sft-v2-42-20261012/test-best-20261013/predictions.jsonl'
    expected=HERE/'verification-r0-20261013/eval-before.json'
    data=HERE/'data/action-v2-goldorder-20261012/test.jsonl'
    if not all(p.exists() for p in (predictions,expected,data)):
        pytest.skip('Full local experiment artifacts unavailable')
    result=score_predictions(read_rows(data),read_rows(predictions))
    baseline=json.loads(expected.read_text())
    assert {k:result[k] for k in baseline}==baseline
