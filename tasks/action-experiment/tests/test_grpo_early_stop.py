from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_grpo import DevEarlyStop, audit_training_prefix, build_parser, validate_options


def test_three_dev_failures_stop_without_counting_initial_dev():
    control=DevEarlyStop(3)
    assert not control.update(0,.60)['stop']
    assert control.update(400,.6004)['bad_count']==1
    assert control.update(800,.599)['bad_count']==2
    assert control.update(1200,.6008)['stop']
    assert control.state_dict()['last_step']==1200


def test_meaningful_improvement_resets_anchor_and_disabled_keeps_old_behavior():
    control=DevEarlyStop(3)
    control.update(0,.60);control.update(400,.60)
    assert control.update(800,.601)['bad_count']==0
    assert not control.update(1200,.6014)['stop']
    disabled=DevEarlyStop(None)
    for step in range(10):assert not disabled.update(step,.5)['stop']
    with pytest.raises(ValueError):control.update(1200,.6)
    with pytest.raises(ValueError):DevEarlyStop(0)


def test_cli_early_stop_is_opt_in():
    args=build_parser().parse_args(['--init-adapter','x','--output','y'])
    assert args.early_stopping_patience is None
    args.early_stopping_patience=0
    with pytest.raises(ValueError):validate_options(args)


def test_prefix_audit_accepts_early_stop_but_never_claims_unseen_rows():
    train=[{'id':str(i),'n_events':int(i%3!=0)} for i in range(12)]
    observed=[{'id':r['id']} for r in train[:8]]
    result=audit_training_prefix(observed,train,True,2,4)
    assert result==train[:8]
    with pytest.raises(RuntimeError):audit_training_prefix(observed,train,False,2,4)
    with pytest.raises(RuntimeError):audit_training_prefix(observed[::-1],train,True,2,4)
    with pytest.raises(RuntimeError):audit_training_prefix(observed[:7],train,True,2,4)
