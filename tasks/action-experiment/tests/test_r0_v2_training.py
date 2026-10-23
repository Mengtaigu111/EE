import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_grpo as training


def test_version_combinations_and_old_default():
    parser=training.build_parser()
    args=parser.parse_args(['--init-adapter','sft','--output','out'])
    assert args.reward_version=='event_link_token_exact_v1'
    assert args.policy_update_mask=='off' and args.max_steps is None
    args.policy_update_mask='field_v2'
    with pytest.raises(ValueError):training.validate_options(args)
    args.reward_version='r0_v2'
    training.validate_options(args)


def test_short_budget_does_not_repeat_and_includes_tail():
    rows=list(range(13))
    assert training.limit_training_rows(rows,2,4)==list(range(8))
    assert training.limit_training_rows(rows,4,4)==rows
    with pytest.raises(ValueError):training.limit_training_rows(rows,5,4)


def test_component_logging_ignores_na_but_total_keeps_empty_and_invalid():
    result=training.reward_component_summary([
        {'components':{'trigger':1.,'type':0.,'argument':1.,'role':1.}},
        {'components':dict.fromkeys(['trigger','type','argument','role'])}])
    assert result['reward_trigger_f1']==1.
    assert result['reward_trigger_f1_count']==1


def test_cumulative_mask_rates_weight_tokens_and_exclude_zero_and_fallback():
    stats=training.TrainingDiagnostics('field_v2')
    def reward(value):return {'reward':value,'components':{'trigger':value,'type':None,'argument':None,'role':None}}
    def mask(sign,n,k,reason=None):return dict(advantage_sign=sign,data_token_count=n,kept_data_token_count=k,fallback_reason=reason,mixed_boundary_token_count=2)
    for r,m in [(reward(1),mask(1,10,9)),(reward(0),mask(1,2,0)),(reward(0),mask(0,20,0)),(reward(0),mask(-1,3,2,'token_ids_mismatch'))]:stats.observe(r,m)
    summary=stats.summary()
    assert summary['positive_data_keep_ratio']==.75
    assert summary['negative_data_keep_ratio'] is None
    assert summary['mask_fallback_rate']==.25
    assert summary['reward_r0_v2']==.25
    assert summary['reward_trigger_f1']==.25
