import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]))


def test_dense_plateau_reduces_2205_stops_2807_without_reset():
    from verify import replay_feedback
    steps=[401,802,1203,1403,1604,1804,2005,2205,2406,2606,2807]
    scores=[.12,.32,.39,.40,.41,.40,.40,.40,.4095,.40,.40]
    rows=[]
    best=None;anchor=None;bad=0;lr=.0001
    for i,(step,score) in enumerate(zip(steps,scores),1):
        new=best is None or score>=best
        if new:best=score;best_step=step
        if anchor is None or score-anchor>=.001-1e-12:anchor=score;bad=0
        else:bad+=1
        before=lr
        if bad==3:lr*=.5
        rows.append(dict(step=step,score=score,official={'AC_f1':score},lr_before=before,lr_after=lr,
                         best_score=best,best_step=best_step,anchor_score=anchor,bad_count=bad,
                         eval_count=i,last_eval_step=step,new_best=new,reduce_lr=bad==3,stop=bad>=6))
    actual=replay_feedback(rows,steps)
    assert actual['best_step']==1604 and actual['last_eval_step']==2807
    assert actual['bad_count']==6 and actual['stop']
    rows[3]['step']=1402
    with pytest.raises(ValueError,match='schedule'):replay_feedback(rows,steps)


def test_saved_lr_rejects_missing_optimizer_groups_and_wrong_base_rate():
    from verify import validate_saved_lr
    scheduler=dict(last_epoch=401,warmup_steps=97,plateau_scale=1.,base_lrs=[.0001,.0001],_last_lr=[.0001,.0001])
    optimizer=dict(param_groups=[dict(lr=.0001)])
    with pytest.raises(ValueError,match='group count'):
        validate_saved_lr(scheduler,optimizer,dict(lr_after=.0001),401,97)
    optimizer['param_groups'].append(dict(lr=.0001))
    scheduler['base_lrs'][0]=.0002
    with pytest.raises(ValueError,match='base LR'):
        validate_saved_lr(scheduler,optimizer,dict(lr_after=.0001),401,97)


def test_final_test_must_point_to_dev_selected_checkpoint(tmp_path):
    from verify import validate_test_metadata
    run=dict(best_checkpoint=str(tmp_path/'tree/checkpoints/checkpoint-4'),
             test_prediction=str(tmp_path/'missing.predictions.jsonl'))
    with pytest.raises(ValueError,match='dev-selected'):
        validate_test_metadata(tmp_path,run,{},2)
