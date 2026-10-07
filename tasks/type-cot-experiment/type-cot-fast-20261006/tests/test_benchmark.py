import importlib.util
from pathlib import Path
import torch

spec=importlib.util.spec_from_file_location('fast_benchmark',Path(__file__).parents[1]/'benchmark.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_fixed_groups_include_longest_stress_and_repeatable():
    lengths=list(range(200))
    indices=module.select_trial_indices(lengths)
    assert len(indices)==160
    assert indices==module.select_trial_indices(lengths)
    assert set(indices[32:64])==set(range(168,200))
    assert module.update_groups(indices)==[indices[i:i+16] for i in range(0,160,16)]

def test_decision_rejects_unstable_or_tight_memory():
    baseline=dict(status='completed',seconds_per_update=10.,finite_loss=True,finite_gradients=True,longest_stress_passed=True,memory_free_at_peak=1000000000)
    candidate={**baseline,'seconds_per_update':9.}
    assert module.choose_microbatch(baseline,candidate)['microbatch']==2
    assert module.choose_microbatch(baseline,{**candidate,'memory_free_at_peak':100})['microbatch']==1
    assert module.choose_microbatch(baseline,{**candidate,'finite_gradients':False})['microbatch']==1
    assert module.choose_microbatch(baseline,{**candidate,'seconds_per_update':9.8})['microbatch']==1

def test_assistant_token_weighted_gradient_invariant_to_microbatch():
    torch.manual_seed(42)
    features=torch.randn(16,9,5,dtype=torch.float64)
    labels=torch.randint(0,7,(16,9));labels[:,:3]=-100
    for i in range(16):labels[i,3:3+i%5]=-100
    count=(labels[:,1:]!=-100).sum()
    original=torch.randn(5,7,dtype=torch.float64)
    results=[]
    for microbatch in (1,2,16):
        weights=original.clone().requires_grad_();total=0.
        for start in range(0,16,microbatch):
            loss=module.assistant_token_loss(features[start:start+microbatch]@weights,labels[start:start+microbatch],count)
            total+=loss.detach();loss.backward()
        results.append((total,weights.grad))
    for loss,grad in results[1:]:
        torch.testing.assert_close(loss,results[0][0],rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(grad,results[0][1],rtol=1e-12,atol=1e-12)
