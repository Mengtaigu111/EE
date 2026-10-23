import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grpo_core import grpo_loss


def tensors():
    current=torch.tensor([[-.8,-.9,-1.2],[-1.1,-.6,-1.]],requires_grad=True)
    return current,torch.full((2,3),-1.),torch.full((2,3),-.95),torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool),torch.tensor([1.,-1.])


def test_off_and_ones_have_identical_loss_and_gradient():
    c,o,r,m,a=tensors()
    baseline,stats=grpo_loss(c,o,r,m,a)
    masked,other=grpo_loss(c,o,r,m,a,field_mask=torch.ones_like(c))
    assert torch.equal(baseline,masked)
    assert stats==other
    assert torch.equal(torch.autograd.grad(baseline,c,retain_graph=True)[0],torch.autograd.grad(masked,c)[0])


def test_only_policy_numerator_masked_with_original_denominator():
    c,o,r,m,a=tensors(); keep=torch.tensor([[0.,1.,0.],[1.,0.,1.]])
    loss,stats=grpo_loss(c,o,r,m,a,field_mask=keep)
    ratio=(c-o).exp();surrogate=torch.minimum(ratio*a[:,None],ratio.clamp(.8,1.2)*a[:,None])
    kl=(r-c).exp()-(r-c)-1
    expected=((.01*kl-surrogate*keep)*m).sum(-1)/m.sum(-1)
    assert torch.allclose(loss,expected.mean())
    assert stats['kl']==grpo_loss(c,o,r,m,a)[1]['kl']
    policy,_=grpo_loss(c,o,r,m,a,beta=0.,field_mask=keep)
    grad=torch.autograd.grad(policy,c)[0]
    assert (grad[keep==0]==0).all()


def test_mask_detached_and_bad_masks_rejected():
    c,o,r,m,a=tensors(); keep=torch.ones_like(c,requires_grad=True)
    loss,_=grpo_loss(c,o,r,m,a,field_mask=keep)
    assert torch.autograd.grad(loss,keep,allow_unused=True)[0] is None
    for bad in [torch.ones(1,3),torch.full_like(c,float('nan')),torch.full_like(c,.5)]:
        with pytest.raises(ValueError):grpo_loss(c,o,r,m,a,field_mask=bad)
