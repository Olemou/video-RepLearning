import pytest
import torch

from tests.test_online_trajectory import example
from vision_model.training.online_trajectory import explicit_prefix_forward
from vision_model.interpretability.attention_competition import attention_logit_gradient, explain_attention_competition
from vision_model.interpretability.online_trajectory import explain_prefix_group


def test_logit_gradient_matches_autograd_and_valid_logit_finite_difference():
    torch.manual_seed(239)
    logits = torch.randn(2,3,5,5,dtype=torch.float64,requires_grad=True)
    a = logits.softmax(-1)
    value = torch.randn_like(a)
    scalar = ((a*value).sum(-1).sin()).sum()
    g, expected = torch.autograd.grad(scalar,(a,logits))
    actual = attention_logit_gradient(a.detach(),g)
    torch.testing.assert_close(actual,expected,atol=1e-14,rtol=1e-12)
    torch.testing.assert_close(actual.sum(-1),torch.zeros_like(actual.sum(-1)),atol=1e-14,rtol=0)
    direction = torch.randn_like(logits)
    def score(x):
        return ((x.softmax(-1)*value).sum(-1).sin()).sum()
    epsilon=1e-5
    difference=(score(logits.detach()+epsilon*direction)-score(logits.detach()-epsilon*direction))/(2*epsilon)
    torch.testing.assert_close((actual*direction).sum(),difference,atol=1e-8,rtol=1e-7)


def test_row_constant_gradient_is_removed_without_changing_valid_attention_scores():
    torch.manual_seed(17)
    logits=torch.randn(2,4,7,7,dtype=torch.float64,requires_grad=True)
    a=logits.softmax(-1)
    g=torch.randn_like(a)
    offset=torch.randn(2,4,7,1,dtype=torch.float64)*100
    base=(a*g).sum()
    extension=base+((a.sum(-1,keepdim=True)-1)*offset).sum()
    torch.testing.assert_close(base,extension,atol=1e-10,rtol=1e-10)
    assert not torch.allclose(g.clamp_min(0), (g+offset).clamp_min(0))
    torch.testing.assert_close(attention_logit_gradient(a,g),attention_logit_gradient(a,g+offset),atol=1e-13,rtol=1e-11)
    assert attention_logit_gradient(a,torch.ones_like(g)).abs().max()<1e-14


def test_prefix_replay_preserves_old_gradients_and_exact_requested_scores():
    torch.set_num_threads(1)
    encoder, model, pixels, _, valid, times, _ = example(2)
    encoder.requires_grad_(False); model.requires_grad_(False)
    reference=torch.stack([explicit_prefix_forward(encoder.encoder,
        pixels[:,:,:2*(t+1)].flatten(0,1).permute(0,2,1,3,4),capture=False)[0].reshape(2,2,-1)
        for t in range(3)],2).detach()
    def score(f):
        out=model(f.flatten(0,1),valid[:,None].expand(-1,2,-1).flatten(0,1),
            times[:,None].expand(-1,2,-1).flatten(0,1),detach_teacher=False)
        return out['embedding'].reshape(2,2,-1)[:,0,0]
    old=explain_prefix_group(encoder.encoder,pixels,score,torch.device('cpu'))
    new=explain_attention_competition(encoder.encoder,pixels,reference,score,torch.device('cpu'))
    torch.testing.assert_close(new['probability_positive'],old['positive_gradients'],rtol=0,atol=0)
    torch.testing.assert_close(new['scores'],old['explicit_scores'],rtol=0,atol=0)
    assert new['explicit_replay_max_error']==0
    assert torch.isfinite(new['logit_signed']).all()
    assert new['logit_positive'].sum()>0
    assert new['logit_positive'].shape==(2,2,3,2,2)


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError):
        attention_logit_gradient(torch.ones(3,3),torch.ones(2,3))
