import torch
from vision_model.training.normal_spatial_student import correspondence_loss


def test_correspondence_prefers_correct_locations_and_detaches_teacher():
    target=torch.eye(4)[None].requires_grad_();query=target.detach().clone().requires_grad_()
    good,_=correspondence_loss(query,target)
    bad,_=correspondence_loss(query,target.roll(1,1))
    assert good<bad
    good.backward();assert query.grad is not None and target.grad is None


def test_spatial_contrast_penalizes_uniform_map_even_with_cosine_fit():
    target=torch.eye(4)[None]
    uniform=torch.ones_like(target)
    collapsed,detail=correspondence_loss(uniform,target)
    matched,_=correspondence_loss(target,target)
    assert collapsed>matched and detail['spatial_contrast_loss']>1
