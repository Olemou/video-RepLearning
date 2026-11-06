import copy
import torch
import pytest
from vision_model.training.trajectory import TrajectoryModel,trajectory_loss
from vision_model.training.relative_clock import RelativeClockTrajectoryModel,trajectory_model_for_clock


def test_relative_clock_is_origin_invariant_and_preserves_real_delta_loss():
    torch.set_num_threads(1);torch.manual_seed(239)
    model=RelativeClockTrajectoryModel(12,hidden=16,embedding_dim=8)
    teacher=torch.randn(4,2,6,12)
    valid=torch.ones(4,6,dtype=torch.bool);valid[0,4:]=False
    times=torch.arange(6)[None].expand(4,-1).float()*.5
    shifted=times+torch.tensor([17.,62.,125.,-.5])[:,None]
    times=times.masked_fill(~valid,float('nan'));shifted=shifted.masked_fill(~valid,float('nan'))
    labels=torch.tensor([0,0,1,1])
    loss,metrics,out=trajectory_loss(model,teacher,valid,times,labels,epoch=20)
    other,other_metrics,other_out=trajectory_loss(model,teacher,valid,shifted,labels,epoch=20)
    torch.testing.assert_close(loss,other,rtol=0,atol=0)
    for k in out:torch.testing.assert_close(out[k],other_out[k],rtol=0,atol=0)
    torch.testing.assert_close(metrics['prediction_loss'],other_metrics['prediction_loss'],rtol=0,atol=0)
    assert torch.isfinite(loss)


def test_absolute_default_and_relative_strict_checkpoint_compatibility():
    torch.manual_seed(7);old=TrajectoryModel(12,hidden=16,embedding_dim=8)
    torch.manual_seed(7);default=trajectory_model_for_clock(12,hidden=16,embedding_dim=8)
    for key,value in old.state_dict().items():
        torch.testing.assert_close(value,default.state_dict()[key],rtol=0,atol=0)
    relative=trajectory_model_for_clock(12,'relative_prefix',hidden=16,embedding_dim=8)
    relative.load_state_dict(copy.deepcopy(old.state_dict()),strict=True)
    x=torch.randn(2,4,12);v=torch.ones(2,4,dtype=torch.bool);t=torch.arange(4).expand(2,-1).float()*.5
    torch.testing.assert_close(relative(x,v,t)['embedding'],old(x,v,t)['embedding'],rtol=0,atol=0)
    with pytest.raises(ValueError):trajectory_model_for_clock(12,'unknown')
