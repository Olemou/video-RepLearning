"""Causal geometry, valid support and global spatial-objective gradients."""
import copy
import torch
import pytest

from vision_model.training.spatial_semantic import (SpatialSemanticBranch,
    mirror_total_variation,attention_diagnostics,spatial_objective)
from vision_model.training.trajectory import TrajectoryModel,trajectory_loss


def case():
    torch.manual_seed(72)
    model=SpatialSemanticBranch(torch.randn(2,8),width=16,hidden=16,embedding_dim=8)
    x=torch.randn(4,2,4,4,16)
    valid=torch.tensor([[1,1]]*4,dtype=torch.bool)
    content=torch.ones(4,4,4,dtype=torch.bool)
    content[:,:,0]=False
    return model,x,valid,content


def test_attention_consumes_values_and_is_label_independent_and_connected():
    model,x,valid,content=case()
    out=model(x,valid,content)
    assert out['embedding'].shape==(4,8)
    torch.testing.assert_close(out['attention'].sum((-2,-1)),torch.ones(4,2,2))
    assert out['attention'][...,0].count_nonzero()==0
    grad=torch.autograd.grad(out['embedding'][:,0].sum(),out['head_attention'],allow_unused=True)[0]
    assert grad is not None and grad.abs().sum()>0
    out=model(x,valid,content)
    mapgrad=torch.autograd.grad(out['embedding'][:,0].sum(),out['attention'])[0]
    assert mapgrad.abs().sum()>0
    out=model(x,valid,content)
    out['embedding'][:,0].sum().backward()
    assert model.query.weight.grad.abs().sum()>0
    assert model.class_projection.weight.grad.abs().sum()>0
    assert model.class_prototypes.grad is None


def test_padding_and_future_times_never_change_earlier_maps():
    model,x,valid,content=case()
    valid[:,1]=False
    a=model(x,valid,content)
    altered=x.clone()
    altered[:,1]=float('nan')
    altered[:,:,:,0]=float('nan')
    b=model(altered,valid,content)
    torch.testing.assert_close(a['embedding'],b['embedding'])
    torch.testing.assert_close(a['attention'],b['attention'])
    assert b['attention'][:,1].count_nonzero()==0
    # Even with both times valid, spatial maps at time zero never consume one.
    valid[:]=True
    original=model(x,valid,content)['attention'][:,0]
    x[:,1]+=100
    changed=model(x,valid,content)['attention'][:,0]
    torch.testing.assert_close(original,changed)


def test_true_mirror_alignment_and_both_symmetric_gradients():
    torch.manual_seed(73)
    a=torch.rand(3,2,2,4,4).softmax(-1)
    a=a/a.sum((-2,-1),keepdim=True)
    pair=torch.stack((a,a.flip(-1)),1).requires_grad_(True)
    valid=torch.ones(3,2,dtype=torch.bool)
    content=torch.ones(3,4,4,dtype=torch.bool)
    assert float(mirror_total_variation(pair,valid,content).detach())==pytest.approx(0.,abs=1e-7)
    changed=pair.detach().clone()
    changed[:,1,...,0,0]+=.4
    changed.requires_grad_(True)
    loss=mirror_total_variation(changed,valid,content)
    grad=torch.autograd.grad(loss,changed)[0]
    assert grad[:,0].abs().sum()>0 and grad[:,1].abs().sum()>0
    separate=torch.stack([mirror_total_variation(changed[i:i+1],valid[i:i+1],content[i:i+1]) for i in range(3)]).mean()
    torch.testing.assert_close(loss,separate)
    torch.testing.assert_close(pair.flip(-1).flip(-1),pair)


def test_no_mirror_weight_and_original_temporal_regression():
    model,x,valid,content=case()
    out=model(x,valid,content)
    labels=torch.tensor([0,1])
    loss,detail=spatial_objective(out,labels,torch.arange(2),valid[::2],content[::2],epoch=20,mirror_weight=0.)
    first=torch.autograd.grad(loss,tuple(model.parameters()),retain_graph=True,allow_unused=True)
    second=torch.autograd.grad(detail['semantic_loss'],tuple(model.parameters()),allow_unused=True)
    for a,b in zip(first,second):
        if a is not None:torch.testing.assert_close(a,b)
    teacher=TrajectoryModel(16,16,8).eval().requires_grad_(False)
    state=copy.deepcopy(teacher.state_dict())
    features=torch.randn(2,2,2,16)
    mask=torch.ones(2,2,dtype=torch.bool)
    times=torch.tensor([[0.,.5]]*2)
    before=trajectory_loss(teacher,features,mask,times,labels,epoch=20)
    model(x,valid,content)['embedding'][:,0].sum().backward()
    after=trajectory_loss(teacher,features,mask,times,labels,epoch=20)
    torch.testing.assert_close(before[0],after[0],rtol=0,atol=0)
    for k in before[2]:torch.testing.assert_close(before[2][k],after[2][k],rtol=0,atol=0)
    for k in state:torch.testing.assert_close(state[k],teacher.state_dict()[k],rtol=0,atol=0)
    maps=model(x,valid,content)['attention'].reshape(2,2,2,2,4,4)
    stats=attention_diagnostics(maps,valid[::2],content[::2])
    assert all(torch.isfinite(v).all() for v in stats.values())


def _distributed_worker(rank,init_file,output):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from vision_model.training.spatial_semantic import SpatialPairedForward
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'file://{init_file}',rank=rank,world_size=2)
    torch.manual_seed(76)
    proto=torch.randn(2,8)
    branch=SpatialSemanticBranch(proto,width=16,hidden=16,embedding_dim=8)
    bank=torch.randn(5,2,3,2,2,16)
    valid=torch.ones(5,3,dtype=torch.bool);valid[-1,-1]=False
    content=torch.ones(5,2,2,dtype=torch.bool)
    labels=torch.tensor([0,0,0,1,1])
    indices=torch.arange(rank,5,2)
    wrapped=DistributedDataParallel(SpatialPairedForward(branch,2))
    out=wrapped(bank[indices],valid[indices],content[indices])
    loss,_=spatial_objective(out,labels[indices],indices,valid[indices],content[indices],epoch=20,mirror_weight=.3)
    loss.backward()
    distributed={k:v.grad.clone() for k,v in branch.named_parameters()}
    # An independent full-batch objective checks semantic-gather and auxiliary
    # weighting together with unequal, unpadded per-rank source counts.
    dist.destroy_process_group()
    reference=SpatialSemanticBranch(proto,width=16,hidden=16,embedding_dim=8)
    reference.load_state_dict(branch.state_dict(),strict=True)
    full=SpatialPairedForward(reference,2)(bank,valid,content)
    expected,_=spatial_objective(full,labels,torch.arange(5),valid,content,epoch=20,mirror_weight=.3)
    expected.backward()
    worst=0.
    for key,p in reference.named_parameters():
        torch.testing.assert_close(distributed[key],p.grad,atol=2e-5,rtol=2e-4)
        worst=max(worst,float((distributed[key]-p.grad).abs().max()))
    if rank==0:torch.save(dict(maximum_gradient_error=worst),output)


def test_distributed_global_gradient_matches_full_batch_with_unequal_shards(tmp_path):
    import torch.multiprocessing as mp
    result=tmp_path/'gradient.pt'
    mp.spawn(_distributed_worker,args=(str(tmp_path/'init'),str(result)),nprocs=2)
    assert torch.load(result,weights_only=True)['maximum_gradient_error']<2e-5
