import copy
import pytest
import torch
import torch.nn.functional as F

from vision_model.training.patch_vico import (patch_vico,uncertainty_weights,
    source_balanced_weights,PatchProjector,local_feature_retention)
from vision_model.training.vico import vico_pairwise_contrastive,vico_pair_weights


@pytest.mark.parametrize('epoch',[0,7,20])
@pytest.mark.parametrize('chunk',[1,5,40])
def test_streamed_matches_dense_loss_and_stop_mining_gradient(epoch,chunk):
    torch.manual_seed(10)
    x=torch.randn(12,8,dtype=torch.double,requires_grad=True)
    labels=torch.tensor([0]*6+[1]*6)
    images=torch.tensor([0,0,1,1,2,2,3,3,4,4,5,5])
    dense,_=vico_pairwise_contrastive(x,labels,source_ids=images,epoch=epoch,curriculum_epochs=20)
    expected=torch.autograd.grad(dense,x)[0]
    streamed,detail=patch_vico(x,labels,images,images,epoch=epoch,chunk_size=chunk)
    actual=torch.autograd.grad(streamed,x)[0]
    torch.testing.assert_close(streamed,dense,atol=2e-14,rtol=1e-12)
    torch.testing.assert_close(actual,expected,atol=2e-13,rtol=1e-11)
    assert float(detail['same_image_weight_max_error'])==0


def test_tied_ranks_and_image_provenance_match_dense():
    x=F.normalize(torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.],[1.,1.],[1.,1.]]).double(),dim=-1)
    cosine=x@x.T
    labels=torch.tensor([0,0,0,0,1,1])
    images=torch.tensor([0,0,1,1,2,2])
    reference,_,_=vico_pair_weights(cosine,labels,images,epoch=20,curriculum_epochs=20,temperature=.1,epsilon=1e-6)
    actual,_=uncertainty_weights(cosine,images[:,None]==images[None],epoch=20,curriculum_epochs=20)
    torch.testing.assert_close(actual,reference,atol=0,rtol=0)
    assert torch.equal(actual[images[:,None]==images[None]],torch.ones_like(actual[images[:,None]==images[None]]))
    assert (actual[images[:,None]!=images[None]]>1).any()


def test_source_balance_is_not_flat_patch_count_and_rejects_bad_metadata():
    labels=torch.tensor([0,0,0,0,1,1,1])
    images=torch.tensor([0,0,0,1,2,2,2])
    sources=torch.tensor([0,0,0,0,1,1,1])
    weights=source_balanced_weights(labels,images,sources)
    torch.testing.assert_close(weights[:3].sum(),torch.tensor(.25,dtype=torch.double))
    torch.testing.assert_close(weights[3],torch.tensor(.25,dtype=torch.double))
    torch.testing.assert_close(weights[4:].sum(),torch.tensor(.5,dtype=torch.double))
    with pytest.raises(ValueError):source_balanced_weights(labels,torch.zeros_like(images),sources)
    with pytest.raises(ValueError):patch_vico(torch.zeros(7,4),labels,images,sources)
    with pytest.raises(ValueError):patch_vico(torch.randn(7,4),labels,images,sources,temperature=0)


def test_local_retention_is_source_balanced_masks_padding_and_detaches_teacher():
    torch.manual_seed(19)
    q=torch.randn(2,2,4,6,requires_grad=True)
    teacher=torch.randn_like(q,requires_grad=True)
    valid=torch.ones(2,2,4,dtype=torch.bool);valid[1,:,1:]=False
    loss=local_feature_retention(q,teacher,valid)
    expected=torch.stack([.5*(q[i][valid[i]]-teacher[i][valid[i]].detach()).square().sum(-1).mean() for i in range(2)]).mean()
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert teacher.grad is None
    assert q.grad[~valid].count_nonzero()==0
    assert float(local_feature_retention(q.detach(),q.detach(),valid))==0
    poisoned=q.detach().clone();poisoned[~valid]=float('nan')
    torch.testing.assert_close(local_feature_retention(poisoned,teacher,valid),loss.detach())


def test_signed_maps_reconstruct_the_actual_classifier_with_padding(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_patch_vico import patch_diagnostics
    torch.manual_seed(45)
    q=F.normalize(torch.randn(3,2,3,16,8),dim=-1)
    valid=torch.tensor([[1,1,1],[1,1,0],[1,0,0]],dtype=torch.bool)
    content=torch.ones(3,16,dtype=torch.bool);content[:,0]=False;content[:,3]=False
    # Spatial support is mirror symmetric; missing frames and cells may hold NaNs.
    content[:,-1]=False;content[:,-4]=False
    support=valid[:,None,:,None] & content[:,None,None]
    q=q.masked_fill(~support[...,None],float('nan'))
    prototypes=F.normalize(torch.randn(2,8),dim=-1)
    z,maps,diag,_,error=patch_diagnostics(q,valid,content,prototypes)
    contribution=maps.sum((-1,-2))/(valid.sum(1)*content.sum(1))[:,None]
    torch.testing.assert_close(contribution,z@(prototypes[1]-prototypes[0]),atol=2e-6,rtol=2e-6)
    assert error<2e-6
    assert all(torch.isfinite(value).all() for value in diag.values())


def _worker(rank,init_file,destination):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'file://{init_file}',rank=rank,world_size=2)
    torch.manual_seed(24)
    model=PatchProjector(8,12,6)
    state=copy.deepcopy(model.state_dict())
    inputs=torch.randn(14,8)
    labels=torch.tensor([0]*8+[1]*6)
    images=torch.tensor([0]*3+[1]*3+[2]*2+[3]*3+[4]*3)
    sources=torch.tensor([0]*6+[1]*2+[2]*6)
    # Source ownership gives 8/6 patches, variable image counts and no repeats.
    chosen=torch.arange(8) if rank==0 else torch.arange(8,14)
    wrapped=DDP(model)
    loss,_=patch_vico(wrapped(inputs[chosen]),labels[chosen],images[chosen],sources[chosen],epoch=13,chunk_size=3)
    loss.backward()
    gradients={k:p.grad.clone() for k,p in model.named_parameters()}
    dist.destroy_process_group()
    reference=PatchProjector(8,12,6);reference.load_state_dict(state)
    expected,_=patch_vico(reference(inputs),labels,images,sources,epoch=13,chunk_size=20,distributed=False)
    expected.backward()
    worst=0.
    for name,p in reference.named_parameters():
        torch.testing.assert_close(gradients[name],p.grad,atol=2e-6,rtol=3e-5)
        worst=max(worst,float((gradients[name]-p.grad).abs().max()))
    if rank==0:torch.save(worst,destination)


def test_distributed_anchor_and_candidate_gradients_match_unequal_global_reference(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_worker,args=(str(tmp_path/'init'),str(tmp_path/'error.pt')),nprocs=2)
    assert torch.load(tmp_path/'error.pt',weights_only=True)<2e-6
