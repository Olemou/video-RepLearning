import pytest
import torch
from vision_model.training.dense_spill import (DenseSpillReadout,candidate_coverage,
    require_candidate_coverage,source_balanced_partial_loss,dense_consensus_candidates)
from vision_model.models.dense_encoder import DenseArchitecture,DenseVideoEncoder,convert_dense_state,rotate_dense


def test_dense_encoder_preserves_video_and_image_grid_and_loads_every_key():
    cfg=DenseArchitecture(width=48,depth=2,heads=2,patch=8,norm_count=4)
    m=DenseVideoEncoder(cfg)
    assert m(torch.randn(1,3,4,24,32)).shape==(1,24,48)
    assert m(torch.randn(1,3,1,24,32)).shape==(1,12,48)
    state={'module.backbone.'+k:v for k,v in m.state_dict().items()}
    twin=DenseVideoEncoder(cfg);twin.load_state_dict(convert_dense_state(state),strict=True)
    del state['module.backbone.video_mod_embed']
    with pytest.raises(RuntimeError,match='video_mod_embed'):twin.load_state_dict(convert_dense_state(state),strict=True)
    with pytest.raises(RuntimeError,match='Unexpected'):convert_dense_state({'wrong':torch.zeros(1)})


def test_dense_rotation_uses_adjacent_frequency_pairs_and_preserves_norm():
    x=torch.randn(2,3,7,8,dtype=torch.float64)
    p=torch.arange(7,dtype=torch.float64)*.5
    actual=rotate_dense(x,p)
    angle=p[:,None]/10000**(torch.arange(4,dtype=torch.float64)/4)
    pairs=x.unflatten(-1,(4,2))
    expected=torch.stack((pairs[...,0]*angle.cos()-pairs[...,1]*angle.sin(),
                          pairs[...,1]*angle.cos()+pairs[...,0]*angle.sin()),-1).flatten(-2)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual.norm(dim=-1),x.norm(dim=-1))


def test_dense_candidates_require_agreement_and_ignore_invalid_times():
    raw=torch.full((2,2,2,16),.1);residual=raw.clone()
    raw[1,0,0,0]=residual[1,0,0,0]=.9
    for x in (raw,residual):x[:,1]=x[:,0].reshape(2,2,4,4).flip(-1).flatten(2)
    valid=torch.tensor([[True,False],[True,False]])
    support=torch.ones(2,16,dtype=torch.bool)
    calibration={m:{'0.95':.2,'0.99':.5} for m in ('raw','residual')}
    t=dense_consensus_candidates(raw,residual,valid,support,torch.tensor([0,1]),calibration)
    assert t[1,0,0,0]==1 and (t[1,0,0,1:]==0).all()
    assert (t[:,:,1]==-1).all() and (t[0,:,0]==0).all()
    assert torch.equal(t[:,1],t[:,0].reshape(2,2,4,4).flip(-1).flatten(2))
    residual[1,1,0,3]=.1 # Independent mirror does not confirm foreground.
    t=dense_consensus_candidates(raw,residual,valid,support,torch.tensor([0,1]),calibration)
    assert t[1,0,0,0]!=1


@pytest.mark.parametrize('normalize_patches',[True,False])
def test_local_contributions_reconstruct_actual_decision_and_ignore_nan_padding(normalize_patches):
    torch.manual_seed(239);head=DenseSpillReadout(8,8,4,normalize_patches=normalize_patches)
    x=torch.randn(2,3,6,8);valid=torch.tensor([[True,True,False],[True,True,True]])
    support=torch.tensor([[True,True,True,True,False,False]]*2)
    mask=valid[...,None]&support[:,None]
    x[~mask]=float('nan')
    result=head(x,valid,support)
    reconstructed=result['contributions'].masked_fill(~mask,0).sum((1,2))/mask.sum((1,2))
    torch.testing.assert_close(reconstructed,result['margin'],atol=2e-7,rtol=1e-5)
    assert torch.isfinite(result['embedding']).all()
    # The class prototypes are part of the real decision, not a detached display head.
    gradient=torch.autograd.grad(result['margin'].sum(),head.projector[-1].weight)[0]
    assert gradient.norm()>0
    head.set_prototypes(head.prototypes.flip(0).clone())
    torch.testing.assert_close(head(x,valid,support)['margin'],-result['margin'])


def test_gate_rejects_cross_scene_and_cross_time_label_shortcuts():
    y=torch.tensor([0,1,1]);t=torch.full((3,2,2,4),-1,dtype=torch.long)
    t[0]=0;t[1:,:,:,0]=1
    assert not candidate_coverage(t,y,torch.arange(3))['ready_for_training']
    t[1,:,0,1]=0;t[1,:,0,0]=-1 # BG at t0, FG at t1 is insufficient.
    assert not candidate_coverage(t,y,torch.arange(3))['ready_for_training']
    with pytest.raises(ValueError,match='Within-spill'):require_candidate_coverage(t,y,torch.arange(3))
    t[1,:,0,0]=1
    assert require_candidate_coverage(t,y,torch.arange(3))['sources_with_both']==1
    # Validation candidates cannot affect the training gate.
    assert not candidate_coverage(t,y,torch.tensor([0,2]))['ready_for_training']


def test_source_balanced_loss_does_not_dilute_cold_background_with_clean_area():
    y=torch.tensor([0,1]);t=torch.full((2,2,1,4),-1,dtype=torch.long)
    t[0]=0;t[1,:,:,0]=0;t[1,:,:,1]=1
    x=torch.zeros_like(t,dtype=torch.float32,requires_grad=True)
    loss,counts=source_balanced_partial_loss(x,t,y)
    torch.testing.assert_close(loss,torch.tensor(2.).log().double())
    g=torch.autograd.grad(loss,x)[0]
    assert g[1,:,:,0].sum()>0 and g[1,:,:,1].sum()<0
    assert counts['cold_background_sources']==1
    # Replicating clean patch area changes no stratum's weight.
    tiled=t.repeat_interleave(3,-1);xx=x.detach().repeat_interleave(3,-1)
    same,_=source_balanced_partial_loss(xx,tiled,y)
    torch.testing.assert_close(same,loss)


def test_constant_cold_map_has_positive_loss_floor_despite_clean_negatives():
    y=torch.tensor([0,1]);t=torch.tensor([[[[0,0]]]*2,[[[0,1]]]*2])
    x=torch.tensor([[[[-40.,-40.]]]*2,[[[0.,0.]]]*2])
    loss,_=source_balanced_partial_loss(x,t,y)
    torch.testing.assert_close(loss,torch.tensor(2/3).double()*torch.tensor(2.).double().log())


def _distributed_loss_worker(rank,port,path):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    module=torch.nn.Linear(1,1,bias=False);module.weight.data.fill_(.37)
    ddp=DistributedDataParallel(module)
    # Rank 0 has only clean sources, rank 1 only cold sources.
    features=torch.arange(1,17,dtype=torch.float32).reshape(4,2,1,2,1)/10
    labels=torch.tensor([0,0,1,1])
    targets=torch.tensor([[[[0,0]]]*2]*2+[[[[0,1]]]*2]*2)
    local=slice(rank*2,(rank+1)*2)
    logits=ddp(features[local]).squeeze(-1)
    loss,_=source_balanced_partial_loss(logits,targets[local],labels[local]);loss.backward()
    if rank==0:torch.save(module.weight.grad, path)
    dist.destroy_process_group()


def test_distributed_stratum_gradient_matches_global_reference(tmp_path):
    import socket
    import torch.multiprocessing as mp
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    output=tmp_path/'gradient.pt'
    mp.spawn(_distributed_loss_worker,args=(port,str(output)),nprocs=2)
    features=torch.arange(1,17,dtype=torch.float32).reshape(4,2,1,2)/10
    target=torch.tensor([[[[0,0]]]*2]*2+[[[[0,1]]]*2]*2)
    weight=torch.tensor(.37,requires_grad=True)
    loss,_=source_balanced_partial_loss(features*weight,target,torch.tensor([0,0,1,1]))
    loss.backward()
    torch.testing.assert_close(torch.load(output,weights_only=True).squeeze(),weight.grad,atol=1e-7,rtol=1e-6)
