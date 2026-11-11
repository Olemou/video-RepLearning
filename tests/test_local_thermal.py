import torch
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from torch.nn.parallel import DistributedDataParallel
from vision_model.training.local_thermal import LocalThermalSettings,local_relative_features,LocalThermalReadout,region_pairs,local_pair_loss,compact_pair,packed_local_pairs,packed_pair_terms


@pytest.mark.parametrize('kind',['pointwise','local_conv'])
def test_relative_features_locality_affine_flip_and_no_future(kind):
    torch.manual_seed(23);x=.2+.4*torch.rand(2,1,144,160)
    a=local_relative_features(x)
    b=local_relative_features(x*.7+.1)
    assert torch.allclose(a,b,atol=3e-4,rtol=3e-4)
    assert torch.allclose(a.flip(-1),local_relative_features(x.flip(-1)),atol=1e-5)
    changed=x.clone();changed[1]=.9;changed[0,:,:40,:40]=.9
    result=local_relative_features(changed)
    assert torch.equal(a[0,:,12,14],result[0,:,12,14])
    # Derivative to pixels outside the declared finite field is exactly zero.
    x.requires_grad_(True);m=LocalThermalReadout(kind=kind);s=m(local_relative_features(x))[0,12,14]
    g=torch.autograd.grad(s,x)[0]
    assert g[1].abs().max()==0 and g[0,:,:40,:40].abs().max()==0


def test_two_dark_regions_and_flat_field_abstention():
    # Analytic machinery control, not a real-liquid annotation.
    image=np.full((144,160),.55,np.float32)
    image[20:44,25:41]=.12;image[80:112,105:129]=.18
    pairs=region_pairs(image)
    assert len(pairs)>=2
    assert any(p['region'][3:5,3:5].sum()>0 for p in pairs)
    assert any(p['region'][11:13,14:16].sum()>0 for p in pairs)
    assert not region_pairs(np.full((144,160),.5,np.float32))


def test_pair_ranking_has_no_constant_scene_offset_shortcut_or_unknown_gradient():
    a=torch.zeros(3,2,2,4,4,requires_grad=True)
    r=torch.zeros(4,4);r[1,1]=1;b=torch.zeros(4,4);b[2,2]=1
    pairs=[[[dict(region=r,ring=b,confidence=1.)],[]],[[],[]],[[],[]]]
    labels=torch.tensor([1,0,1]);valid=torch.ones(3,2,dtype=torch.bool)
    clean,rank,counts=local_pair_loss(a,labels,torch.arange(3),valid,pairs)
    assert torch.allclose(rank,F.softplus(torch.tensor(1.)).double())
    offset=a.detach().clone();offset[0]+=11
    _,other,_=local_pair_loss(offset,labels,torch.arange(3),valid,pairs)
    assert torch.equal(rank,other)
    (clean+rank).backward()
    assert a.grad[0,:,0,1,1].sum()<0 and a.grad[0,:,0,2,2].sum()>0
    assert a.grad[2].abs().max()==0 and counts['paired_positive_sources']==1


def test_packed_and_reference_pair_losses_and_gradients_match():
    torch.manual_seed(33)
    a=torch.randn(4,2,3,4,4,requires_grad=True)
    r=torch.zeros(4,4);r[1:3,1:3]=1;b=1-r
    pair=dict(region=r,ring=b,confidence=.7)
    pairs=[[[pair],[],[pair]],[[pair],[],[]],[[],[],[]],[[],[],[]]]
    labels=torch.tensor([1,0,1,0]);valid=torch.tensor([[1,1,0],[1,1,1],[1,0,0],[1,1,1]],dtype=torch.bool)
    first,second,_=local_pair_loss(a,labels,torch.arange(4),valid,pairs)
    compact=[[[compact_pair(p) for p in time] for time in source] for source in pairs]
    packed=packed_local_pairs(compact,torch.arange(4),valid,(4,4),'cpu')
    x,y=packed_pair_terms(a,labels,valid,packed)
    assert torch.allclose(first,x,atol=1e-7) and torch.allclose(second,y,atol=1e-7)
    g1=torch.autograd.grad(first+second,a,retain_graph=True)[0]
    g2=torch.autograd.grad(x+y,a)[0]
    assert torch.allclose(g1,g2,atol=1e-7)


def _ddp_local_worker(rank,path,result):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    torch.manual_seed(11);features=torch.randn(2,2,1,8,4,4)
    torch.manual_seed(901);model=LocalThermalReadout();wrapped=DistributedDataParallel(model)
    r=torch.zeros(4,4);r[1,1]=1;a=torch.zeros(4,4);a[2,2]=1
    pairs=[[[]],[[dict(region=r,ring=a,confidence=1.)]]]
    valid=torch.ones(1,1,dtype=torch.bool)
    packed=packed_local_pairs(pairs,torch.tensor([rank]),valid,(4,4),'cpu')
    score=wrapped(features[rank].flatten(0,1)).reshape(1,2,1,4,4)
    loss1,loss2=packed_pair_terms(score,torch.tensor([rank]),valid,packed)
    (loss1+loss2).backward()
    if rank==0:torch.save(torch.cat([p.grad.flatten() for p in model.parameters()]),result)
    dist.destroy_process_group()


def test_distributed_local_gradient_with_empty_rank_strata_matches_global(tmp_path):
    mp.spawn(_ddp_local_worker,args=(str(tmp_path/'init'),str(tmp_path/'gradient.pt')),nprocs=2,join=True)
    torch.manual_seed(11);features=torch.randn(2,2,1,8,4,4)
    torch.manual_seed(901);model=LocalThermalReadout()
    r=torch.zeros(4,4);r[1,1]=1;a=torch.zeros(4,4);a[2,2]=1
    pairs=[[[]],[[dict(region=r,ring=a,confidence=1.)]]];valid=torch.ones(2,1,dtype=torch.bool)
    packed=packed_local_pairs(pairs,torch.arange(2),valid,(4,4),'cpu')
    score=model(features.flatten(0,2)).reshape(2,2,1,4,4)
    x,y=packed_pair_terms(score,torch.tensor([0,1]),valid,packed);(x+y).backward()
    expected=torch.cat([p.grad.flatten() for p in model.parameters()])
    torch.testing.assert_close(torch.load(tmp_path/'gradient.pt',weights_only=True),expected,atol=2e-7,rtol=1e-5)
