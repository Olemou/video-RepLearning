import copy
import contextlib
import torch
import pytest
from vision_model.models.encoder import VJEPA2Encoder
from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.training.trajectory import TrajectoryModel,trajectory_loss
from vision_model.training.spill_localization import (SpillPatchHead,SpillPrefixStudent,
    foreground_background_loss,localized_trajectory_loss)


def tiny_student():
    encoder=VJEPA2Encoder(VJEPA2Architecture(frames_per_clip=6,crop_size=16,patch_size=8,
        tubelet_size=2,hidden_size=96,num_hidden_layers=2,num_attention_heads=3,mlp_ratio=2.)).eval()
    return SpillPrefixStudent(encoder.blocks[-1],encoder.norm,train_attention=True,
                             head=SpillPatchHead(96,16,8))


def batch(model,count=4):
    features=[];logits=[]
    inputs=[torch.randn(count*2,p*4,96) for p in (1,2,3)]
    for p,x in enumerate(inputs,1):
        result=model(x,frames=p,height=2,width=2)
        features.append(result['student_features'].reshape(count,2,96))
        logits.append(result['local_logits'].reshape(count,2,4))
    return torch.stack(features,2),torch.stack(logits,2),inputs


def test_local_loss_ignores_unknown_nan_cells_and_balances_classes():
    logits=torch.tensor([0.,2.,-1.,float('nan')],requires_grad=True)
    labels=torch.tensor([1,0,0,-1])
    loss,_=foreground_background_loss(logits,labels,distributed=False)
    expected=.5*(torch.nn.functional.softplus(-logits[0])+
                 torch.nn.functional.softplus(logits[1:3]).mean())
    torch.testing.assert_close(loss,expected.double())
    loss.backward();assert logits.grad[-1]==0 and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError,match='foreground AND background'):
        foreground_background_loss(torch.zeros(2),torch.tensor([0,-1]),distributed=False)


def test_zero_bridge_is_exact_and_teacher_targets_stay_detached():
    torch.manual_seed(12);model=TrajectoryModel(16,16,8)
    teacher=torch.randn(4,2,3,16,requires_grad=True);valid=torch.ones(4,3,dtype=torch.bool)
    times=torch.arange(3).float().expand(4,-1);labels=torch.tensor([0,0,1,1])
    baseline,_,_=trajectory_loss(model,teacher,valid,times,labels,epoch=20)
    student=teacher.detach().clone().requires_grad_(True)
    bridged,_,_=trajectory_loss(model,teacher,valid,times,labels,epoch=20,student_features=student)
    torch.testing.assert_close(bridged,baseline,atol=0,rtol=0)
    gt,gs=torch.autograd.grad(bridged,(teacher,student),allow_unused=True)
    assert gt is None and gs.norm()>0


def test_both_losses_reach_actual_attention_and_frozen_trajectory_is_preserved():
    torch.manual_seed(14);student=tiny_student()
    trajectory=TrajectoryModel(96,16,8).eval().requires_grad_(False)
    teacher=torch.randn(4,2,3,96,requires_grad=True)
    means,logits,_=batch(student)
    points=torch.full_like(logits,-1,dtype=torch.long);points[0,0,0,0]=0;points[2,0,0,1]=1
    valid=torch.ones(4,3,dtype=torch.bool);times=torch.arange(3).float().expand(4,-1)
    labels=torch.tensor([0,0,1,1]);frozen={n:p.detach().clone() for n,p in trajectory.named_parameters()}
    loss,details,_=localized_trajectory_loss(trajectory,teacher,means,logits,points,valid,times,labels,
        fg_weight=.2,epoch=20,distributed=False)
    attention=tuple(student.blocks[0].attn.parameters())
    for component in ('trajectory_loss','foreground_background_loss'):
        grads=torch.autograd.grad(details[component],attention,retain_graph=True)
        assert sum(float(g.norm()) for g in grads)>0
    before={n:p.detach().clone() for n,p in student.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=1e-4)
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert teacher.grad is None and all(p.grad is None for p in trajectory.parameters())
    for n,p in trajectory.named_parameters():torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
    for n,p in student.named_parameters():
        if n in before:torch.testing.assert_close(p,before[n],atol=0,rtol=0)


def test_attribution_targets_trajectory_not_the_auxiliary_local_head():
    from vision_model.interpretability.spill_localization import explain_trajectory_spatial
    torch.manual_seed(18);student=tiny_student().eval().requires_grad_(False)
    trajectory=TrajectoryModel(96,16,8).eval().requires_grad_(False)
    prefixes=[torch.randn(1,p*4,96) for p in (1,2,3)]
    times=torch.arange(3).float();prototypes=torch.nn.functional.normalize(torch.randn(2,8),dim=-1)
    result=explain_trajectory_spatial(student,trajectory,prefixes,times,prototypes,height=2,width=2)
    with torch.no_grad():
        features=torch.stack([student(x,frames=t+1,height=2,width=2)['student_features'] for t,x in enumerate(prefixes)],1)
        expected=trajectory(features,torch.ones(1,3,dtype=torch.bool),times[None])['embedding'][0]
    torch.testing.assert_close(result['embedding'],expected)
    assert abs(result['margin']-float(expected@(prototypes[1]-prototypes[0])))<1e-6
    # Changing only an auxiliary head must not change the claimed decision map.
    before=result['local_probabilities'].clone()
    with torch.no_grad():student.local_head.prototypes.neg_()
    changed=explain_trajectory_spatial(student,trajectory,prefixes,times,prototypes,height=2,width=2)
    torch.testing.assert_close(changed['maps']['positive_gradient'],result['maps']['positive_gradient'],atol=0,rtol=0)
    assert (changed['local_probabilities']-before).abs().max()>0


def _worker(rank,init_file,destination):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    torch.set_num_threads(1);torch.manual_seed(28)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=2)
    model=tiny_student();initial=copy.deepcopy(model.state_dict())
    trajectory=TrajectoryModel(96,16,8).eval().requires_grad_(False)
    teacher=torch.randn(4,2,3,96);labels=torch.tensor([0,0,1,1]);valid=torch.ones(4,3,dtype=torch.bool)
    times=torch.arange(3).float().expand(4,-1)
    points=torch.full((4,2,3,4),-1,dtype=torch.long)
    points[0,0,0,0]=0;points[1,0,1,:2]=0;points[2,0,0,0]=1
    means,logits,inputs=batch(model)
    # One rank has no foreground, the other no background; counts must be global.
    indices=torch.tensor([0,1] if rank==0 else [2,3])
    leaf_mean=means[indices].detach().requires_grad_(True);leaf_logits=logits[indices].detach().requires_grad_(True)
    total,_,_=localized_trajectory_loss(trajectory,teacher[indices],leaf_mean,leaf_logits,points[indices],
        valid[indices],times[indices],labels[indices],fg_weight=.3,epoch=20)
    total.backward();wrapped=DDP(model,broadcast_buffers=False)
    for t,x in enumerate(inputs):
        sync=wrapped.no_sync() if t<2 else contextlib.nullcontext()
        with sync:
            out=wrapped(x.reshape(4,2,*x.shape[1:])[indices].flatten(0,1),frames=t+1,height=2,width=2)
            ((out['student_features'].reshape(2,2,96)*leaf_mean.grad[:,:,t]).sum()+
             (out['local_logits'].reshape(2,2,4)*leaf_logits.grad[:,:,t]).sum()).backward()
    actual={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    dist.destroy_process_group()
    reference=tiny_student();reference.load_state_dict(initial,strict=True)
    means=[];logits=[]
    for t,x in enumerate(inputs):
        out=reference(x,frames=t+1,height=2,width=2)
        means.append(out['student_features'].reshape(4,2,96));logits.append(out['local_logits'].reshape(4,2,4))
    total,_,_=localized_trajectory_loss(trajectory,teacher,torch.stack(means,2),torch.stack(logits,2),points,
        valid,times,labels,fg_weight=.3,epoch=20,distributed=False)
    total.backward();error=0.
    for n,p in reference.named_parameters():
        if p.requires_grad:
            torch.testing.assert_close(actual[n],p.grad,rtol=4e-4,atol=2e-5)
            error=max(error,float((actual[n]-p.grad).abs().max()))
    if rank==0:torch.save(error,destination)


def test_two_rank_joint_streaming_gradients_match_global_reference_with_sparse_class_imbalance(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_worker,args=(str(tmp_path/'init'),str(tmp_path/'error.pt')),nprocs=2)
    assert torch.load(tmp_path/'error.pt',weights_only=True)<2e-5
