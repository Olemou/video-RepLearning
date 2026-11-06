import torch
from vision_model.models.encoder import VJEPA2Encoder
from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.training.final_attention import FinalAttentionStudent, prefix_key_map


def tiny():
    return VJEPA2Encoder(VJEPA2Architecture(frames_per_clip=6, crop_size=16,
        patch_size=8, tubelet_size=2, hidden_size=96, num_hidden_layers=2,
        num_attention_heads=3, mlp_ratio=2.)).eval()


def test_batched_selected_query_replay_parameter_gradients_and_frozen_operations():
    torch.manual_seed(11)
    encoder = tiny()
    model = FinalAttentionStudent(encoder.blocks[-1], encoder.norm,
                                  train_attention=True, hidden=32, embedding_dim=16)
    x = torch.randn(3, 12, 96)
    q, a, tokens = model(x, frames=3, height=2, width=2, capture=True)
    expected = model.norm(model.blocks[-1](x, frames=3, height=2, width=2))[:, -4:]
    torch.testing.assert_close(tokens, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(tokens)
    params = tuple(model.blocks[-1].attn.parameters())
    replay_grad = torch.autograd.grad(tokens, params, upstream, retain_graph=True)
    full_grad = torch.autograd.grad(expected, params, upstream)
    for actual, wanted in zip(replay_grad, full_grad):
        torch.testing.assert_close(actual, wanted, rtol=1e-4, atol=1e-5)
    assert all('attn.' in n or n.startswith('projector.') for n in model.parameter_audit())
    frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = q.detach().clone()
    gradient = torch.autograd.grad(q[..., 0].sum(), a, retain_graph=True)[0]
    assert gradient.abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-6)
    q[..., 0].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    optimizer.step()
    assert (model(x, frames=3, height=2, width=2)-before).abs().max() > 1e-7
    for n, p in model.named_parameters():
        if n in frozen: torch.testing.assert_close(p, frozen[n], rtol=0, atol=0)


def test_key_maps_keep_past_time_and_actual_differentiable_attention():
    a = torch.zeros(2, 3, 4, 12)
    a[..., 1] = 1  # A key in the first tubelet, not the current/last tubelet.
    gradient = torch.ones_like(a)
    assert prefix_key_map(a).argmax(-1).tolist() == [1, 1]
    assert prefix_key_map(a, gradient, kind='signed_attention_gradient').argmax(-1).tolist() == [1, 1]
    assert prefix_key_map(a, -gradient, kind='positive_gradient').sum() == 0


def test_streamed_source_vjp_equals_joint_parameter_gradient():
    from vision_model.training.patch_vico import patch_vico
    torch.manual_seed(52)
    enc=tiny()
    model=FinalAttentionStudent(enc.blocks[-1],enc.norm,train_attention=True,hidden=16,embedding_dim=8)
    x=torch.randn(4,12,96)
    labels=torch.tensor([0,0,1,1]).repeat_interleave(4)
    ids=torch.arange(4).repeat_interleave(4)
    q=model(x,frames=3,height=2,width=2)
    loss,_=patch_vico(q.flatten(0,1),labels,ids,ids,distributed=False)
    params=[p for p in model.parameters() if p.requires_grad]
    expected=torch.autograd.grad(loss,params)
    with torch.no_grad():leaf=model(x,frames=3,height=2,width=2).requires_grad_(True)
    loss,_=patch_vico(leaf.flatten(0,1),labels,ids,ids,distributed=False)
    loss.backward()
    model.zero_grad()
    for i in range(4):
        (model(x[i:i+1],frames=3,height=2,width=2)*leaf.grad[i:i+1]).sum().backward()
    for p,g in zip(params,expected):torch.testing.assert_close(p.grad,g,rtol=2e-4,atol=1e-5)


def test_logit_gradient_equals_autograd_and_removes_probability_row_gauge():
    from vision_model.training.final_attention import attention_logit_gradient
    torch.manual_seed(41)
    logits=torch.randn(2,3,4,12,dtype=torch.double,requires_grad=True)
    a=logits.softmax(-1);gradient=torch.randn_like(a)
    expected=torch.autograd.grad((a*gradient).sum(),logits)[0]
    actual=attention_logit_gradient(a,gradient)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(actual.sum(-1),torch.zeros_like(actual[...,0]),atol=1e-15,rtol=0)
    torch.testing.assert_close(attention_logit_gradient(a,gradient+17),actual,atol=2e-15,rtol=1e-12)


def test_episode_margin_vjp_matches_joint_prefix_gradients_and_finite_difference():
    """The replay must differentiate the normalized episode decision, not a patch score."""
    import torch.nn.functional as F
    torch.manual_seed(71)
    enc=tiny()
    model=FinalAttentionStudent(enc.blocks[-1],enc.norm,hidden=16,embedding_dim=8).eval()
    inputs=[torch.randn(1,4*p,96,requires_grad=True) for p in (1,2,3)]
    outputs=[model(x,frames=p,height=2,width=2,capture=True) for p,x in enumerate(inputs,1)]
    q=torch.stack([entry[0][0] for entry in outputs])
    direction=F.normalize(torch.randn(2,8),dim=-1).diff(dim=0)[0]
    scalar=F.normalize(q.mean((0,1)),dim=-1)@direction
    expected=torch.autograd.grad(scalar,inputs,retain_graph=True)
    leaf=q.detach().requires_grad_(True)
    upstream=torch.autograd.grad(F.normalize(leaf.mean((0,1)),dim=-1)@direction,leaf)[0]
    for p,((patches,attention,_),x,wanted) in enumerate(zip(outputs,inputs,expected),1):
        actual=torch.autograd.grad(patches,x,upstream[p-1][None],retain_graph=True)[0]
        torch.testing.assert_close(actual,wanted,atol=1e-7,rtol=1e-5)
        signed=torch.autograd.grad(scalar,attention,retain_graph=True)[0]
        reverse=torch.autograd.grad(-scalar,attention,retain_graph=True)[0]
        torch.testing.assert_close(reverse,-signed,atol=0,rtol=0)
    # One input coordinate perturbs the complete prefix, including its keys.
    step=.002
    with torch.no_grad():
        values=[]
        for sign in (-1,1):
            changed=inputs[0].clone();changed[0,0,0]+=sign*step
            first=model(changed,frames=1,height=2,width=2)[0]
            mean=torch.stack((first,q[1],q[2])).mean((0,1))
            values.append(F.normalize(mean,dim=-1)@direction)
        finite=(values[1]-values[0])/(2*step)
    torch.testing.assert_close(finite,expected[0][0,0,0],atol=3e-5,rtol=.03)


def _distributed_vjp_worker(rank,init_file,destination):
    import copy
    import contextlib
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from vision_model.training.patch_vico import patch_vico
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'file://{init_file}',rank=rank,world_size=2)
    torch.manual_seed(34);enc=tiny()
    model=FinalAttentionStudent(enc.blocks[-1],enc.norm,train_attention=True,hidden=16,embedding_dim=8)
    initial=copy.deepcopy(model.state_dict());x=torch.randn(4,12,96)
    indices=torch.arange(rank,4,2);labels=torch.tensor([0,0,1,1])
    wrapped=DDP(model)
    with torch.no_grad():leaf=model(x[indices],frames=3,height=2,width=2).requires_grad_(True)
    ids=indices.repeat_interleave(4)
    loss,_=patch_vico(leaf.flatten(0,1),labels[indices].repeat_interleave(4),ids,ids)
    loss.backward()
    for j,index in enumerate(indices):
        sync=wrapped.no_sync() if j==0 else contextlib.nullcontext()
        with sync:
            (wrapped(x[index:index+1],frames=3,height=2,width=2)*leaf.grad[j:j+1]).sum().backward()
    gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    dist.destroy_process_group()
    reference=FinalAttentionStudent(enc.blocks[-1],enc.norm,train_attention=True,hidden=16,embedding_dim=8)
    reference.load_state_dict(initial)
    q=reference(x,frames=3,height=2,width=2);ids=torch.arange(4).repeat_interleave(4)
    loss,_=patch_vico(q.flatten(0,1),labels.repeat_interleave(4),ids,ids,distributed=False)
    loss.backward();maximum=0.
    for name,p in reference.named_parameters():
        if not p.requires_grad:continue
        torch.testing.assert_close(gradients[name],p.grad,rtol=3e-4,atol=2e-5)
        maximum=max(maximum,float((gradients[name]-p.grad).abs().max()))
    if rank==0:torch.save(maximum,destination)


def test_ddp_streamed_attention_and_projector_gradients_equal_global_reference(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_vjp_worker,args=(str(tmp_path/'init'),str(tmp_path/'error.pt')),nprocs=2)
    assert torch.load(tmp_path/'error.pt',weights_only=True)<2e-5
