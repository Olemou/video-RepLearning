"""Actual joint-trajectory attribution, separate from the supervised local map."""
import torch
from vision_model.training.final_attention import prefix_key_map


def explain_trajectory_spatial(student, trajectory, prefixes, timestamps, prototypes, *, height=16, width=16):
    """Differentiate one spill-minus-clean trajectory prototype margin.

    Complete independently encoded prefixes are required, identity view only.
    Cached pre23 inputs are a conditional boundary, not an image-gradient graph.
    All heads/queries and original key times are retained, without normalization.
    """
    if not prefixes or timestamps.shape != (len(prefixes),) or prototypes.ndim != 2 or prototypes.shape[0] != 2:
        raise ValueError('Need prefix list, aligned endpoint times and two training prototypes')
    with torch.enable_grad():
        means=[];local=[];attention=[]
        for end,prefix in enumerate(prefixes,1):
            if prefix.ndim!=3 or prefix.shape[0]!=1:
                raise ValueError('Explain one identity-view source at a time')
            out=student(prefix.detach().requires_grad_(True),frames=end,height=height,width=width,capture=True)
            means.append(out['student_features']);local.append(out['local_logits']);attention.append(out['attention'])
        features=torch.stack(means,1)
        valid=torch.ones(features.shape[:2],dtype=torch.bool,device=features.device)
        output=trajectory(features,valid,timestamps[None].to(features.device),detach_teacher=False)
        embedding=output['embedding'][0]
        if prototypes.shape[1]!=embedding.shape[0] or not torch.isfinite(prototypes).all():
            raise ValueError('Prototypes must match the actual trajectory embedding')
        margin=embedding@(prototypes[1]-prototypes[0]).to(embedding)
        if not torch.isfinite(margin):raise ValueError('Nonfinite decision target')
        gradients=torch.autograd.grad(margin,attention)
    maps={kind:torch.zeros(len(prefixes),height*width,device=features.device)
          for kind in ('raw','positive_gradient','signed_attention_gradient')}
    for end,(a,g) in enumerate(zip(attention,gradients),1):
        for kind in maps:
            maps[kind][:end]+=prefix_key_map(a,g,kind=kind).reshape(end,height*width).detach()
    return dict(margin=float(margin.detach()),embedding=embedding.detach(),
        local_probabilities=torch.stack(local,1)[0].detach().sigmoid(),maps=maps,
        target='Actual joint TrajectoryModel episode spill-minus-clean training-prototype margin',
        attribution_scope='Conditional final-block key sensitivity; not pixel attribution or a spill mask')
