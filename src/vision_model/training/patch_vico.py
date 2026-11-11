"""Direct all-patch ViCo, with explicit image provenance and bounded memory.

Image labels are weak patch associations, not mask annotations. No MIL or
learned spatial pooling is used. The existing episode objective is untouched.
Mining statistics have the same stop-gradient contract as training.vico.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.function import once_differentiable


def uncertainty_weights(cosine, image_equal, *, epoch, curriculum_epochs,
                        positive_weighting=True):
    """Exact average descending ranks using O(rows * candidates) memory.

    Self entries participate in the evidence/ranks, matching the episode
    reference. They are excluded later from the actual contrastive pairs.
    """
    with torch.no_grad():
        cohesion=cosine.softmax(-1).exp()
        noncohesion=(-cosine).softmax(-1).exp()
        uncertainty=2/(cohesion+noncohesion+2)
        ordered=uncertainty.sort(-1).values.contiguous()
        values=uncertainty.contiguous()
        left=torch.searchsorted(ordered,values,right=False)
        right=torch.searchsorted(ordered,values,right=True)
        n=cosine.shape[-1]
        fraction=((n-right).to(cosine.dtype)+(right-left-1).to(cosine.dtype)/2)/(n-1)
        progress=min(max(epoch/curriculum_epochs,0.),1.)
        weights=1+torch.tanh(progress*fraction)
        weights=torch.where(image_equal,1.,weights)
        if not positive_weighting:weights=torch.ones_like(weights)
    return weights,uncertainty


def source_balanced_weights(labels,image_ids,source_ids):
    """Equal sources, equal images within source, equal patches within image."""
    images,inverse,counts=torch.unique(image_ids,sorted=True,return_inverse=True,return_counts=True)
    order=image_ids.argsort(stable=True)
    starts=torch.cat((counts.new_zeros(1),counts.cumsum(0)[:-1]))
    first=order[starts]
    if not torch.equal(labels,labels[first][inverse]) or not torch.equal(source_ids,source_ids[first][inverse]):
        raise ValueError('An original image cannot span sources or conflicting labels')
    sources,source_inverse,image_counts=torch.unique(source_ids[first],return_inverse=True,return_counts=True)
    source_labels=labels[first]
    for source in sources:
        if source_labels[source_ids[first]==source].unique().numel()!=1:
            raise ValueError('A source cannot have conflicting class labels')
    return 1/(len(sources)*image_counts[source_inverse][inverse].double()*counts[inverse])


class _StreamedCompetition(torch.autograd.Function):
    """First-order exact stop-mining gradient, accumulated a row chunk at a time.

    Saves only dL/d(normalized embeddings), not an all-pairs autograd graph.
    The analytic derivative is checked against dense torch autograd in tests.
    Higher-order derivatives through this function are intentionally unsupported.
    """
    @staticmethod
    def forward(ctx,z,labels,image_ids,anchors,anchor_weights,temperature,epsilon,
                epoch,curriculum_epochs,chunk_size,positive_weighting):
        total=z.new_zeros(())
        need_gradient=ctx.needs_input_grad[0]
        gradient=torch.zeros_like(z) if need_gradient else None
        # min positive, max negative, min anchor separation, worst positive
        # pair, min/max positive weight, same-image weight error, anchors.
        stats=z.new_tensor([torch.inf,-torch.inf,torch.inf,0.,torch.inf,-torch.inf,0.,len(anchors)])
        n=len(z)
        candidates=torch.arange(n,device=z.device)
        for start in range(0,len(anchors),chunk_size):
            rows=anchors[start:start+chunk_size]
            cosine=z[rows]@z.T
            same=labels[rows,None].eq(labels[None])
            positive=same & rows[:,None].ne(candidates[None])
            negative=~same
            npositive=positive.sum(-1)
            nnegative=negative.sum(-1)
            if (npositive==0).any() or (nnegative==0).any():
                raise ValueError('Every anchor needs a positive and a negative')
            image_equal=image_ids[rows,None].eq(image_ids[None])
            weights,_=uncertainty_weights(cosine,image_equal,epoch=epoch,
                curriculum_epochs=curriculum_epochs,positive_weighting=positive_weighting)
            logits=cosine/temperature
            negscores=logits.masked_fill(~negative,-torch.inf)
            denominator=torch.logaddexp(torch.logsumexp(negscores,-1),z.new_full((len(rows),),math.log(epsilon)))
            # Detached alpha; its cosine dependence is NOT differentiated.
            logalpha=negscores-denominator[:,None]+.5*nnegative.double().log()[:,None]
            weighted_negative=logits+logalpha
            logmass=torch.logsumexp(weighted_negative,-1)
            competition=torch.logaddexp(logmass,z.new_full(logmass.shape,math.log(epsilon)))
            delta=competition[:,None]-logits-weights.log()
            losses=F.softplus(delta).masked_fill(~positive,0)
            per_anchor=losses.sum(-1)/npositive
            aw=anchor_weights[rows]
            total+=(per_anchor*aw).sum()
            pmin=cosine.masked_fill(~positive,torch.inf).min(-1).values
            nmax=cosine.masked_fill(~negative,-torch.inf).max(-1).values
            stats[0]=torch.minimum(stats[0],pmin.min())
            stats[1]=torch.maximum(stats[1],nmax.max())
            stats[2]=torch.minimum(stats[2],(pmin-nmax).min())
            stats[3]=torch.maximum(stats[3],losses.max())
            stats[4]=torch.minimum(stats[4],weights.masked_fill(~positive,torch.inf).min())
            stats[5]=torch.maximum(stats[5],weights.masked_fill(~positive,-torch.inf).max())
            stats[6]=torch.maximum(stats[6],(weights-1).abs().masked_fill(~image_equal,0).max())
            if need_gradient:
                beta=delta.sigmoid().masked_fill(~positive,0)/npositive[:,None]
                # exp(weighted_negative - competition) includes the exact
                # epsilon attenuation, without a subtraction near unit mass.
                derivative=beta.sum(-1)[:,None]*(weighted_negative-competition[:,None]).exp()-beta
                derivative*=aw[:,None]/temperature
                gradient.index_add_(0,rows,derivative@z)
                gradient+=derivative.T@z[rows]
        if need_gradient:ctx.save_for_backward(gradient)
        ctx.mark_non_differentiable(stats)
        return total,stats

    @staticmethod
    @once_differentiable
    def backward(ctx,loss_gradient,stats_gradient):
        gradient,=ctx.saved_tensors
        return (gradient*loss_gradient,)+(None,)*10


def patch_vico(embeddings,labels,image_ids,source_ids,*,epoch=0,
               curriculum_epochs=20,temperature=.1,epsilon=1e-6,chunk_size=128,
               positive_weighting=True,distributed=True):
    """All valid patches; gather candidates, assign each anchor exactly once.

    Input [patch,D] is already stripped of invalid frames/letterbox patches.
    Distributed loss is world*local weighted anchor sum so DDP averaging yields
    exactly the global source-balanced gradient, including candidate gradients.
    The returned scalar needs a rank mean for a global reporting value.
    """
    if embeddings.ndim!=2 or len(embeddings)<1 or chunk_size<1 or curriculum_epochs<1:
        raise ValueError('Need nonempty [patch,D], positive chunk/curriculum sizes')
    if not math.isfinite(temperature) or temperature<=0 or not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('Temperature and epsilon must be positive and finite')
    for value in (labels,image_ids,source_ids):
        if value.shape!=(len(embeddings),) or value.dtype!=torch.long or value.device!=embeddings.device:
            raise ValueError('Patch metadata must be aligned long tensors on the feature device')
    if not torch.isfinite(embeddings).all() or (embeddings.norm(dim=-1)==0).any():
        raise ValueError('Patch embeddings must be finite and nonzero')
    world=dist.get_world_size() if distributed and dist.is_initialized() else 1
    rank=dist.get_rank() if world>1 else 0
    if world>1:
        from torch.distributed.nn.functional import all_gather
        counts=[None]*world
        dist.all_gather_object(counts,len(embeddings))
        maximum=max(counts)
        def gather(value,grad=False):
            padded=torch.cat((value,value.new_zeros((maximum-len(value),*value.shape[1:]))))
            if grad:parts=all_gather(padded)
            else:
                parts=[torch.empty_like(padded) for _ in range(world)]
                dist.all_gather(parts,padded)
            return torch.cat([part[:count] for part,count in zip(parts,counts)])
        embeddings=gather(embeddings,True)
        labels,image_ids,source_ids=(gather(value) for value in (labels,image_ids,source_ids))
        offset=sum(counts[:rank])
        anchors=torch.arange(offset,offset+counts[rank],device=embeddings.device)
    else:anchors=torch.arange(len(embeddings),device=embeddings.device)
    if set(labels.tolist())!={0,1}:raise ValueError('Need binary clean/spill classes')
    weights=source_balanced_weights(labels,image_ids,source_ids)*world
    with torch.autocast(device_type=embeddings.device.type,enabled=False):
        z=F.normalize(embeddings.double(),dim=-1)
        loss,stats=_StreamedCompetition.apply(z,labels,image_ids,anchors,weights,
            temperature,epsilon,epoch,curriculum_epochs,chunk_size,positive_weighting)
    names=('minimum_positive_cosine','maximum_negative_cosine','minimum_pair_separation',
           'max_positive_pair_loss','positive_weight_min','positive_weight_max',
           'same_image_weight_max_error','owned_anchors')
    return loss,dict(zip(names,stats.unbind()))


class PatchProjector(nn.Module):
    """Pointwise projection only: no attention, coordinates, or spatial mixing."""
    def __init__(self,width=1024,hidden=256,embedding_dim=256):
        super().__init__()
        self.width=width
        self.network=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,hidden),
                                   nn.GELU(),nn.Linear(hidden,embedding_dim))

    def forward(self,patches):
        if patches.shape[-1]!=self.width:raise ValueError('Backbone feature width mismatch')
        return F.normalize(self.network(patches.float()),dim=-1)


def local_feature_retention(q,teacher,valid):
    """New local objective: equal sources, then mean valid view/patch targets.

    The teacher is the immutable initial pointwise projection of frozen V-JEPA
    tokens. This is distinct from the original temporal retention objective.
    For unit vectors, half squared distance equals one minus cosine, with an
    exactly zero loss/gradient at initialization. Targets never receive grads.
    """
    if q.ndim!=4 or q.shape!=teacher.shape or valid.shape!=q.shape[:-1] or valid.dtype!=torch.bool:
        raise ValueError('Need q/teacher [source,view,patch,D] and boolean validity')
    counts=valid.sum((1,2))
    if (counts==0).any():raise ValueError('Every source needs valid patches')
    difference=(q-teacher.detach()).masked_fill(~valid[...,None],0)
    return (difference.square().sum(-1).sum((1,2))/(2*counts)).mean()
