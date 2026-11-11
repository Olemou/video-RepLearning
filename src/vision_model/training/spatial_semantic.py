"""Decoder-free semantic attention alongside an unchanged trajectory teacher.

This is a repository-specific adaptation, not MCTformer or SEAM reproduction.
Class queries are conditioned on frozen training-only episode prototypes. All
queries are used in a fixed order for every video; labels never route inputs.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from .vico import vico_pairwise_contrastive


class SpatialSemanticBranch(nn.Module):
    """Spatial encoder, prototype-conditioned queries, and episode projection.

    Input [B,T,H,W,D] contains only the last tubelet of each causal prefix.
    The spatial encoder independently processes each observed time. Attention
    probabilities actually aggregate frozen patch values into the embedding.
    """
    def __init__(self, class_prototypes: torch.Tensor, *, width=1024,
                 hidden=128, embedding_dim=128, heads=4, layers=1):
        super().__init__()
        if class_prototypes.ndim != 2 or class_prototypes.shape[0] != 2:
            raise ValueError("Need two training-only class prototypes [2,E]")
        if not torch.isfinite(class_prototypes).all() or (class_prototypes.norm(dim=-1)==0).any():
            raise ValueError("Class prototypes must be finite and nonzero")
        if hidden % heads or layers < 1:
            raise ValueError("Invalid spatial Transformer architecture")
        self.width, self.hidden, self.heads = width, hidden, heads
        self.register_buffer("class_prototypes", F.normalize(class_prototypes.detach().float().clone(), dim=-1))
        self.patch = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden))
        self.position = nn.Linear(2, hidden, bias=False)
        self.class_projection = nn.Linear(class_prototypes.shape[1], hidden)
        self.class_offset = nn.Parameter(torch.zeros(2, hidden))
        block = nn.TransformerEncoderLayer(hidden, heads, hidden*2, dropout=0.,
                                           batch_first=True, norm_first=True)
        self.spatial = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.semantic = nn.Sequential(nn.LayerNorm(2*width), nn.Linear(2*width, hidden),
                                      nn.GELU(), nn.Linear(hidden, embedding_dim))

    def forward(self, patches, valid, content):
        if patches.ndim != 5 or patches.shape[-1] != self.width:
            raise ValueError("Expected patches [B,T,H,W,D]")
        b,t,h,w,d = patches.shape
        if valid.shape != (b,t) or content.shape != (b,h,w) or valid.dtype != torch.bool or content.dtype != torch.bool:
            raise ValueError("Expected boolean valid [B,T], content [B,H,W]")
        if not valid[:,0].all() or ((~valid[:,:-1]) & valid[:,1:]).any() or not content.flatten(1).any(1).all():
            raise ValueError("Need nonempty valid temporal prefixes and spatial content")
        support = valid[:,:,None,None] & content[:,None]
        # Ignore padded/letterbox values before every trainable operation.
        values = patches.masked_fill(~support[...,None],0).float()
        if not torch.isfinite(values).all():
            raise ValueError("Observed patch features must be finite")
        rows,cols = torch.meshgrid(torch.linspace(-1,1,h,device=patches.device),
                                    torch.linspace(-1,1,w,device=patches.device),indexing="ij")
        coords = torch.stack((rows,cols),-1).reshape(h*w,2)
        tokens = self.patch(values).reshape(b*t,h*w,self.hidden) + self.position(coords)[None]
        classes = self.class_projection(self.class_prototypes) + self.class_offset
        tokens = torch.cat((classes[None].expand(b*t,-1,-1),tokens),1)
        mask = support.reshape(b*t,h*w)
        # Class tokens ensure a nonempty attention-key set even at padded times.
        padding = torch.cat((torch.zeros(b*t,2,dtype=torch.bool,device=patches.device),~mask),1)
        encoded = self.norm(self.spatial(tokens,src_key_padding_mask=padding))
        q = self.query(encoded[:,:2]).reshape(b*t,2,self.heads,-1).transpose(1,2)
        k = self.key(encoded[:,2:]).reshape(b*t,h*w,self.heads,-1).transpose(1,2)
        logits = (q @ k.transpose(-1,-2))/math.sqrt(self.hidden/self.heads)
        # A dummy key avoids all-minus-infinity softmax at invalid tubelets.
        safe = mask.clone()
        safe[~safe.any(1),0]=True
        attention = logits.masked_fill(~safe[:,None,None],-torch.inf).softmax(-1)
        attention = attention.masked_fill(~mask[:,None,None],0)
        attention = attention.reshape(b,t,self.heads,2,h,w)
        maps = attention.mean(2)
        pooled = torch.einsum("btcij,btijd->btcd",maps,values)
        episode = pooled.sum(1)/valid.sum(1)[:,None,None]
        embedding = F.normalize(self.semantic(episode.flatten(1)),dim=-1)
        return dict(embedding=embedding, attention=maps,
                    head_attention=attention,
                    class_features=pooled)


def mirror_total_variation(attention, valid, content):
    """Symmetric TV on real paired encodes, mean over classes/times/sources.

    Maps [N,2,T,C,H,W] have identity and actual re-encoded mirror views.
    Spatial support must reflect exactly on the checkpoint grid. Invalid
    positions contribute neither error nor normalization mass.
    """
    if attention.ndim != 6 or attention.shape[1]!=2:
        raise ValueError("Need paired attention [N,2,T,C,H,W]")
    n,_,t,_,h,w=attention.shape
    if valid.shape!=(n,t) or content.shape!=(n,h,w):
        raise ValueError("Mirror validity shapes disagree")
    support=content & content.flip(-1)
    if not support.flatten(1).any(1).all() or not valid.any(1).all():
        raise ValueError("Empty aligned support")
    support=valid[:,:,None,None] & support[:,None]
    a=attention[:,0].masked_fill(~support[:,:,None],0)
    b=attention[:,1].flip(-1).masked_fill(~support[:,:,None],0)
    # Intersected support is shared; normalize within it for partial masks.
    def normalize(x):
        return x/x.sum((-2,-1),keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    difference=(normalize(a)-normalize(b)).abs().sum((-2,-1))*.5
    per_time=difference.mean(-1).masked_fill(~valid,0)
    return (per_time.sum(1)/valid.sum(1)).mean()


def attention_diagnostics(attention, valid, content):
    """Per-source entropy, effective area and class similarity; no mask claims."""
    n,v,t,c,h,w=attention.shape
    support=valid[:,None,:,None,None,None] & content[:,None,None,None]
    p=attention.masked_fill(~support,0)
    entropy=-(p*p.clamp_min(1e-30).log()).sum((-2,-1))
    counts=content.sum((-2,-1)).to(p.dtype)
    normalized=entropy/counts.clamp_min(2).log()[:,None,None,None]
    def reduce(x):
        return (x*valid[:,None,:,None]).sum((1,2,3))/(valid.sum(1)*v*c)
    a,b=p[:,:,:,0].flatten(-2),p[:,:,:,1].flatten(-2)
    similarity=F.cosine_similarity(a,b,dim=-1)
    return dict(normalized_entropy=reduce(normalized),effective_patches=reduce(entropy.exp()),
                maximum_probability=reduce(p.flatten(-2).max(-1).values),
                class_map_cosine=(similarity*valid[:,None]).sum((1,2))/(valid.sum(1)*v))


def spatial_objective(output, labels, source_ids, valid, content, *, epoch,
                      mirror_weight=0., curriculum_epochs=20):
    """Global ViCo + mirror; variable per-rank source counts are unpadded."""
    n=labels.numel()
    if output["embedding"].shape[0]!=n*2:
        raise ValueError("Need two views per source")
    z=output["embedding"]
    y=labels.repeat_interleave(2)
    sources=source_ids.repeat_interleave(2)
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn.functional import all_gather
        counts=[None]*dist.get_world_size()
        dist.all_gather_object(counts,n)
        maximum=max(counts)
        def gather(x,grad=False):
            pad=torch.cat((x,x.new_zeros(2*(maximum-n),*x.shape[1:])),0)
            if grad:
                pieces=all_gather(pad)
            else:
                pieces=[torch.empty_like(pad) for _ in counts]
                dist.all_gather(pieces,pad)
            return torch.cat([piece[:2*count] for piece,count in zip(pieces,counts)])
        z,y,sources=gather(z,True),gather(y),gather(sources)
        # Local auxiliary means need weighting when DDP averages rank gradients.
        weight=dist.get_world_size()*n/sum(counts)
    else:
        weight=1.
    semantic,details=vico_pairwise_contrastive(z,y,source_ids=sources,epoch=epoch,
                                              curriculum_epochs=curriculum_epochs)
    maps=output["attention"].reshape(n,2,*output["attention"].shape[1:])
    mirror=mirror_total_variation(maps,valid,content)
    total=semantic+float(mirror_weight)*weight*mirror
    details.update(semantic_loss=semantic,mirror_loss=mirror,
                   optimized_loss=total)
    return total,details


class SpatialPairedForward(nn.Module):
    """One DDP forward owns all memory-bounded source microbatches."""
    def __init__(self,branch,microbatch_sources=4):
        super().__init__()
        self.branch=branch
        if microbatch_sources<1:raise ValueError("Microbatch size must be positive")
        self.microbatch_sources=microbatch_sources

    def forward(self,bank,valid,content):
        n,views,t,h,w,d=bank.shape
        if views!=2:raise ValueError("Expected paired views")
        parts=[]
        for start in range(0,n,self.microbatch_sources):
            stop=min(start+self.microbatch_sources,n)
            size=stop-start
            parts.append(self.branch(bank[start:stop].reshape(size*2,t,h,w,d).float(),
                valid[start:stop,None].expand(-1,2,-1).reshape(size*2,t),
                content[start:stop,None].expand(-1,2,-1,-1).reshape(size*2,h,w)))
        return {k:torch.cat([p[k] for p in parts],0) for k in parts[0]}
