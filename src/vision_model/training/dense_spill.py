"""Decision-connected fixed-mean readout and within-positive supervision guards."""
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


class DenseSpillReadout(nn.Module):
    """No instance selection: every valid patch enters one fixed mean decision.

    Local logits and episode margin use the SAME embeddings and prototypes.
    Contributions average exactly to that episode margin. They remain feature
    contributions, not independent pixel probabilities or segmentation masks.
    """
    def __init__(self, width=1024, hidden=128, dimension=128, temperature=.1, normalize_patches=True):
        super().__init__()
        if min(width,hidden,dimension)<1 or not math.isfinite(temperature) or temperature<=0:
            raise ValueError('Invalid dense readout dimensions/temperature')
        self.projector=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,dimension))
        self.register_buffer('prototypes',F.normalize(torch.randn(2,dimension),dim=-1))
        self.temperature=temperature
        self.normalize_patches=normalize_patches
        self.configuration=dict(width=width,hidden=hidden,dimension=dimension,temperature=temperature,
                                normalize_patches=normalize_patches)

    @torch.no_grad()
    def set_prototypes(self, prototypes):
        if prototypes.shape!=self.prototypes.shape or not torch.isfinite(prototypes).all() or (prototypes.norm(dim=-1)==0).any():
            raise ValueError('Need two finite nonzero training-class prototypes')
        self.prototypes.copy_(F.normalize(prototypes.detach().to(self.prototypes),dim=-1))

    def forward(self, patches, valid, content):
        if patches.ndim!=4 or valid.shape!=patches.shape[:2] or content.shape!=patches.shape[:1]+patches.shape[2:3]:
            raise ValueError('Expected [batch,time,patch,width], valid and spatial supports')
        mask=valid[...,None]&content[:,None,:]
        if not mask.flatten(1).any(1).all():raise ValueError('Every source needs observed content')
        # Padding NaNs never enter a trainable layer or a reduction.
        q=self.projector(patches.masked_fill(~mask[...,None],0).float())
        if self.normalize_patches:q=F.normalize(q,dim=-1)
        q=q.masked_fill(~mask[...,None],0)
        mu=q.sum((1,2))/mask.sum((1,2))[:,None]
        norm=mu.norm(dim=-1).clamp_min(1e-12)
        z=mu/norm[:,None]
        direction=F.normalize(self.prototypes,dim=-1)[1]-F.normalize(self.prototypes,dim=-1)[0]
        raw=q@direction
        return dict(embedding=z,margin=z@direction,local_logits=raw/self.temperature,
                    contributions=raw/norm[:,None,None],valid=mask)


def candidate_coverage(targets, labels, train_indices, *, minimum_fraction=.5):
    """Training-only coverage gate; never fabricate cells to meet the gate."""
    if targets.ndim!=4 or targets.shape[1]!=2 or len(labels)!=len(targets):
        raise ValueError('Need [source,2,time,patch] candidates')
    if targets.dtype!=torch.long or not ((targets==-1)|(targets==0)|(targets==1)).all():
        raise ValueError('Candidates must be long -1/0/1')
    if not 0<minimum_fraction<=1:raise ValueError('Invalid coverage requirement')
    positive=train_indices[labels[train_indices]==1]
    rows=[]
    for index in positive.tolist():
        # A mirror duplicate must not compensate for missing identity support.
        foreground=int((targets[index,0]==1).sum())
        background=int((targets[index,0]==0).sum())
        frames=((targets[index,0]==1).any(-1)&(targets[index,0]==0).any(-1)).nonzero().flatten().tolist()
        rows.append(dict(index=index,foreground=foreground,background=background,
                         frames_with_both=frames,both=bool(frames)))
    covered=sum(r['both'] for r in rows)
    fraction=covered/max(1,len(rows))
    return dict(positive_training_sources=len(rows),sources_with_both=covered,
        fraction_with_both=fraction,minimum_fraction=minimum_fraction,
        ready_for_training=bool(rows) and fraction>=minimum_fraction,
        rows=rows,scope='Unverified target coverage only; never localization accuracy')


def require_candidate_coverage(targets, labels, train_indices, *, minimum_fraction=.5):
    report=candidate_coverage(targets,labels,train_indices,minimum_fraction=minimum_fraction)
    if not report['ready_for_training']:
        raise ValueError(f"Within-spill background coverage failed: {report['sources_with_both']}/{report['positive_training_sources']} positive training sources have both FG and BG; require {minimum_fraction:.0%}. Improve automatic references; do not force patch labels or request manual annotations.")
    return report


def source_balanced_partial_loss(logits, targets, labels):
    """Equal source means within clean-BG, cold-BG, cold-FG strata.

    Cold sources without both candidates abstain from the local term. Their
    episode labels may still enter episode ViCo. DDP normalization uses global
    source counts, so ranks with no member of a stratum remain correct.
    """
    if logits.shape!=targets.shape or logits.ndim!=4 or len(labels)!=len(logits):
        raise ValueError('Need aligned [source,view,time,patch] tensors')
    if targets.dtype!=torch.long or not ((targets==-1)|(targets==0)|(targets==1)).all():raise ValueError('Bad targets')
    if not torch.isfinite(logits[targets>=0]).all():raise ValueError('Nonfinite active local scores')
    terms=[[],[],[]]
    for i in range(len(labels)):
        negative,positive=targets[i]==0,targets[i]==1
        if labels[i]==0:
            if positive.any():raise ValueError('Clean source cannot have foreground')
            if negative.any():terms[0].append(F.softplus(logits[i][negative]).double().mean())
        else:
            paired=negative.any(-1)&positive.any(-1)
            negative=negative&paired[...,None];positive=positive&paired[...,None]
            if negative.any() and positive.any():
                terms[1].append(F.softplus(logits[i][negative]).double().mean())
                terms[2].append(F.softplus(-logits[i][positive]).double().mean())
    world=dist.get_world_size() if dist.is_initialized() else 1
    counts=torch.tensor([len(x) for x in terms],device=logits.device,dtype=torch.float64)
    if world>1:dist.all_reduce(counts)
    if (counts==0).any():raise ValueError('Need clean BG and within-cold FG/BG; local term must abstain')
    zero=logits.masked_fill(targets<0,0).sum()*0
    loss=sum((torch.stack(t).sum() if t else zero)*world/count for t,count in zip(terms,counts))/3
    return loss,dict(clean_background_sources=int(counts[0]),cold_background_sources=int(counts[1]),cold_foreground_sources=int(counts[2]))


def dense_consensus_candidates(raw, residual, valid, content, labels, calibration):
    """Both clean-calibrated tails plus robust excess in at least one geometry.

    Requiring 3-MAD excess in BOTH feature geometries can reject every event,
    although each geometry detects distinct spatial departures. Absolute
    normal-reference novelty must still agree in both independently encoded
    views and both geometries; this never forces an area or a positive cell.
    """
    from .normal_patch_reference import automatic_candidates
    if raw.shape!=residual.shape or raw.ndim!=4 or len(labels)!=len(raw):
        raise ValueError('Need aligned raw/residual novelty maps')
    side=math.isqrt(raw.shape[-1])
    if side*side!=raw.shape[-1]:raise ValueError('Need square patch grids')
    robust=[];absolute=[]
    for mode,scores in [('raw',raw),('residual',residual)]:
        low,high=calibration[mode]['0.95'],calibration[mode]['0.99']
        if high<=low:raise ValueError('Degenerate clean calibration; abstain')
        robust.append(automatic_candidates(scores,valid,content,low=low,high=high))
        reflected=scores[:,1].reshape(len(scores),scores.shape[2],side,side).flip(-1).flatten(2)
        absolute.append((scores[:,0]>high)&(reflected>high))
    foreground=absolute[0]&absolute[1]&((robust[0][:,0]==1)|(robust[1][:,0]==1))
    background=(robust[0][:,0]==0)&(robust[1][:,0]==0)
    eligible=valid[...,None]&content[:,None,:]
    identity=torch.full_like(raw[:,0],-1,dtype=torch.long)
    identity[foreground&eligible]=1;identity[background&eligible]=0
    for i in torch.where(labels==0)[0]:identity[i][eligible[i]]=0
    mirrored=identity.reshape(len(raw),raw.shape[2],side,side).flip(-1).flatten(2)
    return torch.stack((identity,mirrored),1)
