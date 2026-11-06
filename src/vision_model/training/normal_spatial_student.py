"""Normal-only cross-layer patch correspondence. Not a UniNet reproduction.

Only a pointwise embedding projector is trained; there is no image/mask decoder.
Teacher targets are frozen projections from another existing V-JEPA layer.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .normal_patch_reference import local_residual


class NormalSpatialProjector(nn.Module):
    def __init__(self,width=1024,hidden=128,out_dim=128):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,out_dim))

    def forward(self,x):
        return F.normalize(self.net(x.float()),dim=-1)


def correspondence_loss(query,target,*,contrast=True,temperature=.1,margin=.05):
    """Same-position positives, other positions compete, on clean samples only.

    Call with [valid-view-time,content-patch,embedding]. No video label is
    broadcast over a spill clip. Targets and all mining statistics are detached.
    """
    if query.ndim!=3 or query.shape!=target.shape or query.shape[1]<2:
        raise ValueError('Need matching nonempty [frame,patch,embedding]')
    if temperature<=0 or not 0<=margin<=1:raise ValueError('Invalid contrast configuration')
    target=F.normalize(target.detach(),dim=-1);query=F.normalize(query,dim=-1)
    cosine=(1-(query*target).sum(-1)).mean()
    similarity=query@target.transpose(-2,-1)/temperature
    diagonal=similarity.log_softmax(-1).diagonal(dim1=-2,dim2=-1)
    spatial=-diagonal.mean();boundary=F.relu(margin-diagonal.exp()).mean()
    loss=.7*cosine+.3*spatial+boundary if contrast else cosine
    return loss,dict(cosine_loss=cosine,spatial_contrast_loss=spatial,margin_loss=boundary)


def patch_inputs(tokens,support,*,mean=None,std=None,projection=None):
    x=local_residual(tokens.float(),support)
    if mean is not None:x=(x-mean.to(x))/std.to(x)
    if projection is not None:x=F.normalize(x@projection.to(x),dim=-1)
    return x
