"""Finite-field thermal readout and unverified same-image region comparisons.

This optional local detector does not alter V-JEPA normalization, temporal
teachers or the four-loss trajectory. It is not backbone attention or MIL.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from scipy import ndimage


@dataclass(frozen=True)
class LocalThermalSettings:
    radii: tuple = (4, 8, 16, 32)
    stride: int = 8
    noise_floor: float = 1 / 255
    rank_levels: tuple = (.02, .04, .08, .12, .18, .26, .36, .48)
    minimum_pixels: int = 32
    minimum_parent_overlap: float = .65
    minimum_persistence_gain: float = 1.15
    ring_radius: int = 7
    minimum_ring_pixels: int = 32
    minimum_contrast_snr: float = 1.5

    def state_dict(self):
        return asdict(self)


def _box(value, radius):
    k=2*radius+1
    value=F.avg_pool2d(F.pad(value,(radius,radius,0,0),mode='reflect'),(1,k),stride=1)
    return F.avg_pool2d(F.pad(value,(0,0,radius,radius),mode='reflect'),(k,1),stride=1)


def local_relative_features(gray, settings=LocalThermalSettings()):
    """[frames,1,H,W] -> [frames,8,H/8,W/8], bounded radius39.

    No global mean, CDF, positional token, scene features, clock or batch norm.
    Offset/gain invariance holds when the explicit sensor-noise floor is inactive.
    Processing one frame never reads another frame. Reflect padding uses only
    observed content; the original temporal encoder's letterbox is not reused.
    """
    if gray.ndim!=4 or gray.shape[1]!=1 or not gray.is_floating_point():
        raise ValueError('Need floating grayscale [frame,1,height,width]')
    if not torch.isfinite(gray).all() or min(gray.shape[-2:])<=max(settings.radii):
        raise ValueError('Nonfinite pixels or frame smaller than receptive field')
    if any(s%settings.stride for s in gray.shape[-2:]):
        raise ValueError('Content dimensions must divide the local stride')
    fields=[]
    for radius in settings.radii:
        mean=_box(gray,radius)
        scale=(_box(gray.square(),radius)-mean.square()).clamp_min(settings.noise_floor**2).sqrt()
        fields.append(((mean-gray)/scale).clamp(-5,5))
        padded=F.pad(gray,(radius,radius,radius,radius),mode='reflect')
        h,w=gray.shape[-2:]
        neighbors=[padded[:,:,radius+dy:radius+dy+h,radius+dx:radius+dx+w]
                   for dy,dx in [(-radius,-radius),(-radius,0),(-radius,radius),(0,-radius),
                                 (0,radius),(radius,-radius),(radius,0),(radius,radius)]]
        # Ties are neutral, not artificial foreground in flat fields.
        rank=sum((u>gray).to(gray.dtype)+.5*(u==gray).to(gray.dtype) for u in neighbors)/8
        fields.append(2*rank-1)
    return F.avg_pool2d(torch.cat(fields,1),settings.stride,stride=settings.stride)


class LocalThermalReadout(nn.Module):
    """Pointwise trainable readout of eight genuinely local measurements."""
    def __init__(self, channels=8, hidden=24, kind='pointwise'):
        super().__init__()
        if kind not in ['pointwise','local_conv']:raise ValueError('Unknown local readout')
        self.configuration=dict(channels=channels,hidden=hidden,kind=kind)
        self.extra_receptive_radius=0 if kind=='pointwise' else 16
        if kind=='pointwise':
            self.net=nn.Sequential(nn.Linear(channels,hidden),nn.GELU(),nn.Linear(hidden,1))
        else:
            # Two grid-local layers, without upsampling, global pooling or
            # normalization across positions. Adds 2*8 pixels to field radius.
            self.net=nn.Sequential(nn.Conv2d(channels,hidden,3,padding=1,padding_mode='reflect'),nn.GELU(),
                nn.Conv2d(hidden,hidden,3,padding=1,padding_mode='reflect'),nn.GELU(),nn.Conv2d(hidden,1,1))

    def forward(self, features):
        if features.ndim!=4 or features.shape[1]!=self.configuration['channels']:
            raise ValueError('Expected [frame,channel,height,width]')
        if self.configuration['kind']=='local_conv':return self.net(features.float()).squeeze(1)
        return self.net(features.permute(0,2,3,1).float()).squeeze(-1)


def compact_pair(pair):
    result={k:v for k,v in pair.items() if k not in ['region','ring']}
    for name in ['region','ring']:
        value=pair[name].flatten();idx=value.nonzero().flatten()
        result[name+'_indices']=idx.to(torch.int16)
        result[name+'_values']=value[idx].half()
    return result


def pair_masks(pair, shape):
    if 'region' in pair:return pair['region'],pair['ring']
    result=[]
    for name in ['region','ring']:
        value=torch.zeros(math.prod(shape))
        value[pair[name+'_indices'].long()]=pair[name+'_values'].float()
        result.append(value.reshape(shape))
    return tuple(result)


def region_pairs(gray, settings=LocalThermalSettings()):
    """All supported persistent dark components; never a liquid ground truth.

    Region hierarchy is within this image. It never queries other clean scenes,
    assumes a clean first frame, selects a favorable position, or forces an area.
    Partial edge rings may contribute when enough observed ring pixels remain.
    """
    image=np.asarray(gray,dtype=np.float32)
    if image.ndim!=2 or not np.isfinite(image).all():raise ValueError('Need finite grayscale image')
    h,w=image.shape;stride=settings.stride
    if h%stride or w%stride:raise ValueError('Bad local grid')
    thresholds=np.quantile(image,settings.rank_levels)
    levels=[ndimage.label(image<=threshold)[0] for threshold in thresholds]
    pairs=[];seen=[]
    for j in range(len(levels)-1):
        labels,parent=levels[j:j+2]
        sizes=np.bincount(labels.ravel())
        parent_sizes=np.bincount(parent.ravel())
        for k in np.where(sizes>=settings.minimum_pixels)[0]:
            if k==0:continue
            region=labels==k
            parent_id=int(parent[region][0]);overlap=float(sizes[k]/parent_sizes[parent_id])
            gain=overlap/(settings.rank_levels[j]/settings.rank_levels[j+1])
            if overlap<settings.minimum_parent_overlap or gain<settings.minimum_persistence_gain:continue
            # Exclude duplicate nested instances without selecting a desired ROI.
            if any(np.count_nonzero(region&old)/np.count_nonzero(region|old)>.9 for old in seen):continue
            ring=ndimage.binary_dilation(region,iterations=settings.ring_radius)&~region
            # Other equally cold components cannot become background anchors.
            ring &= image>thresholds[j]
            if ring.sum()<settings.minimum_ring_pixels:continue
            a=image[region];b=image[ring]
            noise=max(1.4826*float(np.median(np.abs(b-np.median(b)))),2/255)
            contrast=float((np.median(b)-np.median(a))/noise)
            if contrast<settings.minimum_contrast_snr:continue
            def reduce(mask):
                return torch.from_numpy(mask.astype(np.float32).reshape(h//stride,stride,w//stride,stride).mean((1,3)))
            pairs.append(dict(region=reduce(region),ring=reduce(ring),
                confidence=min(1.,contrast/6)*min(1.,overlap),pixels=int(region.sum()),
                contrast_snr=contrast,level=j,persistence_gain=gain))
            seen.append(region)
    return pairs


def local_pair_loss(logits, labels, source_ids, valid, pairs, *, margin=1.):
    """Source-balanced clean absence + same-source/frame pair ranking.

    Inputs [source,view,frame,H,W]; pairs map local source index to frame lists.
    Unknown positive sources abstain. DDP uses global source counts, with no
    padded sample repeats and gradients equivalent to a global source mean.
    """
    if logits.ndim!=5 or valid.shape!=(len(logits),logits.shape[2]) or len(labels)!=len(logits):
        raise ValueError('Need aligned source/view/frame scores and valid frames')
    clean=[];positive=[]
    for i in range(len(logits)):
        if labels[i]==0:
            clean.append(F.softplus(logits[i,:,valid[i]]).double().mean())
        else:
            numerator=logits[i].sum()*0;denominator=0.
            for t in torch.where(valid[i])[0].tolist():
                for pair in pairs[int(source_ids[i])][t]:
                    r,a=pair_masks(pair,logits.shape[-2:]);r=r.to(logits);a=a.to(logits)
                    weight=float(pair['confidence'])
                    if not weight>0 or r.sum()<=0 or a.sum()<=0:raise ValueError('Empty or invalid region pair')
                    for view in range(logits.shape[1]):
                        rr=r if view==0 else r.flip(-1);aa=a if view==0 else a.flip(-1)
                        gap=(logits[i,view,t]*rr).sum()/rr.sum()-(logits[i,view,t]*aa).sum()/aa.sum()
                        numerator=numerator+weight*F.softplus(margin-gap).double()
                        denominator+=weight
            if denominator:positive.append(numerator/denominator)
    world=dist.get_world_size() if dist.is_initialized() else 1
    counts=torch.tensor([len(clean),len(positive)],device=logits.device,dtype=torch.float64)
    if world>1:dist.all_reduce(counts)
    if (counts==0).any():raise ValueError('Need clean sources and positive same-image pairs globally')
    zero=logits.sum()*0
    terms=[(torch.stack(values).sum() if values else zero)*world/count for values,count in zip([clean,positive],counts)]
    return terms[0],terms[1],dict(clean_sources=int(counts[0]),paired_positive_sources=int(counts[1]))


def packed_local_pairs(pairs, source_ids, valid, shape, device):
    """Pack all component comparisons once; no Python GPU launches per pair."""
    frame_indices=[];weights=[];differences=[];owners=[]
    times=valid.shape[1]
    for slot,index in enumerate(source_ids):
        for t in torch.where(valid[slot])[0].tolist():
            for pair in pairs[int(index)][t]:
                r,a=pair_masks(pair,shape)
                delta=r/r.sum()-a/a.sum()
                for view in (0,1):
                    differences.append((delta if view==0 else delta.flip(-1)).flatten())
                    frame_indices.append((slot*2+view)*times+t)
                    weights.append(float(pair['confidence']));owners.append(slot)
    w=torch.tensor(weights,device=device);owner=torch.tensor(owners,device=device)
    owner=owner.long()
    denominator=torch.zeros(len(source_ids),device=device).index_add_(0,owner,w)
    delta=torch.stack(differences).to(device) if differences else torch.empty((0,math.prod(shape)),device=device)
    return dict(frames=torch.tensor(frame_indices,device=device,dtype=torch.long),delta=delta,
        weights=w/denominator[owner],owners=owner,supported=denominator>0)


def packed_pair_terms(logits, labels, valid, packed):
    selected=logits.reshape(-1,math.prod(logits.shape[-2:]))[packed['frames']]
    gap=(selected*packed['delta']).sum(1)
    values=F.softplus(1-gap)*packed['weights']
    source=torch.zeros(len(logits),device=logits.device).index_add_(0,packed['owners'],values)
    supported=packed['supported']&(labels==1);clean=labels==0
    # Pairs from clean images are retained only as diagnostics, never positive supervision.
    counts=torch.stack((clean.sum(),supported.sum())).double()
    if dist.is_initialized():dist.all_reduce(counts)
    if (counts==0).any():raise ValueError('Need globally clean and supported positive sources')
    world=dist.get_world_size() if dist.is_initialized() else 1
    clean_scores=F.softplus(logits[clean]).masked_fill(~valid[clean,None,:,None,None],0)
    clean_source=clean_scores.sum((1,2,3,4))/(valid[clean].sum(1)*logits.shape[1]*math.prod(logits.shape[-2:]))
    return clean_source.double().sum()*world/counts[0],source[supported].double().sum()*world/counts[1]
