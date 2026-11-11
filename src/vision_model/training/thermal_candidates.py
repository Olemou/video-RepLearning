"""Apparent cold-contrast teacher, not a calibrated temperature or liquid mask."""
import torch
import torch.nn.functional as F
from vision_model.interpretability.map_geometry import letterbox_bounds


def cold_contrast_grid(frames, *, size=384, patch=16):
    """Full-frame local darkness at fixed 31/63-pixel scales.

    No spatial ROI, crop or per-frame heatmap normalization. The grayscale
    inter-decile range compensates contrast gain in the teacher only; the
    pretrained encoder's checkpoint normalization is unchanged. Polarity is an
    explicit cold-video assumption. Dark shoes/objects remain possible errors.
    """
    if frames.ndim!=4 or frames.shape[1]!=3 or frames.dtype!=torch.uint8 or len(frames)%2:
        raise ValueError('Need paired uint8 RGB frames')
    if size%patch:raise ValueError('Resolution must match patch size')
    h,w=frames.shape[-2:];factor=min(size/h,size/w);rh,rw=round(h*factor),round(w*factor)
    gray=F.interpolate(frames.float().mean(1,keepdim=True)/255,size=(rh,rw),mode='bilinear',align_corners=False,antialias=True)
    quantiles=torch.quantile(gray.flatten(1),torch.tensor([.1,.9],device=gray.device),dim=1)
    gain=(quantiles[1]-quantiles[0]).clamp_min(10/255)[:,None,None,None]
    fields=[]
    for kernel in (31,63):
        radius=kernel//2
        local=F.avg_pool2d(F.pad(gray,(radius,radius,0,0),mode='reflect'),(1,kernel),stride=1)
        local=F.avg_pool2d(F.pad(local,(0,0,radius,radius),mode='reflect'),(kernel,1),stride=1)
        fields.append(((local-gray)/gain).clamp_min(0))
    value=torch.stack(fields).amax(0)
    canvas=torch.zeros(len(frames),1,size,size,device=gray.device)
    top,left=(size-rh)//2,(size-rw)//2;canvas[:,:,top:top+rh,left:left+rw]=value
    scores=F.avg_pool2d(canvas,patch,stride=patch).flatten(1)
    # Both actual sampled frames must support a tubelet's cold contrast.
    return scores.unflatten(0,(-1,2)).amin(1)


def cold_pixel_targets(scores, feature_targets, labels, valid, content, *, low, high):
    if not high>low>=0 or scores.shape!=(len(labels),valid.shape[1],content.shape[1]):
        raise ValueError('Need aligned cold scores and independent calibration')
    if feature_targets.shape!=(len(labels),2,valid.shape[1],content.shape[1]):raise ValueError('Bad feature candidates')
    allowed=valid[...,None]&content[:,None,:]
    identity=torch.full_like(scores,-1,dtype=torch.long)
    identity[allowed&(scores>high)]=1
    identity[allowed&(scores<low)&(feature_targets[:,0]==0)]=0
    for i in torch.where(labels==0)[0]:identity[i][allowed[i]]=0
    side=int(content.shape[1]**.5)
    if side*side!=content.shape[1]:raise ValueError('Expected square native grid')
    return torch.stack((identity,identity.reshape(len(labels),valid.shape[1],side,side).flip(-1).flatten(2)),1)


def cold_contrast_from_encoder_pixels(pixels, source_height, source_width):
    """Same local descriptor during caching and every perturbed image forward."""
    if pixels.ndim!=4 or pixels.shape[1]!=3 or pixels.shape[-1]!=pixels.shape[-2]:
        raise ValueError('Need normalized square encoder pixels')
    size=pixels.shape[-1]
    top,left,h,w=letterbox_bounds(source_height,source_width,size)
    mean=pixels.new_tensor([.485,.456,.406])[None,:,None,None]
    std=pixels.new_tensor([.229,.224,.225])[None,:,None,None]
    observed=((pixels[:,:,top:top+h,left:left+w]*std+mean)*255).round().clamp(0,255).to(torch.uint8)
    return cold_contrast_grid(observed,size=size)
