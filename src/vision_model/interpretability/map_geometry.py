"""Invert the official full-field-of-view preprocessing for audit figures."""
import torch
import torch.nn.functional as F


def letterbox_bounds(source_height,source_width,size=256):
    scale=min(size/source_height,size/source_width)
    h=max(1,min(size,round(source_height*scale)))
    w=max(1,min(size,round(source_width*scale)))
    top,left=(size-h)//2,(size-w)//2
    return top,left,h,w


def inverse_letterbox_map(grid,source_height,source_width,size=256,*,interpolation='bilinear'):
    """Interpolate scalar values before colormapping, then remove padding."""
    if grid.ndim!=2:raise ValueError('Expected a native two-dimensional scalar grid')
    top,left,h,w=letterbox_bounds(source_height,source_width,size)
    if interpolation not in ('bilinear','nearest'):raise ValueError('Unsupported map interpolation')
    kwargs=dict(mode=interpolation)
    if interpolation=='bilinear':kwargs['align_corners']=False
    canvas=F.interpolate(grid.float()[None,None],size=(size,size),**kwargs)
    content=canvas[:,:,top:top+h,left:left+w]
    return F.interpolate(content,size=(source_height,source_width),**kwargs)[0,0]


def blend_rgb(frame,color,alpha=.4):
    if frame.shape!=color.shape or frame.shape[-1]!=3 or not 0<=alpha<=1:raise ValueError('Expected matching RGB images and alpha in [0,1]')
    return ((1-alpha)*frame.float()+alpha*color.float()).round().clamp(0,255).to(torch.uint8)


def nonnegative_evidence_colors(values, *, scale):
    """Fixed blue/cyan/green/yellow/red lookup: zero is blue, scale is red.

    No data-dependent normalization. These colors encode sensitivity magnitude,
    not spill probabilities. Return RGB uint8 with one last channel dimension.
    """
    if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
        raise ValueError('Need a positive finite shared scale')
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('Blue-to-red colors require finite nonnegative values')
    u=(values.float()/scale).clamp(0,1)
    red=(4*u-2).clamp(0,1)
    green=torch.minimum(4*u,4-4*u).clamp(0,1)
    blue=(2-4*u).clamp(0,1)
    return (255*torch.stack((red,green,blue),-1)).round().to(torch.uint8)


def spatial_evidence_overlay(frame,grid,*,scale,scope,signed=True,alpha=.4,checkpoint_size=256,style='magnitude'):
    """Explicit magnitude-opacity or fixed-opacity blue-to-red display.

    Magnitude mode preserves source RGB at zero. Blue-red mode intentionally
    colors zero blue: it encodes low sensitivity, not verified background.

    Contextual feature contributions and raw attention are deliberately rejected:
    neither is a local image attribution. Conditional attention-key gradients
    can be displayed only with that explicit scope, not as input-pixel proof.
    No per-image normalization, percentile threshold, ROI, blur or head choice.
    """
    if scope not in ('input_intervention','conditional_attention_key_gradient'):
        raise ValueError('Use input interventions or explicitly conditional key gradients; global/contextual class contributions are not localization overlays')
    if frame.ndim!=3 or frame.shape[-1]!=3 or frame.dtype!=torch.uint8:
        raise ValueError('Expected an RGB uint8 source image')
    if not torch.isfinite(torch.tensor(scale)) or scale<=0 or not 0<=alpha<=1:
        raise ValueError('Need a positive finite shared scale and alpha in [0,1]')
    if not torch.isfinite(grid).all() or (not signed and (grid<0).any()):
        raise ValueError('Invalid evidence values/sign convention')
    if style not in ('magnitude','blue_red') or (style=='blue_red' and signed):
        raise ValueError('Blue-to-red display requires explicitly nonnegative evidence')
    value=inverse_letterbox_map(grid,*frame.shape[:2],checkpoint_size,interpolation='nearest')
    if style=='blue_red':
        color=nonnegative_evidence_colors(value,scale=scale)
        return blend_rgb(frame,color,alpha)
    normalized=(value/scale).clamp(-1 if signed else 0,1)
    opacity=alpha*normalized.abs()[...,None]
    hue=torch.zeros_like(frame,dtype=torch.float32)
    hue[...,0]=255
    if signed:
        hue[...,0]=torch.where(normalized<0,0.,255.)
        hue[...,2]=torch.where(normalized<0,255.,0.)
    return ((1-opacity)*frame.float()+opacity*hue).round().clamp(0,255).to(torch.uint8)
