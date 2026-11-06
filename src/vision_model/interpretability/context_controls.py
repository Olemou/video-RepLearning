"""Systematic context interventions; protected cells are probes, not annotations."""
import torch
import torch.nn.functional as F
from .map_geometry import inverse_letterbox_map


def protected_tiles():
    """Twelve fixed4x4 tiles covering the 12x16 content grid of4:3 inputs."""
    result=[]
    for row in (2,6,10):
        for column in (0,4,8,12):
            protected=torch.zeros(16,16,dtype=torch.bool)
            protected[row:row+4,column:column+4]=True
            core=torch.zeros_like(protected)
            core[row+1:row+3,column+1:column+3]=True
            result.append(dict(row=row,column=column,protected=protected,core=core))
    return result


def change_distant_context(rgb,protected,fill):
    """Preserve a raw-image region exactly; modify only its complement.

    This is an inference diagnostic, never a supported training augmentation.
    The unmodified one-patch border protects the core from resize interpolation.
    """
    if rgb.ndim!=4 or rgb.dtype!=torch.uint8 or protected.shape!=(16,16):
        raise ValueError('Expected uint8 [T,3,H,W] and16x16 protected mask')
    height,width=rgb.shape[-2:]
    if width*3!=height*4:raise ValueError('The fixed probe layout requires4:3 sources')
    raw_mask=inverse_letterbox_map(protected.float(),height,width,interpolation='nearest')>.5
    if fill=='gain09':
        replacement=(rgb.float()*.9).round().byte()
    elif fill=='blur33':
        replacement=F.avg_pool2d(F.pad(rgb.float(),(16,16,16,16),mode='reflect'),33,1).round().clamp(0,255).byte()
    else:raise ValueError(fill)
    return torch.where(raw_mask[None,None],rgb,replacement),raw_mask
