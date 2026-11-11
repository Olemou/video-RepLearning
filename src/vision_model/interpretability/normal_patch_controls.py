"""Label-blind equal-area image interventions for patch novelty diagnostics."""
import torch
import torch.nn.functional as F


def ranked_cells(field,content,count=26,*,lowest=False):
    if field.ndim!=1 or field.shape!=content.shape or count<1 or count>int(content.sum()):
        raise ValueError('Invalid native field/support/removal count')
    indices=torch.where(content)[0]
    return indices[torch.argsort(field[indices],descending=not lowest,stable=True)[:count]]


def remove_cells(pixels,cells,content,*,fill='mean',grid=(16,16)):
    """Perturb normalized model inputs; preserve pixels outside selected cells.

    These are offline validation interventions, never training augmentation.
    Padding cannot be selected or used to determine the observed-content mean.
    """
    if pixels.ndim!=4 or content.shape!=(grid[0]*grid[1],):raise ValueError('Shape mismatch')
    if len(cells.unique())!=len(cells) or not content[cells].all():raise ValueError('Need distinct content cells')
    mask=torch.zeros_like(content);mask[cells]=True
    mask=F.interpolate(mask.float().reshape(1,1,*grid),size=pixels.shape[-2:],mode='nearest').bool()
    support=F.interpolate(content.float().reshape(1,1,*grid),size=pixels.shape[-2:],mode='nearest').bool()
    if fill=='mean':replacement=(pixels*support).sum((-2,-1),keepdim=True)/support.sum()
    elif fill=='blur':replacement=F.avg_pool2d(F.pad(pixels,(16,16,16,16),mode='reflect'),33,stride=1)
    else:raise ValueError('Unknown fill control')
    return torch.where(mask,replacement,pixels)
