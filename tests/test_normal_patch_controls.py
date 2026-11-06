import pytest
import torch
from vision_model.interpretability.normal_patch_controls import ranked_cells,remove_cells


@pytest.mark.parametrize('fill',['mean','blur'])
def test_intervention_changes_only_selected_real_cells(fill):
    x=torch.randn(4,3,64,64,generator=torch.Generator().manual_seed(4))
    support=torch.tensor([0,1,1,1],dtype=torch.bool);cells=ranked_cells(torch.tensor([100.,1.,2.,3.]),support,count=1)
    assert cells.tolist()==[3]
    y=remove_cells(x,cells,support,fill=fill,grid=(2,2))
    assert torch.equal(x[:,:,:32],y[:,:,:32]) and torch.equal(x[:,:,:,:32],y[:,:,:,:32])
    assert not torch.equal(x[:,:,32:,32:],y[:,:,32:,32:])
    with pytest.raises(ValueError):remove_cells(x,torch.tensor([0]),support,grid=(2,2))
