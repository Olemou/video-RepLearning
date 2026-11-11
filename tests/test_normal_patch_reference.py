import pytest
import torch
from vision_model.training.normal_patch_reference import NormalPatchReference,local_residual,automatic_candidates


def test_source_exclusion_and_normal_only_fit():
    x=torch.tensor([[[[[1.,0.],[1.,0.]]]],[[[[0.,1.],[0.,1.]]]]])
    valid=torch.ones(2,1,dtype=torch.bool);content=torch.ones(2,2,dtype=torch.bool)
    ref=NormalPatchReference(per_source=2).fit(x,valid,content,['a','b'],torch.zeros(2,dtype=torch.long))
    assert ref.score(x[0],content[0]).max()<1e-6
    assert torch.allclose(ref.score(x[0],content[0],exclude_source='a'),torch.ones(1,1,2))
    with pytest.raises(ValueError,match='clean'):
        ref.fit(x,valid,content,['a','b'],torch.tensor([0,1]))


def test_residual_removes_global_shift_and_padding_does_not_fit():
    x=torch.randn(2,2,3,4,6,generator=torch.Generator().manual_seed(2));support=torch.tensor([1,1,1,0],dtype=torch.bool)
    assert torch.allclose(local_residual(x,support),local_residual(x+3,support),atol=1e-6)
    content=support.expand(2,-1);valid=torch.tensor([[1,1,0],[1,1,0]],dtype=torch.bool)
    bad=x.clone();bad[:,:,2]=1e6;bad[:,:,:,3]=1e6
    a=NormalPatchReference(mode='whitened_residual',per_source=4).fit(x,valid,content,['a','b'],torch.zeros(2,dtype=torch.long))
    b=NormalPatchReference(mode='whitened_residual',per_source=4).fit(bad,valid,content,['a','b'],torch.zeros(2,dtype=torch.long))
    assert torch.equal(a.bank,b.bank)


def test_pseudo_labels_abstain_on_uniform_anomaly_and_need_mirror_agreement():
    scores=torch.ones(1,2,2,16);valid=torch.tensor([[1,0]],dtype=torch.bool);content=torch.ones(1,16,dtype=torch.bool)
    assert (automatic_candidates(scores,valid,content,low=.2,high=.8)==-1).all()
    scores.zero_();scores[0,0,0,5]=1
    p=automatic_candidates(scores,valid,content,low=.2,high=.8)
    assert p[0,0,0,5]==-1
    scores[0,1,0,6]=1
    p=automatic_candidates(scores,valid,content,low=.2,high=.8)
    assert p[0,0,0,5]==1 and p[0,1,0,6]==1
    assert (p[:,:,1]==-1).all()


def test_channel_selection_is_fit_only_and_score_uses_same_projection():
    x=torch.randn(3,2,2,4,16,generator=torch.Generator().manual_seed(9))
    valid=torch.ones(3,2,dtype=torch.bool);support=torch.ones(3,4,dtype=torch.bool)
    ref=NormalPatchReference(mode='selected_residual',per_source=8).fit(x,valid,support,['a','b','c'],torch.zeros(3,dtype=torch.long))
    assert ref.feature_indices.shape==(4,) and ref.bank.shape[-1]==4
    before=ref.feature_indices.clone();score=ref.score(x[0]+20,support[0],exclude_source='a')
    assert torch.equal(before,ref.feature_indices) and torch.isfinite(score).all()
