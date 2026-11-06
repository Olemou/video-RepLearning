import torch
from unittest.mock import patch
from vision_model.interpretability.context_controls import protected_tiles,change_distant_context


def test_protected_tiles_cover_content_and_have_disjoint_interior_probes():
    tiles=protected_tiles()
    assert len(tiles)==12
    coverage=torch.stack([t['protected'] for t in tiles]).sum(0)
    assert coverage[2:14].eq(1).all()
    assert not coverage[:2].any() and not coverage[14:].any()
    assert all(t['core'].sum()==4 and (t['core']<=t['protected']).all() for t in tiles)


def test_context_change_preserves_raw_and_transformed_probe_pixels():
    from control_experiment81 import transform_bank
    from vision_model.data.transforms import OfficialVJEPA2Transform
    # Exercise real full-frame geometry without downloading processor metadata.
    class Normalizer:
        crop_size={'height':256,'width':256}
        image_mean=[.485,.456,.406]
        image_std=[.229,.224,.225]
        def __call__(self,frames,**kwargs):
            mean=torch.tensor(self.image_mean)[None,:,None,None]
            std=torch.tensor(self.image_std)[None,:,None,None]
            return {'pixel_values_videos':((frames.float()/255-mean)/std)[None]}
    with patch('transformers.AutoVideoProcessor.from_pretrained',return_value=Normalizer()):
        transform=OfficialVJEPA2Transform(preserve_field_of_view=True)
    torch.set_num_threads(1)
    torch.manual_seed(239)
    rgb=torch.randint(0,256,(2,3,120,160),dtype=torch.uint8)
    original=rgb.clone()
    before=transform_bank(rgb,[],transform,include_augmentations=False)
    for tile in protected_tiles():
        for fill in ('gain09','blur33'):
            after,mask=change_distant_context(rgb,tile['protected'],fill)
            assert torch.equal(after[...,mask],rgb[...,mask])
            px=transform_bank(after,[],transform,include_augmentations=False)
            core=tile['core'].repeat_interleave(16,0).repeat_interleave(16,1)
            assert torch.equal(px[0,...,core],before[0,...,core])
    assert torch.equal(rgb,original)


def test_content_only_prefix_readout_and_gradient_match_direct_encoder():
    from audit_content_only_attribution import ContentPrefixEncoder
    from tests.test_online_trajectory import example
    torch.set_num_threads(1)
    original,_,pixels,*_=example(2)
    original.requires_grad_(False)
    support=torch.tensor([[True,False],[True,False]])
    wrapper=ContentPrefixEncoder(original.encoder,support)
    clip=pixels[:,0].permute(0,2,1,3,4).clone().requires_grad_()
    direct=original.encoder(clip)[:,-4:][:,support.flatten()].mean(1)
    actual=wrapper(clip)
    torch.testing.assert_close(actual,direct,atol=0,rtol=0)
    expected,=torch.autograd.grad(direct.square().sum(),clip,retain_graph=True)
    derivative,=torch.autograd.grad(actual.square().sum(),clip)
    torch.testing.assert_close(derivative,expected,atol=0,rtol=0)
    assert set(wrapper.encoder.state_dict())==set(original.encoder.state_dict())
