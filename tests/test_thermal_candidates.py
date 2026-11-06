import torch
from vision_model.training.thermal_candidates import cold_contrast_grid,cold_pixel_targets,cold_contrast_from_encoder_pixels


def test_cold_contrast_finds_synthetic_peripheral_blob_without_location_prior():
    frames=torch.full((4,3,64,64),120,dtype=torch.uint8)
    frames[:,:,32:48,16:32]=20
    score=cold_contrast_grid(frames,size=64)
    assert score.shape==(2,16)
    assert int(score[0].argmax())==9 and score[0,9]>.1
    flipped=cold_contrast_grid(frames.flip(-1),size=64)
    torch.testing.assert_close(flipped,score.reshape(2,4,4).flip(-1).flatten(1))
    constant=cold_contrast_grid(torch.full_like(frames,120),size=64)
    assert float(constant.max())<1e-4
    frames[1]=120
    transient=cold_contrast_grid(frames,size=64)
    assert float(transient[0].max())<1e-4 # Must appear in both actual frames.


def test_thermal_targets_keep_ambiguous_cells_and_padding_ignored():
    scores=torch.tensor([[[.9,.1,.4,.1],[.9,.1,.4,.1]],[[.9,.1,.4,.1],[.9,.1,.4,.1]]])
    feature=torch.zeros(2,2,2,4,dtype=torch.long)
    valid=torch.tensor([[True,False],[True,False]])
    content=torch.tensor([[True,True,True,False]]*2)
    t=cold_pixel_targets(scores,feature,torch.tensor([0,1]),valid,content,low=.2,high=.8)
    assert t[1,0,0].tolist()==[1,0,-1,-1]
    assert t[0,0,0].tolist()==[0,0,0,-1]
    assert (t[:,:,1]==-1).all()


def test_encoder_aligned_descriptor_inverts_padding_and_reacts_to_changed_pixels():
    frames=torch.full((4,3,96,128),120,dtype=torch.uint8);frames[:,:,32:48,16:32]=20
    mean=torch.tensor([.485,.456,.406])[None,:,None,None]
    std=torch.tensor([.229,.224,.225])[None,:,None,None]
    canvas=(mean*255).round().expand(4,3,128,128).clone()
    canvas[:,:,16:112]=frames
    normalized=(canvas/255-mean)/std
    expected=cold_contrast_grid(frames,size=128)
    actual=cold_contrast_from_encoder_pixels(normalized,96,128)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    canvas[:,:,48:64,16:32]=120
    changed=cold_contrast_from_encoder_pixels((canvas/255-mean)/std,96,128)
    assert changed.max()<actual.max()
