import pytest
import torch
from vision_model.interpretability.map_geometry import inverse_letterbox_map,letterbox_bounds,blend_rgb


@pytest.mark.parametrize('row,col',[(2,0),(2,15),(8,8),(13,0),(13,15)])
def test_native_content_impulses_reach_expected_source_positions(row,col):
    grid=torch.zeros(16,16);grid[row,col]=1
    restored=inverse_letterbox_map(grid,480,640)
    # Bilinear resize extends a boundary cell as a flat maximum up to the
    # image edge. Check the maximum set, not its arbitrary first element.
    peaks=(restored>=restored.max()-1e-6).nonzero()
    expected=torch.tensor([((row+.5)*16-32)/.4,(col+.5)*16/.4])
    assert (peaks-expected).abs().max(-1).values.min()<3
    torch.testing.assert_close(grid.flip(-1).flip(-1),grid,rtol=0,atol=0)
    torch.testing.assert_close(inverse_letterbox_map(grid.flip(-1),480,640),restored.flip(-1),rtol=1e-5,atol=2e-6)


def test_padding_excluded_asymmetric_dimensions_and_rgb_blend():
    assert letterbox_bounds(480,640)==(32,0,192,256)
    # Letterbox-only impulse has no value in original pixels (nearest content
    # edge can include half a patch of interpolation, so use distant padding).
    grid=torch.zeros(16,16);grid[0]=1
    assert inverse_letterbox_map(grid,480,640).sum()==0
    assert letterbox_bounds(319,191)==(0,51,256,153)
    assert inverse_letterbox_map(torch.ones(16,16),319,191).shape==(319,191)
    red=torch.tensor([[[255,0,0]]],dtype=torch.uint8)
    blue=torch.tensor([[[0,0,255]]],dtype=torch.uint8)
    assert blend_rgb(red,blue,.4).tolist()==[[[153,0,102]]]


def test_evidence_overlay_zero_is_transparent_and_changes_only_actual_cells():
    from vision_model.interpretability.map_geometry import spatial_evidence_overlay
    source=torch.full((480,640,3),100,dtype=torch.uint8)
    blank=torch.zeros(8,8)
    original=spatial_evidence_overlay(source,blank,scale=2.,scope='input_intervention')
    torch.testing.assert_close(original,source,rtol=0,atol=0)
    evidence=blank.clone();evidence[1,0]=1.;evidence[1,1]=-2.
    overlay=spatial_evidence_overlay(source,evidence,scale=2.,scope='input_intervention',alpha=.4)
    # 8x8 grid: cell[1,0] is source [0:80,0:80] after letterbox removal.
    assert overlay[20,20].tolist()==[131,80,80]
    assert overlay[20,100].tolist()==[60,60,162]
    torch.testing.assert_close(overlay[80:],source[80:],rtol=0,atol=0)
    torch.testing.assert_close(overlay[:80,160:],source[:80,160:],rtol=0,atol=0)


def test_contextual_class_votes_and_raw_attention_cannot_be_painted_as_localization():
    from vision_model.interpretability.map_geometry import spatial_evidence_overlay
    frame=torch.zeros(24,32,3,dtype=torch.uint8)
    for scope in ('contextual_feature_contribution','raw_attention'):
        with pytest.raises(ValueError,match='not localization'):
            spatial_evidence_overlay(frame,torch.ones(8,8),scale=1.,scope=scope)
    with pytest.raises(ValueError):
        spatial_evidence_overlay(frame,-torch.ones(8,8),scale=1.,scope='conditional_attention_key_gradient',signed=False)


def test_blue_red_colors_have_fixed_endpoints_and_shared_numeric_scale():
    from vision_model.interpretability.map_geometry import nonnegative_evidence_colors
    values=torch.tensor([0.,.25,.5,.75,1.])
    assert nonnegative_evidence_colors(values,scale=1.).tolist()==[
        [0,0,255],[0,255,255],[0,255,0],[255,255,0],[255,0,0]]
    # A weak map stays weak: do not turn its own maximum red.
    torch.testing.assert_close(nonnegative_evidence_colors(values[:2],scale=1.),
                               nonnegative_evidence_colors(values,scale=1.)[:2])
    for bad in (torch.tensor([-1.]),torch.tensor([float('nan')])):
        with pytest.raises(ValueError):nonnegative_evidence_colors(bad,scale=1.)


def test_blue_red_overlay_encodes_zero_blue_and_keeps_native_cell_boundary():
    from vision_model.interpretability.map_geometry import spatial_evidence_overlay
    frame=torch.full((480,640,3),100,dtype=torch.uint8)
    grid=torch.zeros(8,8);grid[1,1]=2.
    overlay=spatial_evidence_overlay(frame,grid,scale=2.,scope='conditional_attention_key_gradient',
        signed=False,style='blue_red',alpha=.4)
    assert overlay[20,79].tolist()==[60,60,162]
    assert overlay[20,80].tolist()==[162,60,60]
    with pytest.raises(ValueError):
        spatial_evidence_overlay(frame,grid,scale=2.,scope='input_intervention',style='blue_red')
