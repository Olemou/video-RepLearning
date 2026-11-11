"""Spatial attribution is connected to the trained scalar through frozen V-JEPA."""
import torch
import pytest

from vision_model.interpretability.spatial_final_block import attribute_spatial_final_block
from vision_model.interpretability.temporal_attribution import AttributionTarget
from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.models.encoder import VJEPA2Encoder
from vision_model.training.trajectory import TrajectoryModel


def _setup():
    torch.manual_seed(19)
    encoder = VJEPA2Encoder(VJEPA2Architecture(frames_per_clip=6, crop_size=16,
        patch_size=8, tubelet_size=2, hidden_size=96, num_hidden_layers=2,
        num_attention_heads=3, mlp_ratio=2.)).eval().requires_grad_(False)
    model = TrajectoryModel(96, hidden=16, embedding_dim=8).eval()
    with torch.no_grad():
        torch.nn.init.normal_(model.up.weight, std=.04)
    video = torch.randn(1, 3, 6, 16, 16)
    hidden = []
    handle = encoder.blocks[-1].register_forward_pre_hook(
        lambda _module, inputs: hidden.append(inputs[0].detach().cpu().clone()))
    features = []
    with torch.inference_mode():
        for prefix in range(1, 4):
            tokens = encoder(video[:, :, :2 * prefix])
            features.append(tokens[:, -4:].mean(1).float()[0])
    handle.remove()
    return encoder, model, hidden, torch.stack(features), torch.tensor([.1, .5, .9])


def test_spatial_gradient_maps_are_target_conditioned_and_causal():
    encoder, model, hidden, features, times = _setup()
    target = AttributionTarget("semantic_similarity", reference=torch.randn(8))
    result = attribute_spatial_final_block(encoder, model, hidden, features, times, target)
    assert result.positive_maps.shape == (3, 2, 2)
    assert (result.positive_maps >= 0).all()
    assert result.positive_maps.sum() > 0
    assert result.signed_maps.shape == result.raw_attention_maps.shape
    assert result.replay_feature_max_error < 1e-5
    assert result.replay_score_error < 1e-5
    assert result.feature_gradient_norms.sum() > 0
    opposite = attribute_spatial_final_block(encoder, model, hidden, features, times,
        AttributionTarget("semantic_similarity", reference=-target.reference))
    torch.testing.assert_close(result.raw_attention_maps, opposite.raw_attention_maps)
    assert not torch.allclose(result.positive_maps, opposite.positive_maps)
    early = attribute_spatial_final_block(encoder, model, hidden, features, times,
                                          AttributionTarget("state", timestep=1, component=0))
    assert early.positive_maps[2].count_nonzero() == 0
    assert early.feature_gradient_norms[2] == 0


def test_spatial_replay_rejects_feature_mismatch():
    encoder, model, hidden, features, times = _setup()
    corrupted = features.clone()
    corrupted[0, 0] += 1
    with pytest.raises(RuntimeError, match="replay differs"):
        attribute_spatial_final_block(encoder, model, hidden, corrupted, times,
                                      AttributionTarget("state", timestep=1, component=0))
