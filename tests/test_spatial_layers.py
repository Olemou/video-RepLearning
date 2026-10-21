import pytest
import torch

from vision_model.interpretability.spatial_layers import capture_spatial_attention
from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.models.encoder import VJEPA2Encoder


def test_spatial_capture_matches_fused_forward_and_input_gradient():
    torch.manual_seed(49)
    encoder = VJEPA2Encoder(VJEPA2Architecture(frames_per_clip=4, crop_size=16,
        patch_size=8, hidden_size=96, num_hidden_layers=3, num_attention_heads=3,
        mlp_ratio=2.)).eval().requires_grad_(False)
    video = torch.randn(1, 3, 4, 16, 16, requires_grad=True)
    reference = encoder(video)
    upstream = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, video, upstream)[0]
    with capture_spatial_attention(encoder, (0, 2)) as maps:
        replay = encoder(video)
        actual, da0, da2 = torch.autograd.grad(replay, (video, maps[0], maps[2]), upstream)
    torch.testing.assert_close(reference, replay, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(expected, actual, atol=3e-6, rtol=2e-5)
    assert da0.abs().sum() > 0 and da2.abs().sum() > 0
    for value in maps.values():
        assert value.shape == (1, 3, 8, 8)
        torch.testing.assert_close(value.sum(-1), torch.ones(1, 3, 8))
    assert all("forward" not in block.attn.__dict__ for block in encoder.blocks)
    with pytest.raises(RuntimeError, match="sentinel"):
        with capture_spatial_attention(encoder, (1,)):
            raise RuntimeError("sentinel")
    assert "forward" not in encoder.blocks[1].attn.__dict__


def test_spatial_mean_has_uniform_patch_gradient_and_erases_grid_permutations():
    from vision_model.training.trajectory import TrajectoryModel
    torch.manual_seed(50)
    model = TrajectoryModel(16, hidden=16, embedding_dim=8).eval()
    patches = torch.randn(1, 4, 9, 16, requires_grad=True)
    valid = torch.ones(1, 4, dtype=torch.bool)
    time = torch.arange(4).float()[None]
    embedding = model(patches.mean(2), valid, time, detach_teacher=False)["embedding"]
    upstream = torch.randn_like(embedding)
    gradient = torch.autograd.grad(embedding, patches, upstream)[0]
    torch.testing.assert_close(gradient, gradient[:, :, :1].expand_as(gradient))
    permuted = patches[:, :, torch.randperm(9)]
    changed = model(permuted.mean(2), valid, time, detach_teacher=False)["embedding"]
    torch.testing.assert_close(embedding, changed, atol=1e-6, rtol=1e-5)
