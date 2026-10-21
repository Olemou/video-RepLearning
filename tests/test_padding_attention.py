"""Padding must not alter valid representations or receive gradients."""
import torch
import pytest

from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.models.encoder import SupervisedVideoEncoder, VJEPA2Encoder


def tiny_model():
    torch.manual_seed(19)
    config = VJEPA2Architecture(frames_per_clip=8, crop_size=16, patch_size=8,
                               hidden_size=96, num_hidden_layers=2, num_attention_heads=3,
                               mlp_ratio=2.)
    return SupervisedVideoEncoder(VJEPA2Encoder(config), latent_dim=8).eval()


def test_invalid_frame_content_cannot_change_video_representation():
    model = tiny_model()
    pixels = torch.randn(1, 8, 3, 16, 16)
    altered = pixels.clone()
    altered[:, 4:] = 100 * torch.randn_like(altered[:, 4:])
    valid = torch.tensor([[True] * 4 + [False] * 4])
    a, b = model(pixels, valid), model(altered, valid)
    torch.testing.assert_close(a.video_mean, b.video_mean, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a.video_variance, b.video_variance, atol=1e-6, rtol=1e-5)


def test_padding_receives_no_gradient():
    model = tiny_model()
    pixels = torch.randn(1, 8, 3, 16, 16, requires_grad=True)
    valid = torch.tensor([[True] * 4 + [False] * 4])
    model(pixels, valid).video_mean.square().sum().backward()
    assert pixels.grad[:, :4].abs().sum() > 0
    assert torch.count_nonzero(pixels.grad[:, 4:]) == 0


def test_masked_encoder_matches_truncated_clip_and_preserves_all_valid_path():
    model = tiny_model()
    pixels = torch.randn(1, 8, 3, 16, 16)
    mask = torch.tensor([[True] * 4 + [False] * 4])
    actual = model(pixels, mask)
    expected = model(pixels[:, :4])
    torch.testing.assert_close(actual.video_mean, expected.video_mean, atol=1e-6, rtol=1e-5)
    a = model(pixels)
    b = model(pixels, torch.ones(1, 8, dtype=torch.bool))
    torch.testing.assert_close(a.video_mean, b.video_mean, atol=0, rtol=0)


def test_all_padding_is_rejected_before_attention():
    with pytest.raises(ValueError, match="valid"):
        tiny_model()(torch.randn(1, 8, 3, 16, 16), torch.zeros(1, 8, dtype=torch.bool))
