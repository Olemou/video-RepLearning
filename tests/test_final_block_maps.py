import torch

from vision_model.models.encoder import VJEPA2Encoder
from vision_model.models.checkpoint import VJEPA2Architecture
from vision_model.models.final_block_maps import replay_last_tubelet, temporal_layer_attention
from vision_model.training.trajectory import TrajectoryModel


def encoder():
    return VJEPA2Encoder(VJEPA2Architecture(frames_per_clip=6, crop_size=16,
        patch_size=8, tubelet_size=2, hidden_size=96, num_hidden_layers=2,
        num_attention_heads=3, mlp_ratio=2.)).eval()


def test_final_query_replay_matches_full_fused_block_and_gradients():
    torch.manual_seed(45)
    model = encoder()
    x = torch.randn(1, 12, 96, requires_grad=True)
    full = model.norm(model.blocks[-1](x, frames=3, height=2, width=2))[:, -4:]
    replay, attention = replay_last_tubelet(model, x, frames=3, height=2, width=2)
    torch.testing.assert_close(replay, full, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(replay)
    full_grad = torch.autograd.grad(full, x, upstream)[0]
    replay_grad = torch.autograd.grad(replay, x, upstream, retain_graph=True)[0]
    torch.testing.assert_close(replay_grad, full_grad, rtol=1e-4, atol=2e-6)
    assert attention.shape == (1, 3, 4, 12)
    torch.testing.assert_close(attention.sum(-1), torch.ones(1, 3, 4))
    da = torch.autograd.grad(replay, attention, upstream)[0]
    assert torch.isfinite(da).all() and da.abs().sum() > 0


def test_attribution_bridge_preserves_prediction_and_enables_only_requested_input_gradients():
    torch.manual_seed(8)
    model = TrajectoryModel(16, hidden=16, embedding_dim=8).eval()
    x = torch.randn(2, 3, 16, requires_grad=True)
    valid = torch.ones(2, 3, dtype=torch.bool)
    clock = torch.arange(3).float()[None].expand(2, -1)
    detached = model(x, valid, clock)["embedding"]
    bridged = model(x, valid, clock, detach_teacher=False)["embedding"]
    torch.testing.assert_close(detached, bridged, rtol=0, atol=0)
    assert torch.autograd.grad(detached[:, 0].sum(), x, allow_unused=True)[0] is None
    grad = torch.autograd.grad(bridged[:, 0].sum(), x)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_temporal_attention_matches_actual_multihead_operator():
    torch.manual_seed(7)
    model = TrajectoryModel(16, hidden=16).eval()
    layer = model.temporal.layers[-1]
    hidden = torch.randn(2, 5, 16)
    valid = torch.tensor([[True]*5, [True]*3 + [False]*2])
    value = layer.norm1(hidden)
    causal = torch.ones(5, 5, dtype=torch.bool).triu(1)
    _, expected = layer.self_attn(value, value, value, attn_mask=causal,
                                  key_padding_mask=~valid, need_weights=True,
                                  average_attn_weights=False)
    expected = expected.masked_fill(~valid[:, None, :, None], 0)
    actual = temporal_layer_attention(layer, hidden, valid)
    torch.testing.assert_close(actual, expected)
    assert actual[..., causal].count_nonzero() == 0


def test_custom_final_block_matches_official_transformers_block():
    from transformers import VJEPA2Config
    from transformers.models.vjepa2.modeling_vjepa2 import VJEPA2Layer
    torch.manual_seed(18)
    custom = encoder().blocks[-1]
    config = VJEPA2Config(frames_per_clip=6, crop_size=16, patch_size=8,
                         hidden_act="gelu", layer_norm_eps=1e-6, qkv_bias=True)
    config._attn_implementation = "sdpa"
    official = VJEPA2Layer(config, hidden_size=96, num_attention_heads=3, mlp_ratio=2.).eval()
    mapped = {}
    for name, tensor in custom.state_dict().items():
        if name.startswith("attn.qkv."):
            suffix = name.split(".")[-1]
            for part, value in zip(("query", "key", "value"), tensor.chunk(3, dim=0)):
                mapped[f"attention.{part}.{suffix}"] = value
        else:
            mapped[name.replace("attn.", "attention.")] = tensor
    official.load_state_dict(mapped, strict=True)
    x = torch.randn(1, 12, 96)
    with torch.no_grad():
        actual = custom(x, frames=3, height=2, width=2)
        expected = official(x)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
