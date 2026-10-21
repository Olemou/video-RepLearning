import torch
import torch.nn.functional as F
from vision_model.models.encoder import RoPEAttention, _rotate
from vision_model.models.attention_maps import received_attention


def test_chunked_attention_matches_dense_and_excludes_padding():
    torch.manual_seed(31)
    module = RoPEAttention(96, 3, 2).eval()
    value = torch.randn(1, 12, 96)
    mask = torch.tensor([[True] * 8 + [False] * 4])
    actual = received_attention(module, value, frames=3, height=2, width=2,
                                token_valid_mask=mask, chunk_size=3).flatten()
    q, k, _ = module.qkv(value).reshape(1, 12, 3, 3, 32).permute(2, 0, 3, 1, 4).unbind(0)
    positions = module._positions(torch.arange(12), 2, 2)
    q = torch.cat([_rotate(q[..., i*10:(i+1)*10], p) for i, p in enumerate(positions)] + [q[..., 30:]], -1)
    k = torch.cat([_rotate(k[..., i*10:(i+1)*10], p) for i, p in enumerate(positions)] + [k[..., 30:]], -1)
    probability = ((q @ k.transpose(-1, -2)) / 32**.5).masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
    expected = probability[0, :, :8].mean((0, 1))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(), torch.tensor(1.))
    assert actual[8:].count_nonzero() == 0
    altered = value.clone(); altered[:, 8:] *= 100
    torch.testing.assert_close(actual, received_attention(module, altered, frames=3, height=2, width=2,
                                                        token_valid_mask=mask, chunk_size=5).flatten())


def test_uniform_attention_does_not_invent_a_hotspot():
    module = RoPEAttention(96, 3, 2)
    with torch.no_grad():
        module.qkv.weight.zero_(); module.qkv.bias.zero_()
    actual = received_attention(module, torch.randn(1, 8, 96), frames=2, height=2, width=2)
    torch.testing.assert_close(actual, torch.full((2, 2, 2), 1/8))
