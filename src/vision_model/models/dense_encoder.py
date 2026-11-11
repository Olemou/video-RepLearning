"""Strict, encoder-only V-JEPA 2.1 ViT-L integration.

Architecture/rotation follow facebookresearch/vjepa2 app/vjepa_2_1/models.
Unlike V-JEPA 2, 2.1 uses adjacent RoPE frequency pairs, spatial interpolation
to a 16-cell reference grid, separate image/video tokenizers and four norms.
The original V-JEPA 2 loader and rotation are deliberately unchanged.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib

import torch
from torch import nn
import torch.nn.functional as F

from .encoder import PatchEmbed3D, MLP, RoPEAttention

MODEL_ID = 'vjepa2_1_vitl_dist_vitG_384'
OFFICIAL_URL = 'https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt'


@dataclass(frozen=True)
class DenseArchitecture:
    width: int = 1024
    depth: int = 24
    heads: int = 16
    patch: int = 16
    tubelet: int = 2
    ratio: float = 4.
    # The checkpoint contains a norm at each of four hierarchical endpoints.
    norm_count: int = 4


def rotate_dense(value, positions):
    width = value.shape[-1]
    omega = torch.arange(width // 2, dtype=value.dtype, device=value.device) / (width / 2.)
    frequency = torch.einsum('...,f->...f', positions, 1. / (10000 ** omega))
    sine, cosine = frequency.sin().repeat_interleave(2, -1), frequency.cos().repeat_interleave(2, -1)
    first, second = value.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-second, first), -1).flatten(-2)
    return value * cosine + rotated * sine


class DenseAttention(RoPEAttention):
    def forward(self, value, *, frames, height, width):
        batch, tokens, channels = value.shape
        if tokens != frames * height * width or min(height, width) < 2:
            raise ValueError('Need complete token grid with at least two cells per spatial axis')
        q, k, v = self.qkv(value).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
        time, row, col = self._positions(torch.arange(tokens, device=value.device), height, width)
        positions = (time.float(), row.float() * 15 / (height - 1), col.float() * 15 / (width - 1))
        offset, qs, ks = 0, [], []
        for size, pos in zip((self.d_dim, self.h_dim, self.w_dim), positions):
            qs.append(rotate_dense(q[..., offset:offset+size], pos))
            ks.append(rotate_dense(k[..., offset:offset+size], pos))
            offset += size
        qs.append(q[..., offset:]); ks.append(k[..., offset:])
        x = F.scaled_dot_product_attention(torch.cat(qs, -1), torch.cat(ks, -1), v, dropout_p=0.)
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, channels))


class DenseBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width, eps=1e-6)
        self.attn = DenseAttention(cfg.width, cfg.heads, 24)
        self.norm2 = nn.LayerNorm(cfg.width, eps=1e-6)
        self.mlp = MLP(cfg.width, cfg.ratio)

    def forward(self, x, **grid):
        x = x + self.attn(self.norm1(x), **grid)
        return x + self.mlp(self.norm2(x))


class DenseVideoEncoder(nn.Module):
    def __init__(self, architecture=None):
        super().__init__()
        self.architecture = cfg = architecture or DenseArchitecture()
        if cfg.width % cfg.heads or cfg.width // cfg.heads < 6:
            raise ValueError('Unsupported attention width')
        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, cfg.width))
        self.patch_embed = PatchEmbed3D(cfg.width, cfg.patch, cfg.tubelet)
        self.patch_embed_img = PatchEmbed3D(cfg.width, cfg.patch, 1)
        self.blocks = nn.ModuleList(DenseBlock(cfg) for _ in range(cfg.depth))
        self.norms_block = nn.ModuleList(nn.LayerNorm(cfg.width, eps=1e-6) for _ in range(cfg.norm_count))

    def forward(self, video):
        cfg = self.architecture
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError('Expected [B,3,T,H,W]')
        _, _, time, height, width = video.shape
        if height % cfg.patch or width % cfg.patch or (time != 1 and time % cfg.tubelet):
            raise ValueError('Dimensions must match image/video tokenizer')
        if time == 1:
            x = self.patch_embed_img(video) + self.img_mod_embed
            frames = 1
        else:
            x = self.patch_embed(video) + self.video_mod_embed
            frames = time // cfg.tubelet
        for block in self.blocks:
            x = block(x, frames=frames, height=height//cfg.patch, width=width//cfg.patch)
        return self.norms_block[-1](x)


def convert_dense_state(source):
    converted = {}
    prefix = 'module.backbone.'
    for key, value in source.items():
        if not key.startswith(prefix) or not torch.is_tensor(value):
            raise RuntimeError(f'Unexpected EMA encoder key: {key}')
        target = key[len(prefix):]
        if target in converted:
            raise RuntimeError(f'Duplicate encoder key: {target}')
        converted[target] = value
    return converted


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def load_dense_encoder(path, *, expected_sha256=None):
    """Consume every EMA encoder tensor strictly; never load optimizer/predictor.

    The official training container is trusted input, mapped instead of copied.
    The encoder is frozen and in eval mode on return.
    """
    checksum = file_sha256(path)
    if expected_sha256 is not None and checksum != expected_sha256:
        raise RuntimeError('Dense checkpoint checksum mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if 'ema_encoder' not in saved:
        raise RuntimeError('Distilled ViT-L checkpoint requires ema_encoder')
    with torch.device('meta'):
        encoder = DenseVideoEncoder()
    encoder.load_state_dict(convert_dense_state(saved['ema_encoder']), strict=True, assign=True)
    encoder.eval().requires_grad_(False)
    encoder.provenance = dict(model_id=MODEL_ID, checkpoint_sha256=checksum,
                              checkpoint_key='ema_encoder', encoder_keys=len(saved['ema_encoder']),
                              spatial_rope_reference=16, rotation='adjacent_frequency_pairs')
    return encoder
