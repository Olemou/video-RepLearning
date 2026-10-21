"""Base V-JEPA 2 ViT-L encoder; Hugging Face is only a weight source."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import torch
import torch.nn.functional as F
from torch import nn
from .checkpoint import VJEPA2Architecture, validate_architecture, validate_loading_info
from .gaussian import temporal_gaussian

def _rotate(value: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    width = value.shape[-1]
    omega = torch.arange(width // 2, dtype=value.dtype, device=value.device) / (width / 2.0)
    frequency = torch.einsum("...,f->...f", positions, 1.0 / (10000**omega))
    # Intentional upstream checkpoint-compatible frequency-pair layout.
    sine = frequency.sin().squeeze(-1).repeat(1, 1, 1, 2)
    cosine = frequency.cos().squeeze(-1).repeat(1, 1, 1, 2)
    first, second = value.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return value * cosine + rotated * sine

class PatchEmbed3D(nn.Module):
    def __init__(self, embed_dim: int, patch_size: int, tubelet_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(3, embed_dim, kernel_size=(tubelet_size, patch_size, patch_size), stride=(tubelet_size, patch_size, patch_size))
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.proj(video).flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, dim: int, ratio: float) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(value)))

class RoPEAttention(nn.Module):
    def __init__(self, dim: int, heads: int, grid_size: int) -> None:
        super().__init__()
        self.num_heads, self.head_dim, self.grid_size = heads, dim // heads, grid_size
        self.d_dim = self.h_dim = self.w_dim = 2 * ((self.head_dim // 3) // 2)
        self.qkv, self.proj = nn.Linear(dim, dim * 3, bias=True), nn.Linear(dim, dim)
    @staticmethod
    def _positions(ids: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, ...]:
        per_frame = height * width
        depth = ids // per_frame
        remainder = ids - per_frame * depth
        row = remainder // width
        return depth, row, remainder - width * row
    def forward(self, value: torch.Tensor, *, frames: int, height: int, width: int, token_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, tokens, channels = value.shape
        if tokens != frames * height * width:
            raise ValueError(f"Token/grid mismatch: {tokens} != {frames}*{height}*{width}")
        query, key, val = self.qkv(value).unflatten(-1, (3, self.num_heads, -1)).permute(2, 0, 3, 1, 4).unbind(0)
        positions = self._positions(torch.arange(tokens, device=value.device), height, width)
        offset, q_parts, k_parts = 0, [], []
        for size, position in zip((self.d_dim, self.h_dim, self.w_dim), positions):
            q_parts.append(_rotate(query[..., offset:offset + size], position))
            k_parts.append(_rotate(key[..., offset:offset + size], position))
            offset += size
        if offset < self.head_dim:
            q_parts.append(query[..., offset:])
            k_parts.append(key[..., offset:])
        attention_mask = None if token_valid_mask is None else token_valid_mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(torch.cat(q_parts, -1), torch.cat(k_parts, -1), val,
                                                 attn_mask=attention_mask, dropout_p=0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, channels))

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ratio: float, grid_size: int) -> None:
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim, eps=1e-6), RoPEAttention(dim, heads, grid_size)
        self.norm2, self.mlp = nn.LayerNorm(dim, eps=1e-6), MLP(dim, ratio)
    def forward(self, value: torch.Tensor, *, frames: int, height: int, width: int, token_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        value = value + self.attn(self.norm1(value), frames=frames, height=height, width=width, token_valid_mask=token_valid_mask)
        return value + self.mlp(self.norm2(value))

class VJEPA2Encoder(nn.Module):
    """Encoder only; input shape is [B,C,T,H,W]."""
    def __init__(self, architecture: VJEPA2Architecture | None = None) -> None:
        super().__init__()
        self.architecture = architecture or VJEPA2Architecture()
        cfg = self.architecture
        self.patch_embed = PatchEmbed3D(cfg.hidden_size, cfg.patch_size, cfg.tubelet_size)
        self.blocks = nn.ModuleList(Block(cfg.hidden_size, cfg.num_attention_heads, cfg.mlp_ratio, cfg.crop_size // cfg.patch_size) for _ in range(cfg.num_hidden_layers))
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
    def forward(self, video: torch.Tensor, temporal_valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        _, _, time, height, width = video.shape
        cfg = self.architecture
        if time % cfg.tubelet_size or height % cfg.patch_size or width % cfg.patch_size:
            raise ValueError("Input dimensions must be divisible by tubelet/patch sizes")
        value = self.patch_embed(video)
        grid = (time // cfg.tubelet_size, height // cfg.patch_size, width // cfg.patch_size)
        token_valid_mask = None
        if temporal_valid_mask is not None:
            temporal_valid_mask = temporal_valid_mask.to(device=video.device, dtype=torch.bool)
            if temporal_valid_mask.shape != (video.shape[0], grid[0]):
                raise ValueError("Encoder validity mask must have shape [B, temporal_tubelets]")
            if not temporal_valid_mask.any(dim=1).all():
                raise ValueError("Every video must contain a valid temporal tubelet")
            # Preserve the exact checkpoint path for fully observed clips.
            if not temporal_valid_mask.all():
                token_valid_mask = temporal_valid_mask.repeat_interleave(grid[1] * grid[2], dim=1)
        for block in self.blocks:
            value = block(value, frames=grid[0], height=grid[1], width=grid[2], token_valid_mask=token_valid_mask)
        return self.norm(value)

@dataclass(frozen=True)
class EncoderOutput:
    tokens: torch.Tensor
    temporal_features: torch.Tensor
    temporal_mean: torch.Tensor
    temporal_variance: torch.Tensor
    video_mean: torch.Tensor
    video_variance: torch.Tensor
    temporal_valid_mask: torch.Tensor
    embedding: torch.Tensor

class SupervisedVideoEncoder(nn.Module):
    """Base encoder plus temporal diagonal-Gaussian representation heads."""
    def __init__(
        self,
        encoder: VJEPA2Encoder | None = None,
        *,
        latent_dim: int = 256,
        variance_floor: float = 1e-4,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder or VJEPA2Encoder()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if variance_floor <= 0:
            raise ValueError("variance_floor must be positive")
        hidden = self.encoder.architecture.hidden_size
        self.latent_dim = latent_dim
        self.variance_floor = variance_floor
        self.mean_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent_dim)
        )
        self.variance_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, latent_dim)
        )
        self.freeze_encoder(freeze_encoder)

    def freeze_encoder(self, frozen: bool = True) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not frozen)

    def _tubelet_mask(
        self, mask: torch.Tensor | None, *, batch: int, frames: int, tubelets: int, device: torch.device
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch, tubelets, dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if tuple(mask.shape) == (batch, tubelets):
            return mask
        if tuple(mask.shape) != (batch, frames):
            raise ValueError(f"temporal_valid_mask must be [B,T] or [B,T/tubelet], got {tuple(mask.shape)}")
        size = self.encoder.architecture.tubelet_size
        return mask.unflatten(1, (tubelets, size)).all(dim=-1)

    def forward(
        self,
        pixel_values_videos: torch.Tensor,
        temporal_valid_mask: torch.Tensor | None = None,
    ) -> EncoderOutput:
        if pixel_values_videos.ndim != 5 or pixel_values_videos.shape[2] != 3:
            raise ValueError("pixel_values_videos must be [B,T,3,H,W]")
        batch, frames, _, height, width = pixel_values_videos.shape
        cfg = self.encoder.architecture
        tubelets = frames // cfg.tubelet_size
        spatial_tokens = (height // cfg.patch_size) * (width // cfg.patch_size)
        valid = self._tubelet_mask(
            temporal_valid_mask, batch=batch, frames=frames, tubelets=tubelets, device=pixel_values_videos.device
        )
        tokens = self.encoder(pixel_values_videos.permute(0, 2, 1, 3, 4).contiguous(), temporal_valid_mask=valid)
        temporal_features = tokens.unflatten(1, (tubelets, spatial_tokens)).mean(dim=2)
        temporal_mean, temporal_variance, video_mean, video_variance = temporal_gaussian(
            self.mean_head, self.variance_head, temporal_features, valid, self.variance_floor)
        return EncoderOutput(
            tokens=tokens,
            temporal_features=temporal_features,
            temporal_mean=temporal_mean,
            temporal_variance=temporal_variance,
            video_mean=video_mean,
            video_variance=video_variance,
            temporal_valid_mask=valid,
            embedding=F.normalize(video_mean, dim=-1),
        )

def convert_huggingface_encoder_state(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map all HF encoder tensors to the audited base namespace."""
    converted = {
        "patch_embed.proj.weight": source["embeddings.patch_embeddings.proj.weight"],
        "patch_embed.proj.bias": source["embeddings.patch_embeddings.proj.bias"],
        "norm.weight": source["layernorm.weight"], "norm.bias": source["layernorm.bias"],
    }
    consumed = {"embeddings.patch_embeddings.proj.weight", "embeddings.patch_embeddings.proj.bias", "layernorm.weight", "layernorm.bias"}
    for index in range(24):
        hf, base = f"layer.{index}.", f"blocks.{index}."
        names = ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "attention.proj.weight", "attention.proj.bias", "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")
        for name in names:
            converted[base + name.replace("attention.", "attn.")] = source[hf + name]
            consumed.add(hf + name)
        for suffix in ("weight", "bias"):
            qkv_names = [hf + f"attention.{part}.{suffix}" for part in ("query", "key", "value")]
            converted[base + f"attn.qkv.{suffix}"] = torch.cat([source[name] for name in qkv_names], 0)
            consumed.update(qkv_names)
    if set(source) != consumed:
        raise RuntimeError(f"Unconsumed Hugging Face encoder keys: {sorted(set(source) - consumed)}")
    return converted

def load_pretrained_encoder(model_id: str = "facebook/vjepa2-vitl-fpc64-256", *, local_files_only: bool = False) -> SupervisedVideoEncoder:
    """Construct base encoder first, then strictly import pretrained weights."""
    model = SupervisedVideoEncoder()
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise ImportError("transformers>=4.53 is required to read the checkpoint") from exc
    loaded, info = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", output_loading_info=True, local_files_only=local_files_only)
    validate_loading_info(info)
    validate_architecture(loaded.config, model.encoder.architecture)
    model.encoder.load_state_dict(convert_huggingface_encoder_state(loaded.encoder.state_dict()), strict=True)
    return model
