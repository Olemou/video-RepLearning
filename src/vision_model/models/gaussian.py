"""Shared float32 Gaussian prediction and masked temporal aggregation."""
import torch
import torch.nn.functional as F


def temporal_gaussian(mean_head, variance_head, features, valid, variance_floor):
    mean_t = mean_head(features).float()
    raw_variance = variance_head(features).float()
    with torch.autocast(device_type=features.device.type, enabled=False):
        variance_t = variance_floor + F.softplus(raw_variance)
        valid = valid.to(device=features.device, dtype=torch.bool)
        counts = valid.sum(dim=1, keepdim=True)
        if torch.any(counts == 0):
            raise ValueError("Every video must contain at least one valid temporal tubelet")
        weights = valid.unsqueeze(-1).to(mean_t.dtype)
        mean = (mean_t * weights).sum(dim=1) / counts
        centered = mean_t - mean.unsqueeze(1)
        variance = ((variance_t + centered.square()) * weights).sum(dim=1) / counts
    return mean_t, variance_t, mean, variance
