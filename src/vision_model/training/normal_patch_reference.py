"""Normal-only patch references. Scores are novelty, never verified spill labels."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def local_residual(features, content):
    """Remove each current tubelet's content mean, without future context."""
    if features.ndim < 2 or content.shape != features.shape[-2:-1] or not content.any():
        raise ValueError('Need nonempty native patch support')
    if not torch.isfinite(features[..., content, :]).all():
        raise ValueError('Nonfinite content features')
    return features - features[..., content, :].mean(-2, keepdim=True)


class NormalPatchReference:
    """Deterministic normal bank; fit inputs must contain training normals only.

    Source IDs enable leave-source-out training calibration. Validation samples
    cannot enter fitting through this API. There is no spatial pooling selector.
    """
    def __init__(self, *, mode='raw', per_source=512, seed=239, chunk_size=512):
        if mode not in ('raw', 'residual', 'whitened_residual', 'selected_residual') or min(per_source, chunk_size) < 1:
            raise ValueError('Invalid reference configuration')
        self.mode, self.per_source, self.seed, self.chunk_size = mode, per_source, seed, chunk_size

    def _base(self, features, content):
        return features if self.mode == 'raw' else local_residual(features, content)

    def transform(self, features, content):
        x = self._base(features.float(), content)
        if self.mode == 'whitened_residual':
            x = (x - self.mean.to(x)) / self.std.to(x)
        if self.mode == 'selected_residual':x=x[...,self.feature_indices.to(x.device)]
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def fit(self, tokens, valid, content, source_ids, labels):
        if tokens.ndim != 5 or valid.shape != (len(tokens), tokens.shape[2]) or content.shape != (len(tokens), tokens.shape[3]):
            raise ValueError('Expected [source,view,time,patch,width]')
        if len(set(source_ids)) != len(tokens) or len(source_ids) != len(tokens) or labels.shape != (len(tokens),) or (labels != 0).any():
            raise ValueError('Fit accepts distinct training clean sources only')
        if len(tokens) < 2:
            raise ValueError('Need two normal sources for source-excluded calibration')
        # Statistics use all valid normal positions; sampling is only bank size.
        values = []
        for i in range(len(tokens)):
            x = self._base(tokens[i, :, valid[i]].float(), content[i])
            values.append(x[..., content[i], :].reshape(-1, x.shape[-1]))
        all_values = torch.cat(values)
        self.mean = all_values.mean(0)
        std = all_values.std(0, unbiased=False)
        self.std = std.clamp_min(max(float(std.mean()) * .05, 1e-6))
        if self.mode=='selected_residual':
            source_means=torch.stack([tokens[i,:,valid[i]][...,content[i],:].float().mean((0,1,2)) for i in range(len(tokens))])
            within=all_values.square().mean(0);between=source_means.var(0,unbiased=False)
            ratio=within/(between+between.mean().clamp_min(1e-6)*.01)
            self.feature_indices=ratio.argsort(descending=True,stable=True)[:max(1,tokens.shape[-1]//4)]
        self.bank, self.owners = [], []
        generator = torch.Generator().manual_seed(self.seed)
        for source_id, x in zip(source_ids, values):
            take = torch.randperm(len(x), generator=generator)[:self.per_source].to(x.device)
            x = x[take]
            if self.mode == 'whitened_residual':
                x = (x - self.mean) / self.std
            if self.mode == 'selected_residual':x=x[:,self.feature_indices]
            self.bank.append(F.normalize(x, dim=-1))
            self.owners.extend([source_id] * len(x))
        self.bank = torch.cat(self.bank)
        self.source_ids = list(source_ids)
        return self

    @torch.no_grad()
    def score(self, features, content, *, exclude_source=None):
        query = self.transform(features, content)
        bank = self.bank.to(query)
        if exclude_source is not None:
            allowed = torch.tensor([s != exclude_source for s in self.owners], device=query.device)
            bank = bank[allowed]
        if len(bank) == 0:
            raise ValueError('No independent normal reference remains')
        original_shape = query.shape[:-1]
        distances = []
        for part in query.reshape(-1, query.shape[-1]).split(self.chunk_size):
            distances.append((1 - (part @ bank.T).amax(-1)).clamp_min(0))
        result = torch.cat(distances).reshape(original_shape)
        return result.masked_fill(~content, 0)

    def state_dict(self):
        return dict(mode=self.mode, per_source=self.per_source, seed=self.seed,
                    mean=self.mean.cpu(), std=self.std.cpu(), bank=self.bank.cpu(),
                    owners=self.owners, source_ids=self.source_ids)


def automatic_candidates(scores, valid, content, *, low, high, robust_sigma=3.):
    """Conservative, detached pseudo labels; never force a positive patch.

    Requires exceptional absolute novelty, a within-tubelet robust excess, and
    agreement of independently encoded identity/mirror views. Label -1 abstains.
    These are unverified candidates; no liquid extent/position prior is imposed.
    """
    if scores.ndim != 4 or scores.shape[1] != 2 or content.shape != (len(scores), scores.shape[-1]):
        raise ValueError('Expected [source,2,time,patch] and native supports')
    side = int(scores.shape[-1] ** .5)
    if side ** 2 != scores.shape[-1] or not high > low:
        raise ValueError('Need square grid and strictly ordered calibration')
    result = torch.full_like(scores, -1, dtype=torch.long)
    for i in range(len(scores)):
        support = content[i]
        if not torch.equal(support.reshape(side, side), support.reshape(side, side).flip(-1)):
            raise ValueError('Asymmetric support requires explicit reflection geometry')
        for t in torch.where(valid[i])[0].tolist():
            a = scores[i, 0, t]
            b = scores[i, 1, t].reshape(side, side).flip(-1).flatten()
            med = a[support].median();mad = (a[support] - med).abs().median()
            threshold = torch.maximum(a.new_tensor(high), med + robust_sigma * 1.4826 * mad)
            med_b = b[support].median();mad_b = (b[support] - med_b).abs().median()
            threshold_b = torch.maximum(b.new_tensor(high), med_b + robust_sigma * 1.4826 * mad_b)
            fg = support & (a > threshold) & (b > threshold_b)
            bg = support & (a < low) & (b < low)
            result[i, 0, t, fg] = 1;result[i, 0, t, bg] = 0
            result[i, 1, t] = result[i, 0, t].reshape(side, side).flip(-1).flatten()
    return result.detach()
