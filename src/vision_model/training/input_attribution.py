"""Straight-line input integrated gradients with explicit quadrature diagnostics."""
from __future__ import annotations
import numpy as np
import torch


def input_integrated_gradients(score,input_tensor,baseline,*,steps=16):
    """Gauss-Legendre quadrature of gradients of one explicit scalar.

    This runs differentiable image forwards, never attributes through a cache.
    Completeness error is reported rather than assumed small for finite steps
    or quantized arithmetic. Only the input is differentiated.
    """
    if steps<2 or input_tensor.shape!=baseline.shape:
        raise ValueError('Need matching input/baseline and at least two quadrature points')
    if not torch.isfinite(input_tensor).all() or not torch.isfinite(baseline).all():
        raise ValueError('Inputs must be finite')
    nodes,weights=np.polynomial.legendre.leggauss(steps)
    difference=input_tensor.detach()-baseline.detach()
    accumulator=torch.zeros_like(input_tensor,dtype=torch.float64)
    for node,weight in zip((nodes+1)/2,weights/2):
        observed=(baseline.detach()+float(node)*difference).requires_grad_(True)
        scalar=score(observed)
        if scalar.ndim!=0 or not torch.isfinite(scalar):raise ValueError('Attribution target must be one finite scalar')
        gradient,=torch.autograd.grad(scalar,observed)
        if not torch.isfinite(gradient).all():raise RuntimeError('Nonfinite input gradient')
        accumulator+=float(weight)*gradient.double()
    attribution=difference.double()*accumulator
    with torch.no_grad():
        output=float(score(input_tensor));reference=float(score(baseline))
    residual=float(attribution.sum())-(output-reference)
    checked=abs(residual)<=1e-3+.05*abs(output-reference)
    return attribution.float(),dict(steps=steps,target=output,baseline_target=reference,
        attribution_sum=float(attribution.sum()),completeness_residual=residual,
        relative_completeness_error=abs(residual)/max(abs(output-reference),1e-8),
        completeness_check_passed=checked,
        completeness_tolerance='absolute 1e-3 + 5% of target difference; a numerical check, not a localization metric',
        interpretation='Quadrature passes the completeness check; spatial correctness remains unverified' if checked else
                       'Numerically unvalidated integral: do not interpret as a complete explanation')
