import pytest
import torch
from vision_model.training.input_attribution import input_integrated_gradients


def test_input_integral_matches_polynomial_and_exact_completeness():
    x=torch.tensor([.4,-.8,1.2],dtype=torch.double)
    baseline=torch.tensor([.1,.2,.3],dtype=torch.double)
    def score(v):return (v**3).sum()
    attribution,diagnostic=input_integrated_gradients(score,x,baseline,steps=4)
    torch.testing.assert_close(attribution.double(),x**3-baseline**3,atol=1e-7,rtol=1e-7)
    assert diagnostic['relative_completeness_error']<1e-12
    assert diagnostic['completeness_check_passed']
    with pytest.raises(ValueError):input_integrated_gradients(score,x,baseline,steps=1)


def test_inadequate_quadrature_is_explicitly_rejected():
    def sharp(v):return torch.tanh(10000*(v[0]-.9999))
    _,diagnostic=input_integrated_gradients(sharp,torch.ones(1,dtype=torch.double),torch.zeros(1,dtype=torch.double),steps=2)
    assert not diagnostic['completeness_check_passed']
    assert 'Numerically unvalidated' in diagnostic['interpretation']
