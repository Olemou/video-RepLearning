"""Properties of target-specific causal attention attribution."""
import torch
import pytest

from vision_model.interpretability.temporal_attribution import (
    AttributionTarget, attribute_batch, attribute_trajectory, capture_attention,
    perturbation_check)
from vision_model.interpretability.video_visualization import save_temporal_figures
from vision_model.training.trajectory import TrajectoryModel


def _case(trained=True):
    torch.manual_seed(31)
    model = TrajectoryModel(16, 16, 8).eval()
    if trained:
        with torch.no_grad():
            torch.nn.init.normal_(model.up.weight, std=.04)
            torch.nn.init.normal_(model.future_delta[-1].weight, std=.04)
    x = torch.randn(7, 16)
    valid = torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    times = torch.tensor([0., .2, .7, 1.3, 2.4, 0., 0.])
    return model, x, valid, times


def test_capture_matches_model_forward_and_restores_method():
    model, x, valid, times = _case()
    expected = model(x[None], valid[None], times[None])
    with capture_attention(model, (0, 1)) as attention:
        got = model(x[None], valid[None], times[None])
    for name in expected:
        torch.testing.assert_close(got[name], expected[name], atol=2e-5, rtol=2e-5)
    assert list(attention) == [0, 1]
    assert all(a.requires_grad for a in attention.values())
    assert "forward" not in model.temporal.layers[0].self_attn.__dict__
    with capture_attention(model, (1,)):
        partial = model(x[None], valid[None], times[None])
    for name in expected:
        torch.testing.assert_close(partial[name], expected[name], atol=2e-5, rtol=2e-5)


def test_captured_attention_has_correct_value_path_gradient():
    model, _, _, _ = _case()
    module = model.temporal.layers[0].self_attn
    token = torch.randn(1, 3, 16)
    mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
    with capture_attention(model, (0,)) as captured:
        output, _ = module(token, token, token, need_weights=False, attn_mask=mask)
        gradient = torch.autograd.grad(output.sum(), captured[0])[0]
    # d(sum(out))/dA[q,k] is projected V[k], independent of q.
    packed = torch.nn.functional.linear(token, module.in_proj_weight, module.in_proj_bias)
    value = packed.reshape(1, 3, 3, 4, 4)[:, :, 2].permute(0, 2, 1, 3)
    projected = module.out_proj.weight.sum(0).reshape(4, 4)
    expected = (value * projected[None, :, None]).sum(-1)[:, :, None, :].expand_as(gradient)
    torch.testing.assert_close(gradient, expected)


def test_causality_padding_and_signed_gradients():
    model, x, valid, times = _case()
    target = AttributionTarget("state", timestep=3, component=2)
    a = attribute_trajectory(model, x, valid, times, target, video_id="a", include_maps=True)
    altered = x.clone()
    altered[4:] += 100
    b = attribute_trajectory(model, altered, valid, times, target, video_id="a", include_maps=True)
    torch.testing.assert_close(a.relevance, b.relevance)
    assert a.target_value == pytest.approx(b.target_value, abs=1e-6)
    assert a.relevance[4:].count_nonzero() == 0
    assert a.signed_gradients[..., 4:, :].count_nonzero() == 0
    assert a.signed_gradients[..., :, 4:].count_nonzero() == 0
    assert a.raw_attention[..., 4:, :].count_nonzero() == 0
    assert a.signed_gradients.abs().sum() > 0
    assert a.event_pooling_weights.shape == (7,)
    assert a.event_pooling_weights[4:].count_nonzero() == 0


def test_gradient_definition_and_variant_are_distinct():
    model, x, valid, times = _case()
    target = AttributionTarget("dynamics", timestep=4, component=1)
    plain = attribute_trajectory(model, x, valid, times, target, video_id="a",
                                 include_maps=True, normalization="none")
    product = attribute_trajectory(model, x, valid, times, target, video_id="a",
                                   method="attention_gradient", include_maps=True,
                                   normalization="none")
    expected = plain.signed_gradients.clamp_min(0)
    torch.testing.assert_close(plain.relevance_maps, expected)
    torch.testing.assert_close(product.relevance_maps,
                               (product.raw_attention * product.signed_gradients).clamp_min(0))
    torch.testing.assert_close(plain.relevance, expected[..., :5, :].mean((0, 1, 2)))
    assert not torch.allclose(plain.relevance, product.relevance)


def test_initialization_can_have_zero_attention_relevance():
    model, x, valid, times = _case(trained=False)
    result = attribute_trajectory(model, x, valid, times,
                                  AttributionTarget("state", 2, 0), video_id="init")
    assert result.relevance.count_nonzero() == 0
    assert result.target_value == pytest.approx(float(x[2, 0]))


def test_metadata_accounts_for_gradients_even_when_full_maps_are_not_saved():
    model, x, valid, times = _case()
    target = AttributionTarget("dynamics", timestep=4, component=1)
    full = attribute_trajectory(model, x, valid, times, target, video_id="a", include_maps=True)
    compact = attribute_trajectory(model, x, valid, times, target, video_id="a")
    summary = compact.metadata()["gradient_statistics"]
    assert compact.signed_gradients is None
    assert summary == full.metadata()["gradient_statistics"]
    assert summary["positive_mass"] == pytest.approx(float(full.signed_gradients.clamp_min(0).sum()))
    assert summary["negative_mass"] == pytest.approx(float((-full.signed_gradients).clamp_min(0).sum()))
    assert summary["nonzero_count"] == int(full.signed_gradients.count_nonzero())
    a, g = full.raw_attention, full.signed_gradients
    expected = a * (g - (a * g).sum(-1, keepdim=True))
    assert summary["softmax_logit_positive_mass"] == pytest.approx(float(expected.clamp_min(0).sum()))
    assert summary["softmax_logit_negative_mass"] == pytest.approx(float((-expected).clamp_min(0).sum()))


def test_targets_and_batch_match_single_samples():
    model, x, valid, times = _case()
    features = torch.stack((x, x + .2))
    mask = valid.expand(2, -1)
    stamps = times.expand(2, -1)
    targets = [AttributionTarget("semantic_similarity", reference=torch.randn(8)),
               AttributionTarget("future_error", timestep=2, future_timestep=4)]
    batch = attribute_batch(model, features, mask, stamps, targets, ["a", "b"])
    for i in range(2):
        single = attribute_trajectory(model, features[i], mask[i], stamps[i], targets[i],
                                      video_id=("a", "b")[i])
        torch.testing.assert_close(batch[i].relevance, single.relevance)
        assert batch[i].target_value == pytest.approx(single.target_value)
    assert batch[0].target["kind"] == "semantic_similarity"
    assert batch[1].target["kind"] == "future_error"
    assert batch[1].relevance[3:].count_nonzero() == 0


def test_future_prediction_and_faithfulness_smoke():
    model, x, valid, times = _case()
    target = AttributionTarget("future_prediction", timestep=2, component=0,
                               future_timestep=4)
    result = attribute_trajectory(model, x, valid, times, target, video_id="a")
    check = perturbation_check(model, x, valid, times, target, result.relevance)
    assert check["high_score_change"] >= 0
    assert check["low_score_change"] >= 0
    assert "out of distribution" in check["caveat"]


def test_invalid_metadata_or_noncausal_cache_rejected():
    model, x, valid, times = _case()
    target = AttributionTarget("state", 2, 0)
    with pytest.raises(ValueError, match="past-only"):
        attribute_trajectory(model, x, valid, times, target, video_id="x",
                             temporal_context_policy="full_video")
    with pytest.raises(ValueError, match="frame index"):
        attribute_trajectory(model, x, valid, times, target, video_id="x",
                             frame_indices=torch.full((7,), -1))
    with pytest.raises(MemoryError):
        attribute_trajectory(model, x, valid, times, target, video_id="x",
                             max_attention_elements=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable in this test environment")
def test_cpu_frame_metadata_with_gpu_model_and_valid_mask():
    model, x, valid, times = _case()
    result = attribute_trajectory(model.cuda(), x.cuda(), valid.cuda(), times.cuda(),
                                  AttributionTarget("state", 2, 0), video_id="gpu",
                                  frame_indices=torch.tensor([1, 3, 5, 7, 9, -1, -1]))
    assert result.frame_indices.device.type == "cpu"
    assert result.relevance.device.type == "cpu"
    assert result.valid_length == 5


def test_static_figures_distinguish_query_map_and_target_row(tmp_path):
    model, x, valid, times = _case()
    result = attribute_trajectory(model, x, valid, times, AttributionTarget("state", 3, 0),
                                  video_id="x", include_maps=True)
    paths = save_temporal_figures(result, tmp_path)
    assert {path.name for path in paths} == {
        "temporal_heatmap.png", "timeline.png", "causal_query_key_maps.png"}
    assert all(path.stat().st_size > 0 for path in paths)
