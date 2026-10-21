import math

import pytest
import torch

from vision_model.training.video_contrastive import (
    ResidualContrastiveHead, masked_temporal_mean, supervised_contrastive_excess, temporal_views,
)


def test_temporal_views_exclude_padding_and_keep_valid_content():
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    views = temporal_views(valid, keep_fraction=0.5, generator=torch.Generator().manual_seed(19))
    assert views.shape == (2, 2, 4)
    assert not (views & ~valid[:, None]).any()
    assert views.any(-1).all()
    features = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    pooled = masked_temporal_mean(features.repeat_interleave(2, 0), views.flatten(0, 1))
    assert torch.isfinite(pooled).all()


def test_residual_head_starts_at_pretrained_embedding_and_receives_gradients():
    torch.manual_seed(7)
    inputs = torch.randn(6, 8)
    head = ResidualContrastiveHead(8, bottleneck=4)
    torch.testing.assert_close(head(inputs), torch.nn.functional.normalize(inputs, dim=-1))
    z = head(inputs)
    ce, entropy, excess = supervised_contrastive_excess(z, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert float(ce) >= float(entropy) - 1e-6
    assert float(excess) >= -1e-6
    excess.backward()
    assert head.up.weight.grad is not None
    assert torch.isfinite(head.up.weight.grad).all()
    assert head.up.weight.grad.abs().sum() > 0


def test_supervised_contrastive_floor_counts_all_other_same_class_views():
    z = torch.tensor([[1., 0.]] * 3 + [[-1., 0.]] * 3)
    ce, entropy, excess = supervised_contrastive_excess(z, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert float(entropy) == pytest.approx(math.log(2), abs=1e-6)
    assert float(excess) < 1e-5


def test_masked_pool_ignores_nonfinite_padding():
    features = torch.tensor([[[1., 2.], [float('nan'), float('inf')]]])
    torch.testing.assert_close(masked_temporal_mean(features, torch.tensor([[True, False]])),
                               torch.tensor([[1., 2.]]))


def test_video_framework_learns_and_replays_predictions(tmp_path):
    from vision_model.training.video_contrastive import VideoRepresentationModel, fit_video_representation
    torch.manual_seed(91)
    labels = torch.tensor([0] * 4 + [1] * 4)
    features = torch.randn(8, 3, 6) * 0.1
    features[:, :, 0] += 4
    features[:, :, 1] += (labels * 2 - 1)[:, None] * 0.4
    valid = torch.ones(8, 3, dtype=torch.bool)
    original = features.clone()
    model, history = fit_video_representation(features, valid, labels, epochs=60,
                                             bottleneck=4, learning_rate=0.03, tether=0.01)
    assert history[-1]['contrastive_excess'] < history[0]['contrastive_excess'] * 0.5
    torch.testing.assert_close(features, original)
    result = model.predict(features, valid)
    assert torch.equal(result['labels'], labels)
    torch.testing.assert_close(result['embeddings'].norm(dim=-1), torch.ones(8))
    path = tmp_path / 'model.pt'
    torch.save(model.state_dict(), path)
    restored = VideoRepresentationModel(6, 4)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for key, value in result.items():
        torch.testing.assert_close(restored.predict(features, valid)[key], value)


def test_video_framework_requires_distinct_positive_videos():
    from vision_model.training.video_contrastive import VideoRepresentationModel, fit_video_representation
    features = torch.randn(3, 2, 4)
    valid = torch.ones(3, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match='two distinct'):
        fit_video_representation(features, valid, torch.tensor([0, 1, 1]))
    with pytest.raises(ValueError, match='Fit training prototypes'):
        VideoRepresentationModel(4).predict(features, valid)


def test_excess_decomposition_distinguishes_leakage_and_nonuniform_positives():
    from vision_model.training.video_contrastive import representation_diagnostics
    torch.manual_seed(42)
    result = representation_diagnostics(torch.randn(8, 5), torch.tensor([0] * 4 + [1] * 4))
    assert result['contrastive_excess'] == pytest.approx(
        result['negative_leakage_loss'] + result['positive_nonuniformity_loss'], abs=2e-6)
    assert result['negative_leakage_loss'] >= 0
    assert result['positive_nonuniformity_loss'] >= 0


def test_convergence_requires_patience_and_does_not_claim_budget_exhaustion():
    from vision_model.training.video_contrastive import fit_video_representation
    labels = torch.tensor([0, 0, 1, 1])
    features = torch.tensor([[[1., 0.]], [[1., 0.]], [[-1., 0.]], [[-1., 0.]]])
    valid = torch.ones(4, 1, dtype=torch.bool)
    _, history = fit_video_representation(features, valid, labels, epochs=10, convergence_patience=3)
    assert len(history) == 3
    assert not history[0]['converged']
    assert history[-1]['converged']
    assert history[-1]['minimum_pair_separation'] > 0
    _, exhausted = fit_video_representation(features, valid, labels, epochs=1, convergence_patience=3)
    assert not exhausted[-1]['converged']


def test_every_anchor_must_separate_not_just_class_averages():
    from vision_model.training.video_contrastive import representation_diagnostics
    z = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.], [-1., 0.], [-1., 0.]])
    result = representation_diagnostics(z, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert result['max_anchor_excess'] > 1e-5
    assert result['minimum_pair_separation'] < 0


def test_margin_loss_has_attainable_zero_with_explicit_geometry():
    from vision_model.training.video_contrastive import supervised_margin_contrastive
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]])
    loss, metrics = supervised_margin_contrastive(separated, labels)
    assert float(loss) == 0
    assert float(metrics['max_margin_violation']) == 0
    assert float(metrics['minimum_positive_cosine']) >= .9
    assert float(metrics['maximum_negative_cosine']) <= .1
    collapsed, _ = supervised_margin_contrastive(torch.ones(4, 2), labels)
    assert float(collapsed) > 0


def test_margin_gradient_pushes_active_negative_pairs_apart():
    from vision_model.training.video_contrastive import supervised_margin_contrastive
    angle = torch.tensor(.2, requires_grad=True)
    left = torch.tensor([1., 0.])
    right = torch.stack((angle.cos(), angle.sin()))
    loss, _ = supervised_margin_contrastive(torch.stack((left, left, right, right)), torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert angle.grad < 0  # Gradient descent increases the class angle.


def test_augmented_views_do_not_replace_distinct_positive_videos():
    from vision_model.training.video_contrastive import supervised_margin_contrastive
    with pytest.raises(ValueError, match='another video'):
        supervised_margin_contrastive(torch.randn(4, 3), torch.tensor([0, 0, 1, 1]),
                                      source_ids=torch.tensor([0, 0, 1, 1]))


def test_augmented_framework_learns_and_excludes_padded_nonfinite_values():
    from vision_model.training.video_contrastive import fit_video_representation
    features = torch.tensor([[[1., 0.]], [[1., .1]], [[-1., 0.]], [[-1., .1]]])
    valid = torch.ones(4, 1, dtype=torch.bool)
    model, history = fit_video_representation(features, valid, torch.tensor([0, 0, 1, 1]),
        objective='margin', augmented_features=features.clone(), epochs=10)
    assert history[-1]['converged']
    assert history[-1]['max_margin_violation'] == 0
    assert torch.equal(model.predict(features, valid)['labels'], torch.tensor([0, 0, 1, 1]))
