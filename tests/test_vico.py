import math

import pytest
import torch
import torch.nn.functional as F

from vision_model.training.vico import vico_pairwise_contrastive, vico_pair_weights


def test_eq10_matches_literal_pairwise_denominators_and_gradients():
    torch.manual_seed(27)
    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    z = F.normalize(x, dim=-1)
    scores = z @ z.T / .1
    positive = labels[:, None].eq(labels[None, :]) & ~torch.eye(6, dtype=torch.bool)
    negative = labels[:, None].ne(labels[None, :])
    weights = torch.linspace(1, 1.7, 36, dtype=torch.float64).reshape(6, 6)
    log_alpha = torch.randn(6, 6, dtype=torch.float64)
    actual, _ = vico_pairwise_contrastive(x, labels, fixed_weights=(weights, log_alpha))
    anchors = []
    for i in range(6):
        negatives = (log_alpha[i, negative[i]].exp() * scores[i, negative[i]].exp()).sum()
        terms = []
        for j in torch.where(positive[i])[0]:
            numerator = weights[i, j] * scores[i, j].exp()
            terms.append(-(numerator / (numerator + negatives + 1e-6)).log())
        anchors.append(torch.stack(terms).mean())
    expected = torch.stack(anchors).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    g_actual = torch.autograd.grad(actual, x, retain_graph=True)[0]
    g_expected = torch.autograd.grad(expected, x)[0]
    torch.testing.assert_close(g_actual, g_expected, rtol=1e-11, atol=1e-11)


def test_eq10_has_no_log_positive_count_floor():
    for n in (2, 7, 25):
        x = torch.tensor([[1., 0.]] * n + [[-1., 0.]] * n, dtype=torch.float64)
        labels = torch.tensor([0] * n + [1] * n)
        loss, metrics = vico_pairwise_contrastive(x, labels, epoch=0)
        # sqrt(N_neg) weighting, epsilon exactly retained, one temperature.
        alpha_mass = math.sqrt(n) * (n * math.exp(-10)) / (n * math.exp(-10) + 1e-6)
        expected = math.log1p(alpha_mass * math.exp(-20) + 1e-6 * math.exp(-10))
        assert float(loss) == pytest.approx(expected, rel=1e-8, abs=1e-15)
        assert float(loss) < 1e-7
        assert float(metrics['minimum_pair_separation']) == pytest.approx(2)


def test_fixed_weight_numerical_gradcheck():
    torch.manual_seed(5)
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    fixed = (torch.ones(4, 4, dtype=torch.float64), torch.zeros(4, 4, dtype=torch.float64))
    assert torch.autograd.gradcheck(lambda values: vico_pairwise_contrastive(
        values, labels, fixed_weights=fixed)[0], x)


def test_weights_follow_equations_and_ties_are_permutation_equivariant():
    x = torch.tensor([[1., 0.]] * 3 + [[-1., 0.]] * 3, dtype=torch.float64)
    labels = torch.tensor([0] * 3 + [1] * 3)
    sources = torch.tensor([0, 0, 1, 2, 2, 3])
    cosine = x @ x.T
    w, log_alpha, u = vico_pair_weights(cosine, labels, sources, epoch=100,
        curriculum_epochs=100, temperature=.1, epsilon=1e-6)
    assert w[0, 1] == 1
    assert w[0, 2] > 1
    assert not w.requires_grad and not log_alpha.requires_grad
    mass = math.sqrt(3) * (3 * math.exp(-10)) / (3 * math.exp(-10) + 1e-6)
    torch.testing.assert_close(log_alpha.exp().sum(1), torch.full((6,), mass, dtype=torch.float64),
                               atol=1e-9, rtol=1e-9)
    permutation = torch.tensor([4, 1, 3, 0, 5, 2])
    wp, ap, up = vico_pair_weights(cosine[permutation][:, permutation], labels[permutation], sources[permutation],
        epoch=100, curriculum_epochs=100, temperature=.1, epsilon=1e-6)
    torch.testing.assert_close(wp, w[permutation][:, permutation])
    torch.testing.assert_close(ap, log_alpha[permutation][:, permutation])
    torch.testing.assert_close(up, u[permutation][:, permutation])


def test_collapse_is_not_convergence_and_conflicting_source_labels_are_rejected():
    x = torch.ones(4, 3)
    labels = torch.tensor([0, 0, 1, 1])
    loss, metrics = vico_pairwise_contrastive(x, labels)
    assert loss > .5
    assert metrics['minimum_pair_separation'] == 0
    with pytest.raises(ValueError, match='conflicting labels'):
        vico_pairwise_contrastive(x, labels, source_ids=torch.tensor([0, 1, 0, 2]))


def test_training_uses_raw_eq10_and_replays_projected_video_embeddings(tmp_path):
    from vision_model.training.video_contrastive import VideoRepresentationModel, fit_video_representation
    features = torch.tensor([[[1., 0.]], [[1., .1]], [[-1., 0.]], [[-1., .1]]])
    labels = torch.tensor([0, 0, 1, 1])
    valid = torch.ones(4, 1, dtype=torch.bool)
    model, history = fit_video_representation(features, valid, labels, objective='vico',
        head_type='mlp', bottleneck=8, embedding_dim=4, learning_rate=.02, epochs=150,
        curriculum_epochs=3, adam_epsilon=1e-12, convergence_tolerance=1e-5,
        augmented_features=features.clone())
    assert history[-1]['objective_loss'] == history[-1]['vico_loss']
    assert history[-1]['converged']
    assert history[-1]['optimizer_resets'] == 1
    assert len(history) >= 7
    result = model.predict(features, valid)
    assert torch.equal(result['labels'], labels)
    assert result['embeddings'].shape == (4, 4)
    path = tmp_path / 'vico_model.pt'
    torch.save(model.state_dict(), path)
    restored = VideoRepresentationModel(2, 8, head_type='mlp', embedding_dim=4)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for key, value in result.items():
        torch.testing.assert_close(restored.predict(features, valid)[key], value)


def test_vico_does_not_claim_zero_or_converge_before_curriculum_finishes():
    from vision_model.training.video_contrastive import fit_video_representation
    x = torch.tensor([[[1., 0.]], [[1., 0.]], [[-1., 0.]], [[-1., 0.]]])
    valid = torch.ones(4, 1, dtype=torch.bool)
    _, history = fit_video_representation(x, valid, torch.tensor([0, 0, 1, 1]),
        objective='vico', epochs=3, curriculum_epochs=10, convergence_patience=1)
    assert history[-1]['vico_loss'] > 0
    assert not history[-1]['converged']
