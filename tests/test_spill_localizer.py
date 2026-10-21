import torch

from vision_model.models.spill_localizer import (
    SpillMILHead,
    SpillSegmentationHead,
    spill_mil_loss,
    spill_segmentation_loss,
)


def test_localizer_masks_padding_and_backpropagates_to_head():
    torch.manual_seed(7)
    model = SpillMILHead(hidden_size=8, projection_size=4, top_fraction=.25)
    tokens = torch.randn(2, 3, 2, 2, 8)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    output = model(tokens, valid)
    assert torch.isneginf(output.token_logits[~valid]).all()
    loss, parts = spill_mil_loss(output, torch.tensor([0, 1]), valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0


def test_top_instance_controls_video_logit():
    model = SpillMILHead(hidden_size=2, projection_size=2, top_fraction=.25)
    model.token_head = torch.nn.Identity()
    # Spatial smoothing means the largest 3x3 neighborhood average controls it.
    tokens = torch.tensor([[[[[1.], [4.]], [[2.], [3.]]]]])
    output = model(tokens, torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(output.video_logits, torch.tensor([10 / 9]))


def test_isolated_peak_is_diluted_by_neighborhood_pooling():
    model = SpillMILHead(hidden_size=1, projection_size=1, top_fraction=.04)
    model.token_head = torch.nn.Identity()
    tokens = torch.zeros(1, 1, 5, 5, 1); tokens[0, 0, 2, 2, 0] = 9
    output = model(tokens, torch.ones(1, 1, dtype=torch.bool))
    torch.testing.assert_close(output.video_logits, torch.tensor([1.]))


def test_segmentation_head_and_mask_loss_backpropagate():
    torch.manual_seed(11)
    model = SpillSegmentationHead(hidden_size=8, decoder_size=4)
    tokens = torch.randn(2, 3, 4, 4, 8)
    targets = torch.zeros(2, 3, 4, 4)
    targets[1, 0, 1:3, 1:3] = 1
    supervised = torch.tensor([[True, False, False], [True, True, False]])
    logits = model(tokens)
    loss, parts = spill_segmentation_loss(
        logits, targets, supervised, positive_weight=3.0
    )
    loss.backward()
    assert logits.shape == targets.shape
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())
    assert sum(parameter.grad.abs().sum() for parameter in model.parameters()) > 0


def test_segmentation_loss_ignores_unannotated_tubelets():
    logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
    targets = torch.zeros_like(logits)
    targets[:, 1] = 1
    loss, _ = spill_segmentation_loss(
        logits, targets, torch.tensor([[True, False]])
    )
    loss.backward()
    assert logits.grad[:, 0].abs().sum() > 0
    assert logits.grad[:, 1].abs().sum() == 0


def test_segmentation_loss_rejects_missing_annotations():
    with torch.no_grad(), __import__("pytest").raises(ValueError, match="annotated"):
        spill_segmentation_loss(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, dtype=torch.bool),
        )
