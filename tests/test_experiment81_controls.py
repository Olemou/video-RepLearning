import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_experiment81_shortcuts import image_control
from prepare_experiment81 import normalize_bank
from control_experiment81 import content_cells


def test_image_controls_preserve_time_and_mean_matching_never_clips():
    generator = torch.Generator().manual_seed(11)
    pixels = torch.randint(0, 256, (4, 3, 48, 72), dtype=torch.uint8, generator=generator)
    original = pixels.clone()
    constant = image_control(pixels, "spatial_mean_only")
    assert (constant.float().var((-2, -1)) == 0).all()
    matched = image_control(pixels, "matched_frame_mean")
    torch.testing.assert_close(matched.float().mean((-2, -1)), torch.full((4, 3), 127.5), atol=.51, rtol=0)
    repeated = pixels[:1].expand_as(pixels).clone()
    shuffled = image_control(repeated, "spatial_patch_shuffle")
    assert torch.equal(shuffled[0], shuffled[-1])
    assert torch.equal(pixels, original)
    for t in range(len(pixels)):
        shuffled = image_control(pixels, "spatial_patch_shuffle")
        assert torch.equal(pixels[t].flatten().sort().values, shuffled[t].flatten().sort().values)
    small = torch.randint(0, 256, (2, 3, 120, 160), dtype=torch.uint8, generator=generator)
    shuffled_small = image_control(small, "spatial_patch_shuffle")
    assert torch.equal(small.flatten().sort().values, shuffled_small.flatten().sort().values)


def test_pixel_bank_normalizes_once_and_content_excludes_letterbox():
    pixels = torch.full((2, 4, 3, 32, 32), 128, dtype=torch.uint8)
    result = normalize_bank(pixels, {"mean": [.5] * 3, "std": [.25] * 3})
    expected = torch.full_like(result, ((128 / 255) - .5) / .25)
    torch.testing.assert_close(result, expected)
    support = content_cells(288, 384)
    assert int(support.sum()) == 192
    assert not support[:2].any() and not support[-2:].any()
