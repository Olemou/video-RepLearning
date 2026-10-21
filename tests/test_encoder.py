from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from vision_model.models.checkpoint import VJEPA2Architecture, validate_architecture, validate_loading_info
from vision_model.models.encoder import SupervisedVideoEncoder, VJEPA2Encoder


def valid_config():
    return SimpleNamespace(**VJEPA2Architecture().__dict__)


class EncoderContractTests(unittest.TestCase):
    def test_architecture_mismatch_is_not_suppressed(self) -> None:
        config = valid_config()
        config.hidden_size = 999
        with self.assertRaisesRegex(RuntimeError, "hidden_size"):
            validate_architecture(config, VJEPA2Architecture())

    def test_loading_info_rejects_missing_keys(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing_keys"):
            validate_loading_info({"missing_keys": ["encoder.weight"], "unexpected_keys": []})

    def test_input_layout_and_output_shape(self) -> None:
        config = VJEPA2Architecture(frames_per_clip=4, crop_size=16, patch_size=8,
            tubelet_size=2, hidden_size=96, num_hidden_layers=2,
            num_attention_heads=3, mlp_ratio=2.0)
        model = SupervisedVideoEncoder(VJEPA2Encoder(config))
        output = model(torch.randn(2, 4, 3, 16, 16))
        self.assertEqual(tuple(output.tokens.shape), (2, 8, 96))
        self.assertEqual(tuple(output.temporal_mean.shape), (2, 2, 256))
        self.assertEqual(tuple(output.video_variance.shape), (2, 256))
        self.assertTrue(torch.all(output.temporal_variance > 0))
        self.assertTrue(torch.allclose(output.embedding.norm(dim=-1), torch.ones(2), atol=1e-5))
        with self.assertRaisesRegex(ValueError, "B,T,3"):
            model(torch.zeros(1, 3, 4, 16, 16))


if __name__ == "__main__":
    unittest.main()
