"""Cluster plotting still produces an artifact without matplotlib."""
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_trajectory import _plot_history_pillow


def test_pillow_plot_accepts_zero_initial_loss(tmp_path):
    history = [dict(epoch=0, total_loss=1., semantic_loss=.5,
                    prediction_loss=.2, persistence_loss=.3, alignment_loss=0.,
                    retention_loss=0., minimum_positive_cosine=.1,
                    maximum_negative_cosine=-.1),
               dict(epoch=1, total_loss=.6, semantic_loss=.2,
                    prediction_loss=.1, persistence_loss=.3, alignment_loss=.01,
                    retention_loss=.02, minimum_positive_cosine=.4,
                    maximum_negative_cosine=-.3)]
    _plot_history_pillow(history, tmp_path, 4, 1)
    with Image.open(tmp_path / "curves.png") as figure:
        assert figure.size == (1200, 800)
