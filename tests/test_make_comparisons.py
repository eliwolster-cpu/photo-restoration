import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import make_comparisons as mc  # noqa: E402


def img(w, h, v):
    return np.full((h, w, 3), v, np.uint8)


def test_panel_adds_label_bar_and_resizes():
    p = mc.panel("x", img(400, 300, 100), (80, 60))
    assert p.shape == (60 + mc.LABEL_H, 80, 3)


def test_missing_image_gives_gray_panel():
    p = mc.panel("x", None, (100, 50))
    assert p.shape == (50 + mc.LABEL_H, 100, 3) and p[mc.LABEL_H + 2, 2, 0] == 128


def test_compose_pads_last_row():
    panels = [mc.panel(str(i), img(10, 10, 50), (10, 10)) for i in range(5)]
    out = mc.compose(panels)
    assert out.shape == (2 * (10 + mc.LABEL_H), 30, 3)


def test_build_with_only_original(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imencode(".png", img(60, 40, 90))[1].tofile(str(images / "1_cracks.png"))
    out = mc.build("1_cracks", images, tmp_path / "t", tmp_path / "b")
    assert out.shape == (2 * (40 + mc.LABEL_H), 180, 3)
