import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import baseline_inpaint as bi  # noqa: E402


def synthetic():
    """Smooth gradient image with two white scratches, plus the exact mask."""
    y, x = np.mgrid[0:128, 0:128]
    gray = (x + y).astype(np.uint8)
    image = cv2.merge([gray, gray, gray])
    mask = np.zeros((128, 128), np.uint8)
    cv2.line(mask, (10, 20), (110, 100), 255, 3)
    cv2.line(mask, (100, 10), (30, 120), 255, 3)
    damaged = image.copy()
    damaged[mask > 0] = 255
    return image, damaged, mask


@pytest.mark.parametrize("method", ["telea", "ns"])
def test_scratches_removed_and_outside_mask_unchanged(method):
    clean, damaged, mask = synthetic()
    out = bi.inpaint(damaged, mask, method)
    assert np.array_equal(out[mask == 0], damaged[mask == 0])
    err = np.abs(out[mask > 0].astype(int) - clean[mask > 0].astype(int)).mean()
    assert err < 10  # damaged pixels were 255, far from the true gradient


def test_run_saves_outputs(tmp_path):
    clean, damaged, mask = synthetic()
    cv2.imwrite(str(tmp_path / "img.png"), damaged)
    cv2.imwrite(str(tmp_path / "mask.png"), mask)
    bi.run(tmp_path / "img.png", tmp_path / "mask.png", tmp_path / "out")
    for name in ("img_telea.png", "img_ns.png", "img_compare.png"):
        assert (tmp_path / "out" / name).exists()
    assert cv2.imread(str(tmp_path / "out" / "img_compare.png")).shape[1] == 128 * 4
