import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import auto_mask as am  # noqa: E402


def synthetic():
    """Gradient image with a bright scratch and a dark scratch, plus the exact mask."""
    y, x = np.mgrid[0:128, 0:128]
    gray = (60 + (x + y) // 2).astype(np.uint8)
    image = cv2.merge([gray, gray, gray])
    mask = np.zeros((128, 128), np.uint8)
    cv2.line(mask, (10, 20), (110, 100), 255, 3)
    cv2.line(mask, (100, 10), (30, 120), 255, 3)
    damaged = image.copy()
    cv2.line(damaged, (10, 20), (110, 100), (255, 255, 255), 3)
    cv2.line(damaged, (100, 10), (30, 120), (0, 0, 0), 3)
    return damaged, mask


def test_iou_against_manual_mask_is_high():
    damaged, manual = synthetic()
    assert am.iou(am.auto_mask(damaged, dilate_px=0), manual) > 0.7


def test_default_mask_covers_manual_mask():
    # the default 1px dilation lowers IoU but must still cover all the damage
    damaged, manual = synthetic()
    auto = am.auto_mask(damaged) > 0
    assert auto[manual > 0].all()


def test_clean_image_gives_empty_mask():
    damaged, manual = synthetic()
    y, x = np.mgrid[0:128, 0:128]
    gray = (60 + (x + y) // 2).astype(np.uint8)
    assert am.auto_mask(gray).sum() == 0


def test_iou_edge_cases():
    z = np.zeros((4, 4), np.uint8)
    assert am.iou(z, z) == 1.0
    o = z.copy()
    o[0, 0] = 255
    assert am.iou(z, o) == 0.0
