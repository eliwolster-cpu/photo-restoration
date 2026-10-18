"""Simple automatic damage mask: morphological top-hat (bright) and black-hat (dark) plus a threshold.

Usage: python experiments/auto_mask.py <image> [--manual MASK.png] [--out DIR]
Saves <stem>_automask.png and <stem>_overlay.png; with --manual also prints IoU.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from baseline_inpaint import read_image, write_png

# Parameters
KERNEL_SIZE = 9      # must be larger than the widest scratch/stain to detect
THRESHOLD = 30       # min top/black-hat response (0-255) to count as damage
DILATE_PX = 1        # grow the mask slightly so inpainting covers the damage edges


def auto_mask(image, kernel_size=KERNEL_SIZE, threshold=THRESHOLD, dilate_px=DILATE_PX):
    """Return a uint8 mask (255 = damage) for bright and dark thin defects."""
    if image.dtype != np.uint8:
        raise ValueError(f"unsupported bit depth {image.dtype}; only 8-bit images are supported")
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bright = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    mask = ((np.maximum(bright, dark) > threshold) * 255).astype(np.uint8)
    if dilate_px > 0:
        size = 2 * dilate_px + 1
        mask = cv2.dilate(mask, np.ones((size, size), np.uint8))
    return mask


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def overlay(image, mask):
    img = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = img[..., :3]  # drop alpha if present
    out = img.copy()
    out[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(img, 0.5, out, 0.5, 0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--manual")
    p.add_argument("--out", default="experiments/output")
    a = p.parse_args()
    image = read_image(a.image)
    if image is None:
        raise SystemExit(f"cannot read image: {a.image}")
    mask = auto_mask(image)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.image).stem
    write_png(out / f"{stem}_automask.png", mask)
    write_png(out / f"{stem}_overlay.png", overlay(image, mask))
    print(f"mask area: {100 * (mask > 0).mean():.2f}%")
    if a.manual:
        manual = read_image(a.manual, cv2.IMREAD_GRAYSCALE)
        print(f"IoU vs manual: {iou(mask, manual):.3f}")
