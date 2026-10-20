"""Independent comparison of a result against the original, outside the treated area.

Global sharpening changes high frequencies everywhere, so both images are low-passed
before comparing: what remains is structural change (content moved, added or removed).
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Defaults for the caller; thresholds for decisions live in config.yaml (plan step 10).
BLUR_SIGMA = 2.0
MASK_MARGIN = 5          # pixels around the mask excluded from the comparison
DIFF_TOLERANCE = 12      # gray levels (0-255) on the low-passed images


class CompareError(Exception):
    """Images cannot be compared."""


@dataclass(frozen=True)
class OutsideMaskReport:
    resized: bool            # result had a different size and was resized to the original
    compared_fraction: float  # share of the image that was compared (outside mask + margin)
    changed_fraction: float   # share of compared pixels whose low-passed diff exceeds tolerance
    mean_abs_diff: float
    max_abs_diff: float
    ssim: float               # mean SSIM over compared pixels, 1.0 = identical


def _to_float_gray(image):
    if image.dtype == np.uint16:
        image = (image / 257.0).astype(np.float32)
    else:
        image = image.astype(np.float32)
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def ssim_map(a, b, sigma=1.5):
    """Per-pixel SSIM of two float grayscale images in 0-255 (Gaussian window)."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), sigma)
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def compare_outside_mask(original, result, mask):
    """Compare `result` with `original` everywhere except the mask (and a margin around it).

    mask: uint8, nonzero = treated area, same size as the original.
    """
    if mask.shape[:2] != original.shape[:2]:
        raise CompareError(f"mask size {mask.shape[:2]} does not match original {original.shape[:2]}")
    resized = result.shape[:2] != original.shape[:2]
    if resized:
        h, w = original.shape[:2]
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)

    a = cv2.GaussianBlur(_to_float_gray(original), (0, 0), BLUR_SIGMA)
    b = cv2.GaussianBlur(_to_float_gray(result), (0, 0), BLUR_SIGMA)

    treated = (mask > 0).astype(np.uint8)
    if MASK_MARGIN > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * MASK_MARGIN + 1,) * 2)
        treated = cv2.dilate(treated, kernel)
    outside = treated == 0
    if not outside.any():
        raise CompareError("nothing left to compare outside the mask")

    diff = np.abs(a - b)[outside]
    return OutsideMaskReport(
        resized=resized,
        compared_fraction=float(outside.mean()),
        changed_fraction=float((diff > DIFF_TOLERANCE).mean()),
        mean_abs_diff=float(diff.mean()),
        max_abs_diff=float(diff.max()),
        ssim=float(ssim_map(a, b)[outside].mean()),
    )
