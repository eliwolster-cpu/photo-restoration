"""Stage 6: clean local damage (cracks, stains, scratches) strictly inside a damage mask.

Whatever the backend does, the result is blended back onto the original so every pixel
outside the mask is exactly the original pixel.
"""
import logging
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("restoration.clean")

CV_METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}
BACKENDS = (*CV_METHODS, "lama")


class CleanError(Exception):
    """Cleaning cannot run (bad input, unknown or unavailable backend)."""


@dataclass(frozen=True)
class CleanResult:
    image: np.ndarray
    mask: np.ndarray            # uint8, 255 = treated
    backend: str
    mask_fraction: float        # share of the image covered by the mask
    outside_identical: bool     # True if every pixel outside the mask equals the original


def detect_damage(image, kernel_size=9, threshold=30, dilate_px=1):
    """Automatic mask (255 = damage) from morphological top-hat/black-hat on thin defects."""
    gray = _gray(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    response = np.maximum(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
                          cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
    mask = ((response > threshold) * 255).astype(np.uint8)
    if dilate_px > 0:
        mask = cv2.dilate(mask, np.ones((2 * dilate_px + 1,) * 2, np.uint8))
    return mask


def _gray(image):
    if image.dtype != np.uint8:
        raise CleanError(f"unsupported bit depth {image.dtype}; only 8-bit images are supported")
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)


def _inpaint_opencv(image, mask, method, radius):
    return cv2.inpaint(image, mask, radius, CV_METHODS[method])


def _inpaint_lama(image, mask):
    # Heavy optional dependency (torch); deliberately not in requirements.txt (see CLAUDE.md).
    try:
        from PIL import Image
        from simple_lama_inpainting import SimpleLama
    except ImportError as e:
        raise CleanError("backend 'lama' needs the optional package simple-lama-inpainting "
                         "(pip install simple-lama-inpainting)") from e
    rgb = cv2.cvtColor(image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
                       cv2.COLOR_BGR2RGB)
    out = np.array(SimpleLama()(Image.fromarray(rgb), Image.fromarray(mask)))[: image.shape[0], : image.shape[1]]
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out if image.ndim == 3 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


def blend_to_original(original, result, mask):
    """Take `result` inside the mask and the original everywhere else."""
    out = original.copy()
    out[mask > 0] = result[mask > 0]
    return out


def clean(image, cfg, mask=None, name=""):
    """Inpaint the damage and return CleanResult.

    cfg is the `clean` section of config.yaml. If `mask` is None it is detected
    automatically; a given mask is binarized (nonzero = damage). Alpha is kept as is.
    """
    backend = cfg.get("backend", "telea")
    if backend not in BACKENDS:
        raise CleanError(f"unknown clean backend '{backend}' (choose from {BACKENDS})")
    _gray(image)  # validates bit depth
    if mask is None:
        m = cfg.get("mask", {})
        mask = detect_damage(image, m.get("kernel_size", 9), m.get("threshold", 30), m.get("dilate_px", 1))
    elif mask.shape[:2] != image.shape[:2]:
        raise CleanError(f"mask size {mask.shape[:2]} does not match image {image.shape[:2]}")
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    fraction = float((mask > 0).mean())

    if fraction == 0:
        result = image.copy()
    else:
        alpha = image[..., 3:] if image.ndim == 3 and image.shape[2] == 4 else None
        color = image[..., :3] if alpha is not None else image
        if backend == "lama":
            filled = _inpaint_lama(color, mask)
        else:
            filled = _inpaint_opencv(color, mask, backend, cfg.get("radius", 3))
        result = blend_to_original(image, np.concatenate([filled, alpha], axis=2) if alpha is not None else filled, mask)

    identical = bool(np.array_equal(result[mask == 0], image[mask == 0]))
    log.info("clean %s: backend=%s mask=%.2f%% outside_mask_identical=%s",
             name or "-", backend, 100 * fraction, identical)
    return CleanResult(result, mask, backend, fraction, identical)
