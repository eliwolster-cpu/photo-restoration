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
    stain_fraction: float = 0.0  # part of the mask that is stain correction rather than crack inpainting


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


def detect_stains(image, cfg, exclude_boxes=()):
    """Mask (255) of compact colored blotches: chroma far from the local median, not on faces.

    Only color images are checked (a gray scan has no chroma to tell a stain from the scene).
    """
    if image.ndim != 3 or image.shape[2] < 3:
        return np.zeros(image.shape[:2], np.uint8)
    lab = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2LAB)
    chroma = lab[..., 1:].astype(np.float32) - cv2.medianBlur(lab, cfg.get("background_size", 61))[..., 1:]
    dev = cv2.GaussianBlur(np.hypot(chroma[..., 0], chroma[..., 1]), (0, 0), 1.5)
    # hysteresis: a region needs a strong core (chroma_threshold) and grows to its weaker edge
    low = cv2.morphologyEx(((dev > cfg.get("chroma_low", 5)) * 255).astype(np.uint8), cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, st, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    strong = np.unique(labels[dev > cfg.get("chroma_threshold", 10)])
    cand = np.where(np.isin(labels, strong[strong > 0]), 255, 0).astype(np.uint8)
    margin = int(cfg.get("border_margin", 0.03) * min(cand.shape))
    if margin > 0:   # the scan edge against the white background looks like a stain
        cand[:margin] = cand[-margin:] = 0
        cand[:, :margin] = cand[:, -margin:] = 0
    n, labels, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    max_area = cfg.get("max_area_fraction", 0.01) * cand.size
    mask = np.zeros_like(cand)
    for i in range(1, n):
        if cfg.get("min_area", 25) <= st[i, cv2.CC_STAT_AREA] <= max_area:
            mask[labels == i] = 255
    grow = cfg.get("dilate_px", 6)
    if grow > 0:
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    for x, y, w, h in exclude_boxes:
        mask[max(y, 0):y + h, max(x, 0):x + w] = 0
    return mask


def correct_stains(image, mask, cfg):
    """Fix brightness and color inside the mask without inventing detail.

    Per Lab channel: new = low-frequency level of the surroundings (from pixels outside the
    mask) + the pixel's own fine detail, so grain and real edges under a stain survive.
    """
    if not mask.any():
        return image.copy()
    sigma = cfg.get("detail_sigma", 3.0)
    reach = cfg.get("surround_sigma", 12.0)
    lab = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    known = (mask == 0).astype(np.float32)
    out = lab.copy()
    for c in range(3):
        ch = lab[..., c]
        detail = ch - cv2.GaussianBlur(ch, (0, 0), sigma)
        level = cv2.GaussianBlur(ch * known, (0, 0), reach) / np.maximum(cv2.GaussianBlur(known, (0, 0), reach), 1e-3)
        out[..., c] = level + detail
    fixed = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    result = image.copy()
    result[..., :3][mask > 0] = fixed[mask > 0]
    return result


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


def clean(image, cfg, mask=None, name="", exclude_boxes=()):
    """Correct stains and inpaint cracks; return CleanResult.

    cfg is the `clean` section of config.yaml. `mask` (cracks) is detected automatically if
    None; a given mask is binarized (nonzero = damage). If cfg["stains"]["enabled"], colored
    blotches outside `exclude_boxes` (x, y, w, h) get brightness/color correction first.
    Everything outside the union of both masks is the original. Alpha is kept as is.
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

    stains_cfg = cfg.get("stains", {})
    stain_mask = detect_stains(image, stains_cfg, exclude_boxes) if stains_cfg.get("enabled", False)         else np.zeros_like(mask)
    union = np.maximum(mask, stain_mask)
    fraction = float((union > 0).mean())
    stain_fraction = float((stain_mask > 0).mean())

    result = image.copy()
    if stain_mask.any():
        result = correct_stains(result, stain_mask, stains_cfg)
    if mask.any():
        alpha = result[..., 3:] if result.ndim == 3 and result.shape[2] == 4 else None
        color = result[..., :3] if alpha is not None else result
        if backend == "lama":
            filled = _inpaint_lama(color, mask)
        else:
            filled = _inpaint_opencv(color, mask, backend, cfg.get("radius", 3))
        result = blend_to_original(result, np.concatenate([filled, alpha], axis=2) if alpha is not None else filled, mask)
    result = blend_to_original(image, result, union)

    identical = bool(np.array_equal(result[union == 0], image[union == 0]))
    log.info("clean %s: backend=%s mask=%.2f%% (stains %.2f%%) outside_mask_identical=%s",
             name or "-", backend, 100 * fraction, 100 * stain_fraction, identical)
    return CleanResult(result, union, backend, fraction, identical, stain_fraction)
