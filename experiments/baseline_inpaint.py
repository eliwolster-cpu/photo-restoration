"""Baseline inpainting with OpenCV (TELEA and NS) using a manual mask.

Usage: python experiments/baseline_inpaint.py <image> <mask.png> [--out DIR] [--radius N]
The mask is a black/white PNG; white pixels mark damage to be filled.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def read_image(path, flags=cv2.IMREAD_UNCHANGED):
    # np.fromfile + imdecode, because cv2.imread/imwrite fail on non-ASCII Windows paths
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flags)


def write_png(path, image):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buf.tofile(str(path))


def load_mask(path):
    mask = read_image(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"cannot read mask: {path}")
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def inpaint(image, mask, method, radius=3):
    """Inpaint only inside the mask; pixels outside it are copied from the original.

    Supports 8-bit gray, BGR and BGRA (alpha is kept as is). Other bit depths raise ValueError.
    """
    if image.dtype != np.uint8:
        raise ValueError(f"unsupported bit depth {image.dtype}; only 8-bit images are supported")
    if image.ndim == 3 and image.shape[2] == 4:
        out = image.copy()
        out[..., :3] = inpaint(image[..., :3], mask, method, radius)
        return out
    filled = cv2.inpaint(image, mask, radius, METHODS[method])
    out = image.copy()
    out[mask > 0] = filled[mask > 0]
    return out


def side_by_side(image, mask, telea, ns):
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if image.ndim == 2:
        image, telea, ns = (cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) for i in (image, telea, ns))
    elif image.shape[2] == 4:
        image, telea, ns = (i[..., :3] for i in (image, telea, ns))
    return np.hstack([image, mask_bgr, telea, ns])


def run(image_path, mask_path, out_dir, radius=3):
    image = read_image(image_path)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    mask = load_mask(mask_path)
    if mask.shape != image.shape[:2]:
        raise ValueError("mask size does not match image size")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    results = {m: inpaint(image, mask, m, radius) for m in METHODS}
    for m, img in results.items():
        write_png(out_dir / f"{stem}_{m}.png", img)
    write_png(out_dir / f"{stem}_compare.png", side_by_side(image, mask, results["telea"], results["ns"]))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("mask")
    p.add_argument("--out", default="experiments/output")
    p.add_argument("--radius", type=int, default=3)
    a = p.parse_args()
    run(a.image, a.mask, a.out, a.radius)
    print(f"saved to {a.out}")
