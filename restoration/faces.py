"""Face regions: detect faces in the original and compare each region with the result.

SSIM only shows that a face region changed; it cannot prove identity (open decision in
docs/plan.md). Small faces are reported separately because detection and comparison are
unreliable on them.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from restoration.compare import _to_float_gray, ssim_map

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.6
MIN_FACE_SIDE = 40       # pixels; smaller faces are marked small (thresholds move to config in step 10)
PAD = 0.2                # share of the box added on each side before comparing
BLUR_SIGMA = 1.0


class FaceError(Exception):
    """Face detection is unavailable."""


@dataclass(frozen=True)
class Face:
    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass(frozen=True)
class FaceCheck:
    box: Face
    small: bool      # shorter side below MIN_FACE_SIDE
    ssim: float      # SSIM between original and result in the padded face box


@dataclass(frozen=True)
class FacesReport:
    checks: list
    original_count: int
    result_count: int   # faces detected in the result; fewer than the original is suspicious


def _as_bgr(image):
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image[:, :, :3]


def detect_faces(image, model_path=DEFAULT_MODEL, score_threshold=SCORE_THRESHOLD):
    """Detect faces with YuNet. Returns a list of Face."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FaceError(f"face model not found: {model_path} (run: python models/download_models.py)")
    image = _as_bgr(image)
    h, w = image.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(model_path), "", (w, h), score_threshold)
    _, found = detector.detect(image)
    if found is None:
        return []
    faces = []
    for row in found:
        x, y, fw, fh = (int(round(v)) for v in row[:4])
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + fw, w), min(y + fh, h)
        if x1 > x0 and y1 > y0:
            faces.append(Face(x0, y0, x1 - x0, y1 - y0, float(row[-1])))
    return faces


def compare_faces(original, result, detector=detect_faces, min_face_side=MIN_FACE_SIDE):
    """Compare every face region of the original with the same region of the result."""
    if result.shape[:2] != original.shape[:2]:
        h, w = original.shape[:2]
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(_to_float_gray(original), (0, 0), BLUR_SIGMA)
    b = cv2.GaussianBlur(_to_float_gray(result), (0, 0), BLUR_SIGMA)
    h, w = a.shape

    faces = detector(original)
    checks = []
    for f in faces:
        px, py = int(f.w * PAD), int(f.h * PAD)
        x0, y0 = max(f.x - px, 0), max(f.y - py, 0)
        x1, y1 = min(f.x + f.w + px, w), min(f.y + f.h + py, h)
        ssim = float(ssim_map(a[y0:y1, x0:x1], b[y0:y1, x0:x1]).mean())
        checks.append(FaceCheck(f, min(f.w, f.h) < min_face_side, ssim))
    return FacesReport(checks, len(faces), len(detector(result)))
