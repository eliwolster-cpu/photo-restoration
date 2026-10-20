from pathlib import Path

import cv2
import numpy as np
import pytest

from restoration.faces import DEFAULT_MODEL, Face, FaceError, compare_faces, detect_faces

REAL_PHOTO = Path("experiments/images/5_faces.jpg")


@pytest.fixture
def photo():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 256, (300, 300), dtype=np.uint8), (0, 0), 4)
    return cv2.cvtColor(cv2.normalize(img, None, 40, 215, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)


def fake_detector(*boxes):
    return lambda image: [Face(x, y, w, h, 0.9) for x, y, w, h in boxes]


def test_unchanged_face_has_high_ssim(photo):
    report = compare_faces(photo, photo.copy(), fake_detector((100, 100, 80, 80)))
    assert report.original_count == report.result_count == 1
    assert report.checks[0].ssim > 0.999 and not report.checks[0].small


def test_changed_face_has_low_ssim(photo):
    result = photo.copy()
    result[100:180, 100:180] = 255 - result[100:180, 100:180]
    report = compare_faces(photo, result, fake_detector((100, 100, 80, 80)))
    assert report.checks[0].ssim < 0.5


def test_small_face_is_marked(photo):
    report = compare_faces(photo, photo.copy(), fake_detector((50, 50, 20, 25)))
    assert report.checks[0].small


def test_face_box_at_image_edge(photo):
    report = compare_faces(photo, photo.copy(), fake_detector((270, 270, 60, 60)))
    assert report.checks[0].ssim > 0.999


def test_no_faces(photo):
    report = compare_faces(photo, photo.copy(), fake_detector())
    assert report.checks == [] and report.original_count == 0


def test_resized_result_is_aligned(photo):
    big = cv2.resize(photo, (900, 900), interpolation=cv2.INTER_CUBIC)
    report = compare_faces(photo, big, fake_detector((100, 100, 80, 80)))
    assert report.checks[0].ssim > 0.95


def test_missing_model(photo):
    with pytest.raises(FaceError):
        detect_faces(photo, model_path="nope.onnx")


@pytest.mark.skipif(not (DEFAULT_MODEL.is_file() and REAL_PHOTO.is_file()), reason="needs model and local test photo")
def test_real_photo_detects_faces_and_marks_them_small():
    img = cv2.imdecode(np.fromfile(str(REAL_PHOTO), np.uint8), cv2.IMREAD_COLOR)
    report = compare_faces(img, img.copy())
    assert report.original_count >= 5 and report.result_count == report.original_count
    assert all(c.small and c.ssim > 0.999 for c in report.checks)
