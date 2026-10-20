import cv2
import numpy as np
import pytest

from restoration.compare import CompareError, compare_outside_mask


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (200, 200), dtype=np.uint8), (0, 0), 6)
    img = cv2.normalize(img, None, 40, 215, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def mask():
    m = np.zeros((200, 200), np.uint8)
    m[90:110, 20:180] = 255
    return m


def test_identical_means_no_change(photo, mask):
    r = compare_outside_mask(photo, photo.copy(), mask)
    assert r.changed_fraction == 0 and r.max_abs_diff == 0 and r.ssim > 0.999 and not r.resized


def test_change_inside_mask_is_ignored(photo, mask):
    result = photo.copy()
    result[95:105, 40:160] = 255
    r = compare_outside_mask(photo, result, mask)
    assert r.changed_fraction == 0


def test_injected_change_outside_mask_is_flagged(photo, mask):
    result = photo.copy()
    result[20:60, 20:60] = 255 - result[20:60, 20:60]
    r = compare_outside_mask(photo, result, mask)
    assert r.changed_fraction > 0.02 and r.ssim < 0.99


def test_global_sharpening_is_not_flagged(photo, mask):
    soft = cv2.GaussianBlur(photo, (0, 0), 1.0)
    sharp = cv2.addWeighted(photo, 1.5, soft, -0.5, 0)
    r = compare_outside_mask(photo, sharp, mask)
    assert r.changed_fraction == 0 and r.ssim > 0.98


def test_resized_result_is_aligned(photo, mask):
    big = cv2.resize(photo, (800, 800), interpolation=cv2.INTER_CUBIC)
    r = compare_outside_mask(photo, big, mask)
    assert r.resized and r.changed_fraction == 0


def test_grayscale_and_16bit(photo, mask):
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    r = compare_outside_mask(gray.astype(np.uint16) * 257, gray, mask)
    assert r.changed_fraction == 0


def test_mask_size_mismatch(photo):
    with pytest.raises(CompareError):
        compare_outside_mask(photo, photo, np.zeros((10, 10), np.uint8))


def test_full_mask_raises(photo):
    with pytest.raises(CompareError):
        compare_outside_mask(photo, photo, np.full((200, 200), 255, np.uint8))
