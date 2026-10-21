import cv2
import numpy as np
import pytest

from restoration import clean as clean_mod
from restoration.clean import CleanError, blend_to_original, clean, detect_damage
from restoration.sharpen import sharpen

CFG = {"backend": "telea", "radius": 3}


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (200, 200), dtype=np.uint8), (0, 0), 8)
    img = cv2.normalize(img, None, 60, 190, cv2.NORM_MINMAX)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


@pytest.fixture
def scratched(photo):
    damaged = photo.copy()
    damaged[:, 98:101] = 255
    mask = np.zeros((200, 200), np.uint8)
    mask[:, 96:103] = 255
    return photo, damaged, mask


@pytest.mark.parametrize("backend", ["telea", "ns"])
def test_damage_removed_and_outside_mask_exact(scratched, backend):
    clean_img, damaged, mask = scratched
    r = clean(damaged, {**CFG, "backend": backend}, mask)
    assert r.outside_identical
    assert np.array_equal(r.image[mask == 0], damaged[mask == 0])
    err_before = np.abs(damaged[:, 98:101].astype(int) - clean_img[:, 98:101]).mean()
    err_after = np.abs(r.image[:, 98:101].astype(int) - clean_img[:, 98:101]).mean()
    assert err_after < err_before / 3


def test_blend_forces_original_outside_mask(photo):
    mask = np.zeros(photo.shape[:2], np.uint8)
    mask[10:20, 10:20] = 255
    out = blend_to_original(photo, 255 - photo, mask)
    assert np.array_equal(out[mask == 0], photo[mask == 0])
    assert np.array_equal(out[mask > 0], (255 - photo)[mask > 0])


def test_auto_mask_finds_scratch(scratched):
    _, damaged, mask = scratched
    auto = detect_damage(damaged)
    assert (auto[:, 98:101] > 0).mean() > 0.9


def test_empty_mask_returns_unchanged(photo):
    r = clean(photo, CFG, np.zeros(photo.shape[:2], np.uint8))
    assert np.array_equal(r.image, photo) and r.mask_fraction == 0


def test_gray_and_alpha_supported(scratched):
    _, damaged, mask = scratched
    gray = cv2.cvtColor(damaged, cv2.COLOR_BGR2GRAY)
    assert clean(gray, CFG, mask).outside_identical
    bgra = np.dstack([damaged, np.full(mask.shape, 7, np.uint8)])
    r = clean(bgra, CFG, mask)
    assert r.outside_identical and (r.image[..., 3] == 7).all()


def test_errors(photo):
    with pytest.raises(CleanError):
        clean(photo, {"backend": "magic"})
    with pytest.raises(CleanError):
        clean(photo, CFG, np.zeros((5, 5), np.uint8))
    with pytest.raises(CleanError):
        clean(photo.astype(np.uint16), CFG)


def test_lama_backend_missing_dependency_is_clear(scratched, monkeypatch):
    _, damaged, mask = scratched
    monkeypatch.setitem(__import__("sys").modules, "simple_lama_inpainting", None)
    with pytest.raises(CleanError, match="simple-lama-inpainting"):
        clean(damaged, {"backend": "lama"}, mask)


def test_lama_result_is_still_blended(scratched, monkeypatch):
    _, damaged, mask = scratched
    monkeypatch.setattr(clean_mod, "_inpaint_lama", lambda image, m: np.zeros_like(image))
    r = clean(damaged, {"backend": "lama"}, mask)
    assert r.outside_identical and (r.image[mask > 0] == 0).all()


def test_sharpen_increases_sharpness_and_keeps_size(photo):
    s = sharpen(photo, 0.5, 1.0)
    lap = lambda x: cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    assert s.shape == photo.shape and lap(s) > lap(photo)


def _sepia_with_stain():
    rng = np.random.default_rng(1)
    gray = cv2.GaussianBlur(rng.integers(120, 160, (200, 200), dtype=np.uint8), (0, 0), 3)
    img = cv2.merge([(gray * 0.8).astype(np.uint8), (gray * 0.9).astype(np.uint8), gray])  # B, G, R: sepia
    stained = img.copy()
    cv2.circle(stained, (100, 100), 8, (40, 40, 150), -1)   # reddish-brown blotch
    return img, cv2.GaussianBlur(stained, (0, 0), 1.2)


def test_stain_is_corrected_without_touching_the_rest():
    clean_img, stained = _sepia_with_stain()
    cfg = {"backend": "telea", "stains": {"enabled": True, "border_margin": 0}}
    r = clean(stained, cfg, np.zeros(stained.shape[:2], np.uint8))
    assert r.stain_fraction > 0 and r.outside_identical
    err_before = np.abs(stained[90:110, 90:110].astype(int) - clean_img[90:110, 90:110]).mean()
    err_after = np.abs(r.image[90:110, 90:110].astype(int) - clean_img[90:110, 90:110]).mean()
    assert err_after < err_before / 2


def test_stains_are_off_by_default_and_skip_faces():
    _, stained = _sepia_with_stain()
    off = clean(stained, CFG, np.zeros(stained.shape[:2], np.uint8))
    assert off.stain_fraction == 0 and np.array_equal(off.image, stained)
    cfg = {"backend": "telea", "stains": {"enabled": True, "border_margin": 0}}
    on_face = clean(stained, cfg, np.zeros(stained.shape[:2], np.uint8), exclude_boxes=[(60, 60, 80, 80)])
    assert on_face.stain_fraction == 0
