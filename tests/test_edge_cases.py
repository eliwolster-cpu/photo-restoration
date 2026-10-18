"""Edge cases: alpha/16-bit images, Hebrew filenames, fake images, and input immutability."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import auto_mask  # noqa: E402
import baseline_inpaint as bi  # noqa: E402
import make_mask  # noqa: E402
from restoration.ingest import IngestError, ingest, sha256_file  # noqa: E402


def gradient(channels=3, dtype=np.uint8):
    y, x = np.mgrid[0:64, 0:64]
    g = (x + y).astype(np.uint8)
    img = g if channels == 1 else cv2.merge([g] * channels)
    if channels == 4:
        img[..., 3] = 200  # non-opaque alpha
    return img.astype(np.uint16) * 257 if dtype == np.uint16 else img


def scratch_mask():
    mask = np.zeros((64, 64), np.uint8)
    cv2.line(mask, (5, 5), (58, 50), 255, 3)
    return mask


def save(path, image):
    bi.write_png(path, image) if path.suffix == ".png" else cv2.imencode(path.suffix, image)[1].tofile(str(path))
    return path


# --- alpha PNG and 16-bit TIFF in baseline_inpaint -------------------------------------

def test_png_with_alpha_inpainted_and_alpha_kept(tmp_path):
    img, mask = gradient(4), scratch_mask()
    damaged = img.copy()
    damaged[mask > 0, :3] = 255
    img_path, mask_path = save(tmp_path / "a.png", damaged), save(tmp_path / "m.png", mask)
    before = sha256_file(img_path)
    results = bi.run(img_path, mask_path, tmp_path / "out")
    for out in results.values():
        assert out.shape == (64, 64, 4)
        assert np.array_equal(out[..., 3], damaged[..., 3])
        assert np.array_equal(out[mask == 0], damaged[mask == 0])
        assert np.abs(out[mask > 0, :3].astype(int) - img[mask > 0, :3]).mean() < 10
    assert cv2.imread(str(tmp_path / "out" / "a_compare.png")).shape == (64, 256, 3)
    assert sha256_file(img_path) == before


@pytest.mark.parametrize("channels", [1, 3])
def test_16bit_tiff_gives_clear_error(tmp_path, channels):
    img_path = save(tmp_path / "x.tif", gradient(channels, np.uint16))
    mask_path = save(tmp_path / "m.png", scratch_mask())
    before = sha256_file(img_path)
    with pytest.raises(ValueError, match="8-bit"):
        bi.run(img_path, mask_path, tmp_path / "out")
    assert sha256_file(img_path) == before


def test_auto_mask_alpha_ok_and_16bit_error():
    assert auto_mask.auto_mask(gradient(4)).shape == (64, 64)
    assert auto_mask.overlay(gradient(4), scratch_mask()).shape == (64, 64, 3)
    with pytest.raises(ValueError, match="8-bit"):
        auto_mask.auto_mask(gradient(3, np.uint16))


# --- ingest with the same files ---------------------------------------------------------

def test_ingest_accepts_alpha_png_and_16bit_tiff(tmp_path):
    png = save(tmp_path / "a.png", gradient(4))
    tif = save(tmp_path / "x.tif", gradient(3, np.uint16))
    hashes = [sha256_file(png), sha256_file(tif)]
    assert ingest(png, tmp_path / "s").channels == 4
    assert ingest(tif, tmp_path / "s").channels == 3
    assert [sha256_file(png), sha256_file(tif)] == hashes


# --- Hebrew filenames --------------------------------------------------------------------

def test_hebrew_filename_through_whole_flow(tmp_path):
    img, mask = gradient(3), scratch_mask()
    img_path = save(tmp_path / "תמונה_סבא.png", img)
    mask_path = save(tmp_path / "מסכה.png", mask)
    before = sha256_file(img_path)
    r = ingest(img_path, tmp_path / "s")
    assert r.source_name == "תמונה_סבא.png" and r.original_path.is_file()
    bi.run(img_path, mask_path, tmp_path / "פלט")
    assert (tmp_path / "פלט" / "תמונה_סבא_telea.png").exists()
    assert auto_mask.auto_mask(bi.read_image(img_path)).shape == (64, 64)
    assert sha256_file(img_path) == before


def test_make_mask_cli_with_hebrew_names(tmp_path):
    marked = gradient(3)
    marked[10:20, 10:20] = (30, 30, 220)  # BGR red
    src = save(tmp_path / "מסומן.png", marked)
    dst = tmp_path / "מסכה.png"
    before = sha256_file(src)
    subprocess.run([sys.executable, str(ROOT / "experiments" / "make_mask.py"), str(src), str(dst)],
                   check=True, capture_output=True)
    mask = bi.read_image(dst, cv2.IMREAD_GRAYSCALE)
    assert (mask[10:20, 10:20] == 255).all() and (mask > 0).sum() == 100
    assert sha256_file(src) == before


def test_restore_cli_with_hebrew_filename(tmp_path):
    src = save(tmp_path / "תמונה.png", gradient(3))
    proc = subprocess.run([sys.executable, str(ROOT / "restore.py"), str(src), "--storage", str(tmp_path / "s")],
                          capture_output=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


# --- fake images ------------------------------------------------------------------------

def test_text_file_named_jpg(tmp_path):
    fake = tmp_path / "fake.jpg"
    fake.write_text("this is definitely not a jpeg")
    mask_path = save(tmp_path / "m.png", scratch_mask())
    before = sha256_file(fake)
    with pytest.raises(IngestError, match="corrupt"):
        ingest(fake, tmp_path / "s")
    with pytest.raises(ValueError, match="cannot read image"):
        bi.run(fake, mask_path, tmp_path / "out")
    assert sha256_file(fake) == before
    assert not (tmp_path / "s" / "originals").exists() or not any((tmp_path / "s" / "originals").iterdir())


def test_truncated_jpeg_is_not_silently_accepted(tmp_path):
    good = save(tmp_path / "g.jpg", gradient(3))
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(good.read_bytes()[:30])
    with pytest.raises(IngestError):
        ingest(cut, tmp_path / "s")
