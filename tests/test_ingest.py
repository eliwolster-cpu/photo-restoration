import os
from datetime import datetime

import cv2
import numpy as np
import pytest

from restoration.ingest import IngestError, ingest, sha256_file, verify_original
import restore


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "in" / "grandpa.jpg"
    path.parent.mkdir()
    img = np.random.default_rng(0).integers(0, 255, (80, 120, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_ingest_stores_readonly_copy_with_same_hash(photo, tmp_path):
    before = sha256_file(photo)
    r = ingest(photo, tmp_path / "storage", now=datetime(2026, 10, 18, 14, 22, 33))
    assert r.job_id == f"20261018-142233-{before[:8]}"
    assert (r.width, r.height, r.channels) == (120, 80, 3)
    assert not r.duplicate
    assert r.original_path.parent == tmp_path / "storage" / "originals"
    assert sha256_file(r.original_path) == before == r.sha256
    assert sha256_file(photo) == before  # input untouched
    with pytest.raises(PermissionError):
        open(r.original_path, "wb")
    assert verify_original(r)


def test_same_file_twice_is_duplicate(photo, tmp_path):
    first = ingest(photo, tmp_path / "storage")
    second = ingest(photo, tmp_path / "storage")
    assert second.duplicate and second.job_id == first.job_id
    assert len(list((tmp_path / "storage" / "originals").iterdir())) == 1


def test_verify_detects_changed_original(photo, tmp_path):
    r = ingest(photo, tmp_path / "storage")
    os.chmod(r.original_path, 0o666)
    r.original_path.write_bytes(b"tampered")
    assert not verify_original(r)


def test_grayscale_channels(tmp_path):
    p = tmp_path / "g.png"
    cv2.imwrite(str(p), np.full((64, 64), 128, np.uint8))
    assert ingest(p, tmp_path / "s").channels == 1


@pytest.mark.parametrize("name,content,msg", [
    ("missing.jpg", None, "not found"),
    ("empty.jpg", b"", "empty"),
    ("bad.jpg", b"not an image at all", "corrupt"),
    ("doc.txt", b"hello", "unsupported"),
])
def test_invalid_input_raises_clear_error(tmp_path, name, content, msg):
    p = tmp_path / name
    if content is not None:
        p.write_bytes(content)
    with pytest.raises(IngestError, match=msg):
        ingest(p, tmp_path / "s")
    assert not (tmp_path / "s" / "originals").exists() or not any((tmp_path / "s" / "originals").iterdir())


def test_tiny_image_rejected(tmp_path):
    p = tmp_path / "t.png"
    cv2.imwrite(str(p), np.zeros((8, 8), np.uint8))
    with pytest.raises(IngestError, match="too small"):
        ingest(p, tmp_path / "s")


def test_cli_exit_codes(photo, tmp_path, capsys):
    storage = str(tmp_path / "s")
    assert restore.main([str(photo), "--storage", storage]) == 0
    assert restore.main([str(photo), "--storage", storage]) == 0
    assert "duplicate" in capsys.readouterr().out
    assert restore.main([str(tmp_path / "nope.jpg"), "--storage", storage]) == 1
