"""Ingest: validate an input image, hash it, and store a read-only copy of the original."""
import hashlib
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# Sanity limits; they move to config.yaml once the config module exists (plan step 5).
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_PIXELS = 100_000_000
MIN_SIDE = 32


class IngestError(Exception):
    """The input cannot be processed (missing, unsupported, corrupt or unreasonable)."""


@dataclass(frozen=True)
class IngestResult:
    job_id: str
    sha256: str
    source_name: str
    original_path: Path
    width: int
    height: int
    channels: int
    duplicate: bool  # True if this exact file was already ingested


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate(path):
    """Return (width, height, channels) or raise IngestError."""
    if not path.is_file():
        raise IngestError(f"file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"unsupported format '{path.suffix}' (supported: {sorted(SUPPORTED_EXTENSIONS)})")
    size = path.stat().st_size
    if size == 0:
        raise IngestError(f"file is empty: {path}")
    if size > MAX_FILE_BYTES:
        raise IngestError(f"file too large ({size} bytes, max {MAX_FILE_BYTES})")
    # np.fromfile + imdecode, because cv2.imread fails on non-ASCII Windows paths
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IngestError(f"file is corrupt or not a readable image: {path}")
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    if min(width, height) < MIN_SIDE:
        raise IngestError(f"image too small ({width}x{height}, min side {MIN_SIDE})")
    if width * height > MAX_PIXELS:
        raise IngestError(f"image too large ({width}x{height} pixels)")
    return width, height, channels


def _find_duplicate(originals_dir, sha256):
    for candidate in originals_dir.glob(f"*-{sha256[:8]}.*"):
        if sha256_file(candidate) == sha256:
            return candidate
    return None


def ingest(path, storage_root="storage", now=None):
    """Validate `path` and store a read-only copy under <storage_root>/originals/.

    The input file is only read. If the same content was ingested before, the existing
    result is returned with duplicate=True and nothing is written.
    """
    path = Path(path)
    width, height, channels = _validate(path)
    sha256 = sha256_file(path)
    originals = Path(storage_root) / "originals"
    originals.mkdir(parents=True, exist_ok=True)

    existing = _find_duplicate(originals, sha256)
    if existing is not None:
        job_id = existing.stem
        stored, duplicate = existing, True
    else:
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        job_id = f"{stamp}-{sha256[:8]}"
        stored = originals / f"{job_id}{path.suffix.lower()}"
        shutil.copyfile(path, stored)
        if sha256_file(stored) != sha256:
            stored.unlink()
            raise IngestError("copy of the original does not match its hash")
        os.chmod(stored, stat.S_IREAD)
        duplicate = False

    return IngestResult(job_id, sha256, path.name, stored, width, height, channels, duplicate)


def verify_original(result):
    """True if the stored original still matches the hash taken at ingest."""
    return result.original_path.is_file() and sha256_file(result.original_path) == result.sha256
