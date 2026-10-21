import json

import cv2
import numpy as np
import pytest

import restore
from restoration import generative as g
from restoration.faces import Face


def smooth(seed=0, size=(160, 200)):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 256, (*size, 3), dtype=np.uint8), (0, 0), 8)
    return cv2.normalize(img, None, 40, 215, cv2.NORM_MINMAX)


def test_paste_original_faces_keeps_face_core_exact_and_leaves_the_rest():
    original, result = smooth(1), smooth(2)
    face = Face(70, 50, 60, 60, 0.9)
    out = g.paste_original_faces(original, result, [face])
    assert np.array_equal(out[70:90, 90:110], original[70:90, 90:110])   # core is the original
    assert np.array_equal(out[:20, :20], result[:20, :20])               # far away is the model output


def test_match_size_resizes_and_reports_shift():
    original = smooth(3)
    same, resized, shift = g.match_size(original, original.copy())
    assert not resized and shift < 0.5
    big, resized, _ = g.match_size(original, cv2.resize(original, None, fx=2, fy=2))
    assert resized and big.shape == original.shape
    _, _, shift = g.match_size(original, np.roll(original, 8, axis=1))
    assert shift > 5


def test_api_key_comes_from_env_file_and_is_required(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(g.GenerativeError, match="GEMINI_API_KEY"):
        g.load_api_key(tmp_path / ".env")
    (tmp_path / ".env").write_text("OTHER=1\nGEMINI_API_KEY='abc123'\n", encoding="utf-8")
    assert g.load_api_key(tmp_path / ".env") == "abc123"


def run_cli(tmp_path, fake):
    src = tmp_path / "photo.png"
    cv2.imencode(".png", smooth(4, (200, 160)))[1].tofile(str(src))
    restore.gemini_restore = fake
    code = restore.main([str(src), "--engine", "gemini", "--storage", str(tmp_path / "s")])
    report = next((tmp_path / "s" / "reports").glob("*-restored.json"))
    return code, json.loads(report.read_text(encoding="utf-8"))


def test_gemini_engine_flags_a_changed_scene_and_missing_faces(tmp_path, monkeypatch):
    monkeypatch.setattr(restore, "gemini_restore", restore.gemini_restore)   # restored after the test
    code, report = run_cli(tmp_path, lambda image, *a, **k: smooth(99, image.shape[:2]))
    assert code == 0 and report["status"] == "review" and report["engine"] == "gemini"
    assert any("structure SSIM" in r for r in report["reasons"])
    assert any("no face detected" in r for r in report["reasons"])
    assert report["sent_to_external_service"] == "Google Gemini API"


def test_gemini_failure_marks_the_job_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(restore, "gemini_restore", restore.gemini_restore)

    def boom(*a, **k):
        raise g.GenerativeError("no key")

    restore.gemini_restore = boom
    src = tmp_path / "photo.png"
    cv2.imencode(".png", smooth(5, (200, 160)))[1].tofile(str(src))
    assert restore.main([str(src), "--engine", "gemini", "--storage", str(tmp_path / "s")]) == 1
