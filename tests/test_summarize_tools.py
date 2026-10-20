import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import summarize_tools as st  # noqa: E402


def photo():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (120, 160), dtype=np.uint8), (0, 0), 4)
    return cv2.cvtColor(cv2.normalize(img, None, 40, 215, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)


def test_identical_has_no_change():
    scale, changed, structural, mean_diff = st.metrics(photo(), photo())
    assert (scale, changed, structural, mean_diff) == (1.0, 0.0, 0.0, 0.0)


def test_upscaled_result_is_resized_back():
    img = photo()
    big = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
    scale, changed, structural, _ = st.metrics(img, big)
    assert scale == 4.0 and structural == 0.0


def test_real_change_is_counted():
    img = photo()
    other = img.copy()
    other[:60] = 255 - other[:60]
    _, changed, structural, _ = st.metrics(img, other)
    assert changed > 30 and structural > 30


def test_summarize_writes_rows_and_marks_missing(tmp_path):
    images, tools, base = tmp_path / "images", tmp_path / "output_b", tmp_path / "output"
    for d in (images, tools / "realesrgan", base):
        d.mkdir(parents=True)
    img = photo()
    cv2.imencode(".png", img)[1].tofile(str(images / "1_cracks.png"))
    cv2.imencode(".png", img)[1].tofile(str(tools / "realesrgan" / "1_cracks.png"))
    (tools / "timings.csv").write_text(
        "tool,image,seconds,ok,note\nrealesrgan,1_cracks,5.0,True,\n", encoding="utf-8")
    rows = st.summarize(images, tools, base)
    by_tool = {r["tool"]: r for r in rows}
    assert by_tool["realesrgan"]["changed_pct"] == "0.0" and by_tool["realesrgan"]["seconds"] == "5.0"
    assert by_tool["codeformer_w07"]["notes"] == "not run"
    out = tmp_path / "r.csv"
    st.write_csv(rows, out)
    with open(out, newline="", encoding="utf-8-sig") as f:
        assert next(csv.DictReader(f)).keys() >= {"fidelity_1_5", "improvement_1_5"}
