"""Experiment B summary: objective metrics for every tool output, with empty manual-score columns.

Usage: python experiments/summarize_tools.py [--images DIR] [--tools DIR] [--baseline DIR] [--out CSV]
Reads <tools>/<variant>/<stem>.png (from the Colab notebook), <tools>/timings.csv and the OpenCV
baseline outputs from experiment A. Writes <tools>/results.csv.
Outputs of other sizes are resized back to the original (INTER_AREA) before comparing.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from baseline_inpaint import read_image

# variant folder in output_b -> (tool name in timings.csv)
VARIANTS = {
    "realesrgan": "realesrgan",
    "codeformer_w07": "codeformer",
    "bopbtl_scratch": "bopbtl",
    "bopbtl_scratch_face": "bopbtl",
}
STEMS = ["1_cracks", "2_stains", "3_tear", "4_blur", "5_faces"]
RAW_TOLERANCE = 8        # gray levels; a pixel counts as changed above this
LOWPASS_SIGMA = 2.0
LOWPASS_TOLERANCE = 12   # structural change: ignores fine sharpening, as in restoration/compare.py
COLUMNS = ["tool", "image", "out_size", "scale", "changed_pct", "structural_changed_pct",
           "mean_abs_diff", "seconds", "tool_note", "fidelity_1_5", "improvement_1_5",
           "face_changed_yes_no_unclear", "notes"]


def to_gray(image):
    if image.dtype == np.uint16:
        image = (image / 257).astype(np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32)


def metrics(original, result):
    """Return (scale, changed_pct, structural_changed_pct, mean_abs_diff) of result vs original."""
    h, w = original.shape[:2]
    scale = result.shape[1] / w
    if result.shape[:2] != (h, w):
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA)
    a, b = to_gray(original), to_gray(result)
    diff = np.abs(a - b)
    low = np.abs(cv2.GaussianBlur(a, (0, 0), LOWPASS_SIGMA) - cv2.GaussianBlur(b, (0, 0), LOWPASS_SIGMA))
    return (scale, 100 * float((diff > RAW_TOLERANCE).mean()),
            100 * float((low > LOWPASS_TOLERANCE).mean()), float(diff.mean()))


def read_timings(path):
    """Last successful row per (tool, image): (seconds, note)."""
    timings = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["ok"] == "True":
                timings[(row["tool"], row["image"])] = (row["seconds"], row["note"])
    return timings


def find_original(images_dir, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        p = Path(images_dir) / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def row_for(tool, stem, original, result_path, seconds, note):
    row = {"tool": tool, "image": stem, "seconds": seconds, "tool_note": note}
    result = read_image(result_path) if result_path and result_path.is_file() else None
    if result is None:
        row["notes"] = "not run"
        return row
    scale, changed, structural, mean_diff = metrics(original, result)
    row.update(out_size=f"{result.shape[1]}x{result.shape[0]}", scale=f"{scale:.2f}",
               changed_pct=f"{changed:.1f}", structural_changed_pct=f"{structural:.1f}",
               mean_abs_diff=f"{mean_diff:.2f}")
    return row


def summarize(images_dir, tools_dir, baseline_dir):
    tools_dir = Path(tools_dir)
    timings = read_timings(tools_dir / "timings.csv")
    rows = []
    for stem in STEMS:
        original_path = find_original(images_dir, stem)
        original = read_image(original_path) if original_path else None
        if original is None:
            print(f"skip (original missing): {stem}")
            continue
        for variant, tool in VARIANTS.items():
            seconds, note = timings.get((tool, stem), ("", ""))
            rows.append(row_for(variant, stem, original, tools_dir / variant / f"{stem}.png", seconds, note))
        baseline_csv = Path(baseline_dir) / "results.csv"
        base_seconds = {}
        if baseline_csv.is_file():
            with open(baseline_csv, newline="", encoding="utf-8-sig") as f:
                base_seconds = {(r["image"], r["mask"], r["method"]): r["seconds"] for r in csv.DictReader(f)}
        for (image, mask, method), seconds in sorted(base_seconds.items()):
            if image != stem:
                continue
            path = Path(baseline_dir) / stem / f"{stem}_{mask}_{method}.png"
            rows.append(row_for(f"opencv_{mask}_{method}", stem, original, path, seconds, ""))
    return rows


def write_csv(rows, out):
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default=here / "images")
    parser.add_argument("--tools", default=here / "output_b")
    parser.add_argument("--baseline", default=here / "output")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.tools) / "results.csv"
    rows = summarize(args.images, args.tools, args.baseline)
    write_csv(rows, out)
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
