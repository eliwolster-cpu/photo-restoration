"""Feasibility run: OpenCV inpainting on the experiment images, {manual, auto} mask x {TELEA, NS}.

Usage: python experiments/run_feasibility.py [--images DIR] [--out DIR]
An image N_name.ext may have a red-marked copy N_name_marked.(jpg|png); it gives the manual mask.
Images without a marked copy are run with the automatic mask only.
Writes <out>/<stem>/ outputs and <out>/results.csv with empty manual-score columns.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2

from auto_mask import auto_mask, iou
from baseline_inpaint import METHODS, inpaint, read_image, write_png
from make_mask import red_mask

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
COLUMNS = ["image", "mask", "method", "mask_area_pct", "iou_auto_vs_manual", "seconds",
           "fidelity_1_5", "improvement_1_5", "notes"]


def find_pairs(images_dir):
    """Return [(original_path, marked_path_or_None)] for top-level images named like 1_cracks.jpg."""
    files = [p for p in sorted(Path(images_dir).iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    originals = [p for p in files if not p.stem.endswith("_marked")]
    marked = {p.stem: p for p in files if p.stem.endswith("_marked")}
    return [(p, marked.get(f"{p.stem}_marked")) for p in originals]


def run(images_dir, out_dir):
    out_dir = Path(out_dir)
    rows = []
    for path, marked_path in find_pairs(images_dir):
        image = read_image(path)
        if image is None:
            print(f"skip (unreadable): {path.name}")
            continue
        folder = out_dir / path.stem
        folder.mkdir(parents=True, exist_ok=True)
        masks = {"auto": auto_mask(image)}
        write_png(folder / f"{path.stem}_auto_mask.png", masks["auto"])
        if marked_path is not None:
            marked = read_image(marked_path, cv2.IMREAD_COLOR)
            if marked is None or marked.shape[:2] != image.shape[:2]:
                print(f"skip manual mask (unreadable or size mismatch): {marked_path.name}")
            else:
                masks["manual"] = red_mask(marked)
                write_png(folder / f"{path.stem}_manual_mask.png", masks["manual"])
        iou_value = f"{iou(masks['auto'], masks['manual']):.3f}" if "manual" in masks else ""
        for mask_name, mask in masks.items():
            for method in METHODS:
                start = time.perf_counter()
                result = inpaint(image, mask, method)
                seconds = time.perf_counter() - start
                write_png(folder / f"{path.stem}_{mask_name}_{method}.png", result)
                rows.append([path.stem, mask_name, method, f"{100 * (mask > 0).mean():.2f}",
                             iou_value, f"{seconds:.2f}", "", "", ""])
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", default="experiments/images")
    p.add_argument("--out", default="experiments/output")
    a = p.parse_args()
    rows = run(a.images, a.out)
    print(f"{len(rows)} rows written to {Path(a.out) / 'results.csv'}")
