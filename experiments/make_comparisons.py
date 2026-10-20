"""Side-by-side comparison image per experiment photo, for manual scoring.

Usage: python experiments/make_comparisons.py [--images DIR] [--tools DIR] [--baseline DIR] [--out DIR]
Each panel is resized to the original's size (INTER_AREA); missing outputs get a gray "not run" panel.
Writes <out>/<stem>.png, 3 panels per row. Labels are ASCII (cv2.putText has no Hebrew).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from baseline_inpaint import read_image, write_png
from summarize_tools import STEMS, find_original

LABEL_H = 28
COLS = 3


def panel(label, image, size):
    """image resized to size=(w, h) with a label bar above; None gives a gray 'not run' panel."""
    w, h = size
    if image is None:
        body = np.full((h, w, 3), 128, np.uint8)
        cv2.putText(body, "not run", (w // 2 - 45, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        if image.dtype == np.uint16:
            image = (image / 257).astype(np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        body = cv2.resize(image[:, :, :3], (w, h), interpolation=cv2.INTER_AREA)
    bar = np.full((LABEL_H, w, 3), 40, np.uint8)
    cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, body])


def compose(panels):
    rows = []
    for i in range(0, len(panels), COLS):
        row = panels[i:i + COLS]
        while len(row) < COLS:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def opencv_path(baseline_dir, stem):
    """Manual-mask TELEA if it exists (the fairest baseline), else automatic-mask TELEA."""
    for mask in ("manual", "auto"):
        p = Path(baseline_dir) / stem / f"{stem}_{mask}_telea.png"
        if p.is_file():
            return mask, p
    return None, None


def build(stem, images_dir, tools_dir, baseline_dir):
    original = read_image(find_original(images_dir, stem)) if find_original(images_dir, stem) else None
    if original is None:
        return None
    size = (original.shape[1], original.shape[0])
    mask, base = opencv_path(baseline_dir, stem)
    tools = Path(tools_dir)
    sources = [("original", original),
               (f"opencv {mask} mask, TELEA" if base else "opencv (none)", read_image(base) if base else None)]
    for label, variant in [("Real-ESRGAN x4 (shrunk)", "realesrgan"), ("CodeFormer w=0.7", "codeformer_w07"),
                           ("Bringing Old Photos", "bopbtl_scratch")]:
        p = tools / variant / f"{stem}.png"
        sources.append((label, read_image(p) if p.is_file() else None))
    return compose([panel(label, image, size) for label, image in sources])


def main(argv=None):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default=here / "images")
    parser.add_argument("--tools", default=here / "output_b")
    parser.add_argument("--baseline", default=here / "output")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else Path(args.tools) / "compare"
    out.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        image = build(stem, args.images, args.tools, args.baseline)
        if image is None:
            print(f"skip (original missing): {stem}")
            continue
        write_png(out / f"{stem}.png", image)
        print(f"wrote {out / (stem + '.png')} {image.shape[1]}x{image.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
