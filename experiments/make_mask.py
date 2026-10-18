"""Turn a red-marked copy of a photo into a black/white mask (white = damage).

Usage: python experiments/make_mask.py <marked.png> <mask.png>
Red is detected loosely: R > 180, G < 90, B < 90.
"""
import sys

import cv2
import numpy as np

from baseline_inpaint import read_image, write_png


def red_mask(marked_bgr):
    b, g, r = marked_bgr[..., 0], marked_bgr[..., 1], marked_bgr[..., 2]
    return (((r > 180) & (g < 90) & (b < 90)) * 255).astype(np.uint8)


if __name__ == "__main__":
    marked = read_image(sys.argv[1], cv2.IMREAD_COLOR)
    if marked is None:
        raise SystemExit(f"cannot read image: {sys.argv[1]}")
    mask = red_mask(marked)
    write_png(sys.argv[2], mask)
    print(f"mask area: {100 * (mask > 0).mean():.2f}%")
