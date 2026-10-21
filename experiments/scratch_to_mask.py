"""Turn the scratch mask of 'Bringing Old Photos Back to Life' into a damage mask for restore.py.

Usage: python experiments/scratch_to_mask.py <image> <scratch_mask.png> <out_mask.png>
The tool's mask is a few pixels off the original size, so it is resized (nearest) to the image.
Face boxes (YuNet, padded) are removed from the mask: a crack across a face stays, because
inpainting there would invent a face. Edit the result by hand with draw_mask.py --start.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_inpaint import read_image, write_png  # noqa: E402
from restoration.faces import PAD, detect_faces  # noqa: E402

DILATE_PX = 1


def scratch_to_mask(image, scratch):
    h, w = image.shape[:2]
    mask = cv2.resize((scratch > 127).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((2 * DILATE_PX + 1,) * 2, np.uint8))
    faces = detect_faces(image)
    for f in faces:
        px, py = int(f.w * PAD), int(f.h * PAD)
        mask[max(f.y - py, 0):f.y + f.h + py, max(f.x - px, 0):f.x + f.w + px] = 0
    return mask, len(faces)


if __name__ == "__main__":
    image = read_image(sys.argv[1], cv2.IMREAD_COLOR)
    scratch = read_image(sys.argv[2], cv2.IMREAD_GRAYSCALE)
    if image is None or scratch is None:
        raise SystemExit("cannot read image or scratch mask")
    mask, n_faces = scratch_to_mask(image, scratch)
    Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
    write_png(sys.argv[3], mask)
    print(f"{sys.argv[3]}: mask {100 * (mask > 0).mean():.2f}%, {n_faces} face(s) excluded")
