"""Turn the scratch mask of 'Bringing Old Photos Back to Life' into a damage mask for restore.py.

Usage: python experiments/scratch_to_mask.py <image> <scratch_mask.png> <out_mask.png>
           [--dilate N] [--keep-faces | --narrow-faces PX] [--union OTHER_MASK.png]
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


def scratch_to_mask(image, scratch, dilate_px=DILATE_PX, keep_faces=False, narrow_face_px=0, user=None):
    """Damage mask from the scratch mask.

    keep_faces=True leaves the mask on faces too (looser, riskier; restore.py then says review).
    narrow_face_px>0: on faces keep only cracks narrower than that many pixels, and never on
    eyes, nose or mouth. `user` is an extra hand-drawn mask, applied outside faces only in that mode.
    """
    h, w = image.shape[:2]
    raw = cv2.resize((scratch > 127).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(raw, np.ones((2 * dilate_px + 1,) * 2, np.uint8))
    if user is not None:
        mask = np.maximum(mask, np.where(user > 0, 255, 0).astype(np.uint8))
    faces = detect_faces(image)
    if narrow_face_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (narrow_face_px, narrow_face_px))
        thin = cv2.dilate(cv2.subtract(raw, cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)), np.ones((3, 3), np.uint8))
        for f in faces:
            box = (slice(max(f.y, 0), f.y + f.h), slice(max(f.x, 0), f.x + f.w))
            allowed = thin[box].copy()
            if f.landmarks:
                feat = np.zeros_like(allowed)
                ox, oy = max(f.x, 0), max(f.y, 0)
                pts = [(int(x - ox), int(y - oy)) for x, y in f.landmarks]
                r = max(2, int(0.14 * f.w))
                for p in pts[:3]:                         # eyes and nose tip
                    cv2.circle(feat, p, r, 255, -1)
                cv2.line(feat, pts[3], pts[4], 255, max(2, int(0.18 * f.w)))   # mouth
                allowed[feat > 0] = 0
            mask[box] = allowed
        return mask, len(faces)
    for f in (() if keep_faces else faces):
        px, py = int(f.w * PAD), int(f.h * PAD)
        mask[max(f.y - py, 0):f.y + f.h + py, max(f.x - px, 0):f.x + f.w + px] = 0
    return mask, len(faces)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("scratch")
    p.add_argument("out")
    p.add_argument("--dilate", type=int, default=DILATE_PX)
    p.add_argument("--keep-faces", action="store_true")
    p.add_argument("--narrow-faces", type=int, default=0, metavar="PX",
                   help="on faces keep only cracks narrower than PX pixels, away from eyes/nose/mouth")
    p.add_argument("--union", help="also add this mask (e.g. your hand-edited one)")
    a = p.parse_args()
    image = read_image(a.image, cv2.IMREAD_COLOR)
    scratch = read_image(a.scratch, cv2.IMREAD_GRAYSCALE)
    if image is None or scratch is None:
        raise SystemExit("cannot read image or scratch mask")
    user = None
    if a.union:
        user = read_image(a.union, cv2.IMREAD_GRAYSCALE)
        if user is None or user.shape != image.shape[:2]:
            raise SystemExit(f"cannot use --union mask: {a.union}")
    mask, n_faces = scratch_to_mask(image, scratch, a.dilate, a.keep_faces, a.narrow_faces, user)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_png(a.out, mask)
    print(f"{a.out}: mask {100 * (mask > 0).mean():.2f}%, {n_faces} face(s) {'narrow cracks only' if a.narrow_faces else 'kept' if a.keep_faces else 'excluded'}")
