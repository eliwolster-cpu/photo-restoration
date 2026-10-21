"""Draw a damage mask on a photo (white = damage) in an OpenCV window.

Usage: python experiments/draw_mask.py <image> [--out experiments/masks/<stem>_mask.png]
If the output file already exists it is loaded, so you can continue where you stopped.

Mouse:  left drag = paint   right drag = erase   wheel = zoom   middle drag = pan
Keys:   [ / ] smaller / bigger brush    z undo    arrow keys = pan
        S save    Q save and quit    Esc quit without saving
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from baseline_inpaint import read_image, write_png

VIEW_W, VIEW_H = 1100, 800
MAX_UNDO = 30
KEY_PAN = {2424832: (-1, 0), 2555904: (1, 0), 2490368: (0, -1), 2621440: (0, 1)}  # arrows on Windows


class Editor:
    """Mask painting state and rendering; no window code, so it can be tested."""

    def __init__(self, image, mask=None, view=(VIEW_W, VIEW_H)):
        self.image = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.image = self.image[..., :3]
        h, w = self.image.shape[:2]
        self.mask = np.zeros((h, w), np.uint8) if mask is None else mask.copy()
        self.vw, self.vh = view
        self.zoom = min(self.vw / w, self.vh / h)
        self.ox = self.oy = 0.0          # image coordinates of the view's top-left corner
        self.brush = max(2, round(min(h, w) / 150))   # radius in image pixels
        self.history = []
        self.last = None
        self.cursor = None

    def to_image(self, vx, vy):
        return self.ox + vx / self.zoom, self.oy + vy / self.zoom

    def begin_stroke(self):
        self.history.append(self.mask.copy())
        del self.history[:-MAX_UNDO]
        self.last = None

    def paint(self, vx, vy, erase=False):
        x, y = (int(round(v)) for v in self.to_image(vx, vy))
        value = 0 if erase else 255
        if self.last is not None:
            cv2.line(self.mask, self.last, (x, y), value, 2 * self.brush)
        cv2.circle(self.mask, (x, y), self.brush, value, -1)
        self.last = (x, y)

    def undo(self):
        if self.history:
            self.mask = self.history.pop()

    def zoom_at(self, vx, vy, factor):
        ix, iy = self.to_image(vx, vy)
        self.zoom = float(np.clip(self.zoom * factor, 0.1, 40))
        self.ox, self.oy = ix - vx / self.zoom, iy - vy / self.zoom

    def pan(self, dvx, dvy):
        self.ox -= dvx / self.zoom
        self.oy -= dvy / self.zoom

    def resize_brush(self, factor):
        self.brush = max(1, round(self.brush * factor) if factor > 1 else int(self.brush * factor))

    def render(self):
        m = np.float32([[self.zoom, 0, -self.ox * self.zoom], [0, self.zoom, -self.oy * self.zoom]])
        interp = cv2.INTER_NEAREST if self.zoom >= 4 else cv2.INTER_AREA if self.zoom < 1 else cv2.INTER_LINEAR
        size = (self.vw, self.vh)
        view = cv2.warpAffine(self.image, m, size, flags=interp, borderValue=(40, 40, 40))
        mview = cv2.warpAffine(self.mask, m, size, flags=cv2.INTER_NEAREST) > 0
        view[mview] = (0.5 * view[mview] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
        if self.cursor is not None:
            cv2.circle(view, self.cursor, max(1, round(self.brush * self.zoom)), (0, 255, 255), 1)
        text = f"brush {self.brush}px  zoom {self.zoom:.2f}  mask {100 * (self.mask > 0).mean():.1f}%   [S]ave  [Q]save+quit  [Esc]quit"
        cv2.putText(view, text, (8, self.vh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(view, text, (8, self.vh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return view


def run(image_path, out_path, start=None):
    image = read_image(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image: {image_path}")
    out_path = Path(out_path)
    mask = None
    source = out_path if out_path.is_file() else start
    if source:
        mask = read_image(source, cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != image.shape[:2]:
            raise SystemExit(f"mask does not match the image: {source}")
        print(f"loaded mask: {source}")
    ed = Editor(image, mask)
    state = {"drag": None}
    title = f"draw mask - {Path(image_path).name}"

    def on_mouse(event, x, y, flags, _):
        ed.cursor = (x, y)
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            ed.begin_stroke()
            state["drag"] = "erase" if event == cv2.EVENT_RBUTTONDOWN else "paint"
            ed.paint(x, y, state["drag"] == "erase")
        elif event == cv2.EVENT_MBUTTONDOWN:
            state["drag"], state["pos"] = "pan", (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            if state["drag"] == "pan":
                ed.pan(x - state["pos"][0], y - state["pos"][1])
                state["pos"] = (x, y)
            else:
                ed.paint(x, y, state["drag"] == "erase")
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            state["drag"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            ed.zoom_at(x, y, 1.25 if flags > 0 else 0.8)

    def save():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_png(out_path, ed.mask)
        print(f"saved {out_path} ({100 * (ed.mask > 0).mean():.2f}% of the image)")

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    while True:
        cv2.imshow(title, ed.render())
        key = cv2.waitKeyEx(15)
        if key == -1:
            continue
        if key in KEY_PAN:
            dx, dy = KEY_PAN[key]
            ed.pan(-dx * 60, -dy * 60)
        c = chr(key & 0xFF).lower() if 0 <= key < 256 else ""
        if c == "[":
            ed.resize_brush(0.8)
        elif c == "]":
            ed.resize_brush(1.25)
        elif c == "z":
            ed.undo()
        elif c == "s":
            save()
        elif c == "q":
            save()
            break
        elif key == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--out")
    p.add_argument("--start", help="mask to start from (e.g. an automatic proposal), used if --out does not exist yet")
    a = p.parse_args()
    run(a.image, a.out or f"experiments/masks/{Path(a.image).stem}_mask.png", a.start)
