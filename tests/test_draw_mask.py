import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from draw_mask import Editor  # noqa: E402


def editor():
    return Editor(np.full((200, 300, 3), 128, np.uint8), view=(300, 200))


def test_paint_lands_on_image_coordinates_and_erase_removes_it():
    ed = editor()
    ed.begin_stroke()
    ed.paint(100, 50)
    assert ed.mask[50, 100] == 255 and ed.mask[150, 250] == 0
    ed.begin_stroke()
    ed.paint(100, 50, erase=True)
    assert not ed.mask.any()


def test_stroke_is_continuous_and_undo_restores():
    ed = editor()
    ed.begin_stroke()
    ed.paint(10, 100)
    ed.paint(200, 100)
    assert ed.mask[100, 100] == 255
    ed.undo()
    assert not ed.mask.any()


def test_zoom_keeps_point_under_cursor_and_paint_follows_view():
    ed = editor()
    before = ed.to_image(150, 100)
    ed.zoom_at(150, 100, 2.0)
    assert np.allclose(ed.to_image(150, 100), before)
    ed.pan(30, 0)
    ed.begin_stroke()
    ed.paint(150, 100)
    x, y = (int(round(v)) for v in ed.to_image(150, 100))
    assert ed.mask[y, x] == 255


def test_render_shows_mask_in_red_and_keeps_view_size():
    ed = editor()
    ed.begin_stroke()
    ed.paint(150, 100)
    view = ed.render()
    assert view.shape == (200, 300, 3)
    b, g, r = view[100, 150]
    assert r > b and r > g
