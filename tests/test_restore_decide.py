import numpy as np

import restore
from restoration.compare import OutsideMaskReport
from restoration.faces import Face, FaceCheck, FacesReport

CFG = {"max_mask_fraction": 0.10, "min_outside_ssim": 0.98,
       "max_outside_changed_fraction": 0.01, "min_face_ssim": 0.90}
OUTSIDE = OutsideMaskReport(False, 0.9, 0.0, 0.0, 0.0, 1.0)


def faces(box=Face(40, 40, 60, 60, 0.9), small=False):
    return FacesReport([FaceCheck(box, small, 1.0)], 1, 1)


def mask_with(y0, y1, x0, x1):
    m = np.zeros((200, 200), np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def test_mask_touching_a_face_is_never_done():
    status, reasons = restore._decide(CFG, 0.01, OUTSIDE, faces(), mask_with(50, 60, 50, 60))
    assert status == "review" and any("touches the face" in r for r in reasons)


def test_mask_away_from_faces_is_done():
    status, reasons = restore._decide(CFG, 0.01, OUTSIDE, faces(), mask_with(150, 160, 150, 160))
    assert status == "done" and reasons == []


def test_overlapping_faces_lists_only_touched_ones():
    f = FacesReport([FaceCheck(Face(40, 40, 60, 60, 0.9), False, 1.0),
                     FaceCheck(Face(120, 120, 30, 30, 0.9), False, 1.0)], 2, 2)
    assert restore._overlapping_faces(f, mask_with(50, 60, 50, 60)) == [0]
