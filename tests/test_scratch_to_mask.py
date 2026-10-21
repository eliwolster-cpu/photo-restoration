import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from scratch_to_mask import scratch_to_mask  # noqa: E402


def test_scratch_mask_is_resized_to_the_image_and_binary():
    image = np.full((100, 80, 3), 128, np.uint8)
    scratch = np.zeros((104, 76), np.uint8)
    scratch[:, 40] = 255
    mask, faces = scratch_to_mask(image, scratch)
    assert mask.shape == (100, 80) and set(np.unique(mask)) <= {0, 255}
    assert mask[:, 38:46].any() and faces == 0


def test_narrow_mode_keeps_only_thin_cracks_away_from_features(monkeypatch):
    import scratch_to_mask as m
    from restoration.faces import Face

    face = Face(50, 50, 100, 100, 0.9, ((80, 80), (120, 80), (100, 100), (85, 125), (115, 125)))
    monkeypatch.setattr(m, "detect_faces", lambda image: [face])
    image = np.full((200, 200, 3), 128, np.uint8)
    scratch = np.zeros((200, 200), np.uint8)
    scratch[:, 60:62] = 255      # thin crack, outside the features
    scratch[55:70, 100:115] = 255  # thick blob on the forehead
    scratch[78:82, 55:145] = 255   # thin line through both eyes
    scratch[:, 20:40] = 255        # wide crack outside the face
    mask, n = m.scratch_to_mask(image, scratch, narrow_face_px=5)
    assert n == 1
    assert mask[100, 61] > 0                      # thin crack kept on the face
    assert not mask[58:68, 103:112].any()         # thick blob removed
    assert not mask[78:82, 75:85].any()           # eye zone untouched
    assert mask[100, 25] > 0                      # outside the face the loose mask applies
