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
