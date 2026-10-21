"""Stage 7: mild global unsharp mask. Runs after cleaning and only if enabled in config."""
import cv2
import numpy as np


def sharpen(image, amount=0.5, sigma=1.0):
    """Unsharp mask on 8-bit images: image + amount * (image - blurred). Alpha is kept."""
    if image.dtype != np.uint8:
        raise ValueError(f"unsupported bit depth {image.dtype}; only 8-bit images are supported")
    if image.ndim == 3 and image.shape[2] == 4:
        out = image.copy()
        out[..., :3] = sharpen(image[..., :3], amount, sigma)
        return out
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
