"""Global generative restoration (Gemini image model) with the original faces pasted back.

The model sees the whole photo, so it may redraw anything. Two safeguards keep the result close
to the original: face regions are replaced by the original pixels (feathered), and the result is
compared with the original at low frequency (see restore.py) so moved or changed objects are flagged.
"""
import io
import os
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = "gemini-2.5-flash-image"   # "Nano Banana"; gemini-3-pro-image-preview is "Nano Banana Pro"
DEFAULT_PROMPT = (
    "Restore this scanned old photograph. Remove scratches, cracks, creases, stains, dust and "
    "fading, and recover natural contrast. Keep every person's face, identity, expression, hair, "
    "clothing and pose exactly as they are. Do not add, remove, move or redraw any person or "
    "object. Keep the same framing, composition and aspect ratio, and do not crop. If the photo "
    "is black and white or sepia, keep it that way: no colorization."
)


class GenerativeError(Exception):
    """The generative step failed (missing key or package, no image returned)."""


def load_api_key(env_path=".env", name="GEMINI_API_KEY"):
    """The key from the environment or from a KEY=value line in .env. Never logged."""
    key = os.environ.get(name)
    path = Path(env_path)
    if not key and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                key = line.split("=", 1)[1].strip().strip("\"'")
    if not key:
        raise GenerativeError(f"{name} not found: put it in {env_path} as {name}=... (the file is git-ignored)")
    return key


def gemini_restore(image, prompt=DEFAULT_PROMPT, model=DEFAULT_MODEL, api_key=None):
    """Send `image` (8-bit BGR) to Gemini and return the restored image as 8-bit BGR."""
    try:
        from google import genai
        from PIL import Image
    except ImportError as e:
        raise GenerativeError("needs the package google-genai (pip install google-genai)") from e
    try:   # corporate networks re-sign TLS; trust the OS certificate store instead of certifi only
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    api_key = api_key or load_api_key()
    ok, png = cv2.imencode(".png", image[..., :3] if image.ndim == 3 else image)
    if not ok:
        raise GenerativeError("cannot encode the image")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model, contents=[prompt, Image.open(io.BytesIO(png.tobytes()))])
    for part in response.parts or []:
        if getattr(part, "inline_data", None) and part.inline_data.data:
            out = cv2.imdecode(np.frombuffer(part.inline_data.data, np.uint8), cv2.IMREAD_COLOR)
            if out is not None:
                return out
    raise GenerativeError("the model returned no image (it may have refused or answered with text only)")


def match_size(original, result):
    """Resize `result` to the original's size. Returns (image, resized, shift_px).

    shift_px is the global translation between the two (phase correlation on gray images): the
    model may crop or shift slightly, and then pasting faces back would misalign.
    """
    h, w = original.shape[:2]
    resized = result.shape[:2] != (h, w)
    if resized:
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_AREA if result.shape[0] > h else cv2.INTER_CUBIC)
    gray = lambda i: cv2.GaussianBlur(cv2.cvtColor(i[..., :3], cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i, (0, 0), 2).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(gray(original), gray(result))
    return result, resized, float(np.hypot(dx, dy))


def paste_original_faces(original, result, faces, pad=0.25, feather=0.15):
    """Replace each face region of `result` with the original pixels, blended at the edge.

    faces: restoration.faces.Face boxes detected on the original. The region is an ellipse
    around the padded box, so hair, ears and neck belong to the face too.
    """
    out = result.astype(np.float32)
    orig = original.astype(np.float32)
    h, w = original.shape[:2]
    for f in faces:
        weight = np.zeros((h, w), np.float32)
        cx, cy = f.x + f.w / 2, f.y + f.h / 2
        axes = (max(1, int(f.w * (0.5 + pad))), max(1, int(f.h * (0.5 + pad))))
        cv2.ellipse(weight, (int(cx), int(cy)), axes, 0, 0, 360, 1.0, -1)
        sigma = max(1.0, feather * min(axes))
        weight = cv2.GaussianBlur(weight, (0, 0), sigma)
        weight = np.maximum(weight, (weight > 0.999).astype(np.float32))
        w3 = weight[..., None] if out.ndim == 3 else weight
        out = out * (1 - w3) + orig * w3
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
