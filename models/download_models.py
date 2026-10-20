"""Download the YuNet face detector (MIT license, opencv_zoo) and verify its hash."""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
TARGET = Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"


def main():
    if TARGET.is_file() and hashlib.sha256(TARGET.read_bytes()).hexdigest() == SHA256:
        print(f"already present: {TARGET}")
        return 0
    data = urllib.request.urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        print("error: downloaded file does not match the expected hash", file=sys.stderr)
        return 1
    TARGET.write_bytes(data)
    print(f"saved: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
