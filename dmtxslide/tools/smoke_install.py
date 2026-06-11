"""Install smoke test: confirm a BUILT/INSTALLED copy ships the YOLO model, loads it, and
decodes — i.e. that package-data is wired correctly and the Reader isn't silently falling back
to the classical proposer. Uses a synthetic (PHI-free) code, so it's safe to run anywhere.

Run it against a CLEAN install to actually exercise packaging (not the editable dev tree):

    python -m build  # or: pip wheel . --no-deps -w dist/
    python -m venv /tmp/dmr_env && /tmp/dmr_env/bin/pip install "dist/datamatrix_reader-*.whl[yolo]"
    /tmp/dmr_env/bin/python tools/smoke_install.py

Exits non-zero on any failure.
"""
import sys

import cv2
import numpy as np
import zxingcpp

from datamatrix_reader.detect import DEFAULT_MODEL
from datamatrix_reader.register import _detector
from datamatrix_reader.reader import Reader

_DM = zxingcpp.BarcodeFormat.DataMatrix


def main():
    if not DEFAULT_MODEL.exists():
        sys.exit(f"FAIL: model not shipped with the install: {DEFAULT_MODEL}")
    if _detector() is None:
        sys.exit("FAIL: YoloDetector did not load (onnxruntime missing or model unreadable)")
    grid = np.asarray(zxingcpp.create_barcode("SMOKE-TEST-0001", _DM).to_image())
    img = cv2.resize(grid, None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST)
    r = Reader().read(img)
    if not (r.ok and r.payload == b"SMOKE-TEST-0001"):
        sys.exit(f"FAIL: decode mismatch: {r.payload!r}")
    print(f"OK: model shipped + YoloDetector loaded + decoded {r.payload!r} (stage {r.stage})")
    print(f"    model at {DEFAULT_MODEL}")


if __name__ == "__main__":
    main()
