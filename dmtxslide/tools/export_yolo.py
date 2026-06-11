"""Export the trained YOLO weights to ONNX and install it where the Reader looks
(src/datamatrix_reader/models/dm_yolo.onnx). Run after tools/train_yolo.py.

    .venv/bin/python -m tools.export_yolo [weights.pt]
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = "/tmp/dm_yolo_runs/dm_neg/weights/best.pt"
DST = Path("src/datamatrix_reader/models/dm_yolo.onnx")


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    onnx = YOLO(weights).export(format="onnx", imgsz=640, opset=12)
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, DST)
    print(f"exported {weights} -> {DST}  ({DST.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
