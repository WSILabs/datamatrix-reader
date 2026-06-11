"""Learned DataMatrix detector (YOLOv8) + format gate.

`YoloDetector` wraps an exported ONNX model via onnxruntime (lean — no torch at runtime;
ONNX Runtime has a C API for the eventual port). `detect()` returns code boxes in native
pixel coordinates, confidence-desc. The model is trained on synthetic scenes
(tools/make_yolo_dataset + tools/train_yolo) to find DataMatrix codes and REJECT QR /
Aztec / cassette-mesh look-alikes (negatives in the training set).

`format_gate()` reads a tight crop with zxing across the 2D formats: a readable code
short-circuits the expensive repair, and a QR/Aztec is flagged so we don't waste the
DataMatrix-repair brute force on a code that isn't a DataMatrix.

The model file is optional — if it's absent the Reader falls back to the classical
proposer (locate.propose). Train + export with tools/train_yolo.py + tools/export_yolo.py.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
# 2D matrix formats zxing can read at the gate
_2D = (zxingcpp.BarcodeFormat.DataMatrix | zxingcpp.BarcodeFormat.QRCode
       | zxingcpp.BarcodeFormat.Aztec)

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "dm_yolo.onnx"


def _letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh),
                                                      interpolation=cv2.INTER_LINEAR)
    return canvas, r, left, top


def _nms(xyxy, scores, iou_thr=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        x0 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        y0 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        x1 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        y1 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
        a_i = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        a_r = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / (a_i + a_r - inter + 1e-9)
        idxs = rest[iou < iou_thr]
    return keep


class YoloDetector:
    """ONNX YOLOv8 DataMatrix detector. detect(image) -> [(cx, cy, size, conf), ...]."""

    def __init__(self, model_path=DEFAULT_MODEL, conf=0.30, size=640):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(model_path),
                                         providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.conf = conf
        self.size = size

    def detect(self, image):
        bgr = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        lb, r, padx, pady = _letterbox(bgr, self.size)
        x = (lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        out = self.sess.run(None, {self.iname: x})[0]      # (1, 5, N) for 1 class
        out = out[0].T                                     # (N, 5): cx,cy,w,h,score
        scores = out[:, 4]
        m = scores > self.conf
        out, scores = out[m], scores[m]
        if not len(out):
            return []
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        res = []
        for i in _nms(xyxy, scores):
            x0, y0, x1, y1 = xyxy[i]
            x0, x1 = (x0 - padx) / r, (x1 - padx) / r       # unmap to native
            y0, y1 = (y0 - pady) / r, (y1 - pady) / r
            res.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                        max(x1 - x0, y1 - y0), float(scores[i])))
        res.sort(key=lambda t: -t[3])
        return res


def format_gate(crop):
    """zxing-read a tight crop across 2D formats. Returns (payload, format_name, pos):
    pos is a (4,2) float32 array of the code's corners in `crop` coords (or None).
      (bytes,'DataMatrix',pos) -> decoded; (bytes,'QRCode'/'Aztec',pos) -> non-DM 2D code,
      skip the DataMatrix repair; (None, None, None) -> nothing read -> caller repairs."""
    res = zxingcpp.read_barcodes(np.ascontiguousarray(crop), formats=_2D)
    if not res:
        return None, None, None
    p = res[0].position
    quad = np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                     [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]],
                    np.float32)
    return res[0].bytes, res[0].format.name, quad
