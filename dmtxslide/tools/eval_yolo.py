"""Evaluate the trained YOLO DataMatrix detector against the classical proposer.

For the 7 WSI residual labels: run YOLO, then for each detected box (confidence-desc)
crop+normalize+decode. Reports box count (decoys → >1), detection latency, decode time,
and whether it decodes — the head-to-head vs the classical propose+brute path.

    .venv/bin/python -m tools.eval_yolo
"""
import csv
import glob
import re
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import dmtxslide.register as R

WEIGHTS = "/tmp/dm_yolo_runs/dm/weights/best.pt"


def _gt():
    gt = {}
    for r in csv.DictReader(open("corpus/wsi_labels/labels.csv", newline="")):
        m = re.search(r"scan_(\d+)_", r["file"])
        if m:
            gt.setdefault(m.group(1), r["payload"].encode())
    return gt


def main():
    model = YOLO(WEIGHTS)
    gt = _gt()
    files = glob.glob("corpus/wsi_labels/*.png")
    hard = ["329", "330", "331", "342", "394", "432", "96"]
    ok = 0
    for h in hard:
        f = [x for x in files if re.search(rf"scan_{h}_", x)][0]
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        t = time.perf_counter()
        res = model.predict(bgr, imgsz=640, conf=0.25, device="mps", verbose=False)[0]
        det_ms = (time.perf_counter() - t) * 1000
        b = res.boxes
        n = len(b)
        conf = b.conf.cpu().numpy() if n else np.array([])
        xyxy = b.xyxy.cpu().numpy() if n else np.zeros((0, 4))
        decoded, decidx = False, -1
        t2 = time.perf_counter()
        for rank, i in enumerate(np.argsort(-conf)):
            x0, y0, x1, y1 = xyxy[i]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            size = max(x1 - x0, y1 - y0)
            up, _tf = R._normalize(g, cx, cy, size)
            if up is None:
                continue
            if R.decode_auto(up)[0] == gt[h]:
                decoded, decidx = True, rank
                break
        dec_s = time.perf_counter() - t2
        ok += decoded
        print(f"scan{h}: {n} box(es) det={det_ms:.0f}ms  decode={dec_s:.1f}s  "
              f"-> {'OK @box#' + str(decidx) if decoded else 'miss'}")
    print(f"\n{ok}/7 decoded via YOLO boxes")


if __name__ == "__main__":
    main()
