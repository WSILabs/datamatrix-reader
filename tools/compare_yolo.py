"""Head-to-head of the two trained detectors (dm_neg = neutral palette/distractors,
dm_real = realistic dark-gray palette + colorized distractors) on the data that matters:
real WSI residual codes, a synthetic look-alike rejection battery, and the real cassette
mesh photos. Synthetic mAP can't separate them; this can.

    .venv/bin/python -m tools.compare_yolo
"""
import csv
import glob
import re

import cv2
import numpy as np
import zxingcpp
from PIL import Image
from ultralytics import YOLO

import datamatrix_reader.register as R

MODELS = {"dm_neg": "/tmp/dm_yolo_runs/dm_neg/weights/best.pt",
          "dm_real": "/tmp/dm_yolo_runs/dm_real/weights/best.pt"}
_DM = zxingcpp.BarcodeFormat.DataMatrix


def _gt():
    g = {}
    for r in csv.DictReader(open("corpus/wsi_labels/labels.csv", newline="")):
        m = re.search(r"scan_(\d+)_", r["file"])
        if m:
            g.setdefault(m.group(1), r["payload"].encode())
    return g


def _canvas(tile, sub=(245, 245, 245)):
    H, W = 700, 900
    c = np.full((H, W, 3), sub, np.uint8)
    th, tw = tile.shape[:2]
    c[250:250 + th, 350:350 + tw] = tile
    return c


def _dm(sub, ink):
    a = np.asarray(zxingcpp.create_barcode(b"1-S-25-00828 A8-1", _DM).to_image())
    t = cv2.resize(a, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST).astype(np.float32)[..., None] / 255
    return ((t * (np.array(sub) / 255) + (1 - t) * (np.array(ink) / 255)) * 255).astype(np.uint8)


def _mesh(sub, ink):
    cell, n = 10, 22
    t = np.full((n * cell, n * cell, 3), sub, np.uint8)
    for i in range(n):
        for j in range(n):
            cv2.circle(t, (int((j + .5) * cell), int((i + .5) * cell)), 3, tuple(int(x) for x in ink), -1)
    return t


def _code(fmt):
    a = np.asarray(zxingcpp.create_barcode("https://example.com/123", fmt).to_image())
    t = cv2.resize(a, None, fx=12, fy=12, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)


def _load(p):
    g = cv2.imread(p)
    return g if g is not None else cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    gt = _gt()
    files = glob.glob("corpus/wsi_labels/*.png")
    hard = ["329", "330", "331", "342", "394", "432", "96"]
    battery = {
        "DM neutral (want DETECT)": _canvas(_dm((245, 245, 245), (40, 40, 40))),
        "DM colored (want DETECT)": _canvas(_dm((60, 230, 255), (20, 20, 20)), (60, 230, 255)),
        "mesh neutral (want reject)": _canvas(_mesh((225, 225, 225), (90, 90, 90))),
        "mesh COLORED (want reject)": _canvas(_mesh((60, 230, 255), (20, 20, 20)), (60, 230, 255)),
        "QR (want reject)": _canvas(_code(zxingcpp.BarcodeFormat.QRCode)),
        "Aztec (want reject)": _canvas(_code(zxingcpp.BarcodeFormat.Aztec)),
    }
    cassette = ["22106_angle_EAI-0302-10A.jpg",
                "csm_IP_C_IP_S_Slides_1_16_a517685468.jpg.webp"]

    for name, path in MODELS.items():
        m = YOLO(path)
        print(f"\n========== {name} ==========")
        # WSI 7
        ok = 0
        nboxes = []
        for h in hard:
            f = [x for x in files if re.search(rf"scan_{h}_", x)][0]
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            r = m.predict(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), imgsz=640, conf=0.30, device="mps", verbose=False)[0]
            n = len(r.boxes)
            nboxes.append(n)
            if n:
                xyxy = r.boxes.xyxy.cpu().numpy()
                i = r.boxes.conf.cpu().numpy().argmax()
                x0, y0, x1, y1 = xyxy[i]
                up, _ = R._normalize(g, (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0))
                if up is not None and R.decode_auto(up)[0] == gt[h]:
                    ok += 1
        print(f"  WSI residual: {ok}/7 decoded   | boxes/label: {nboxes} (fewer extra = cleaner)")
        # rejection battery
        for label, img in battery.items():
            r = m.predict(img, imgsz=640, conf=0.30, device="mps", verbose=False)[0]
            n = len(r.boxes)
            cf = [round(float(c), 2) for c in (r.boxes.conf.cpu().numpy() if n else [])]
            print(f"  {label:28} -> {'DETECT' if n else 'reject':6} ({n} box, conf={cf})")
        # real cassette mesh photos
        for c in cassette:
            r = m.predict(_load("corpus/pathology_samples/" + c), imgsz=640, conf=0.30, device="mps", verbose=False)[0]
            n = len(r.boxes)
            cf = [round(float(x), 2) for x in (r.boxes.conf.cpu().numpy() if n else [])]
            print(f"  cassette {c[:30]:30} -> {n} box conf={cf}")


if __name__ == "__main__":
    main()
