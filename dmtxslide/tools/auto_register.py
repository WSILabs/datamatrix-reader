"""Validation harness for datamatrix_reader.register on the 7 cascade-residual WSI crops
(corpus/wsi_labels/manual_L_fix/*.png). The registration logic lives in the package
(src/datamatrix_reader/register.py); this just runs decode_auto on each crop and reports the
recovered grid params vs the human stamp-tool alignments.

    .venv/bin/python -m tools.auto_register
"""
import csv
import json
import re
import time
from pathlib import Path

import cv2

from datamatrix_reader.register import decode_auto, detect_area

CORPUS = Path("corpus/wsi_labels")
FOLDER = CORPUS / "manual_L_fix"


def main():
    # the 7 manual_L_fix crops are named scanNNN.png and have unique scan numbers, so
    # keying GT by scan number is safe HERE (corpus-wide tools must key by filename —
    # scan numbers are not globally unique, e.g. scan_180 is two slides).
    gt = {}
    for r in csv.DictReader((CORPUS / "labels.csv").open(newline="")):
        m = re.search(r"scan_(\d+)_", r["file"])
        if m:
            gt.setdefault(m.group(1), r["payload"].encode())
    human = json.loads((FOLDER / "stamp_params.json").read_text()) \
        if (FOLDER / "stamp_params.json").exists() else {}

    ok = 0
    files = sorted(FOLDER.glob("*.png"))
    for f in files:
        scan = re.search(r"scan(\d+)", f.stem).group(1)
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        payload, params = decode_auto(g)
        dt = time.perf_counter() - t0
        truth = gt.get(scan)
        verdict = "DECODED ✓" if payload == truth else ("WRONG" if payload else "no decode")
        if payload == truth:
            ok += 1
        h = human.get(scan, {})
        hs = f" | human M={h['M']} cell={h['cell']}" if h else ""
        print(f"scan{scan}: {verdict:10} ({dt:4.1f}s){hs}")
        if params:
            print(f"         -> {params}")
    print(f"\n{ok}/{len(files)} auto-decoded")


if __name__ == "__main__":
    main()
