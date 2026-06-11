"""Full-corpus validation of the Reader's finder-registration fallback on FULL labels
(not the pre-isolated crops). The Reader runs its cascade then, on a miss,
register.recover() (ROI-crop -> upscale -> 3-detector union + solid-side L test +
canonical border repaint, ECC-validated).

Reports: cascade correct, fallback-recovered, WRONG (must be 0), and per-label timing
(every fallback-triggered label is printed with its time; overall p50/p95/max).

    .venv/bin/python -m tools.validate_full
"""
import csv
import re
import time
from pathlib import Path

import cv2
import numpy as np

from datamatrix_reader.reader import Reader

CORPUS = Path("corpus/wsi_labels")


def main():
    # key GT by FILENAME — scan numbers are NOT unique (e.g. scan_180 is two slides)
    gt = {r["file"]: r["payload"].encode()
          for r in csv.DictReader((CORPUS / "labels.csv").open(newline=""))}

    reader = Reader()
    files = sorted(p for p in CORPUS.glob("*.png"))
    cascade_ok = recovered = wrong = no_decode = 0
    times = []
    n = len(files)
    print(f"validating {n} full labels (cascade + ROI fallback)...\n")
    for idx, f in enumerate(files, 1):
        m = re.search(r"scan_(\d+)_", f.name)
        if not m:
            continue
        scan = m.group(1)
        truth = gt.get(f.name)
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        res = reader.read(g)                          # cascade + autoreg fallback (built in)
        payload, dt = res.payload, res.elapsed_ms / 1000.0
        times.append(dt)

        if payload == truth:
            if res.stage == "autoreg":
                recovered += 1
                print(f"  [{idx}/{n}] scan{scan}: RECOVERED via fallback  ({dt:.1f}s)")
            else:
                cascade_ok += 1
        elif payload is not None:
            wrong += 1
            print(f"  [{idx}/{n}] scan{scan}: !! WRONG ({payload!r} != {truth!r}) "
                  f"[{res.stage}] ({dt:.1f}s)")
        else:
            no_decode += 1
            print(f"  [{idx}/{n}] scan{scan}: no decode ({dt:.1f}s)")
        if idx % 50 == 0:
            print(f"    ...{idx}/{n} processed")

    times = np.array(times)
    total_ok = cascade_ok + recovered
    print(f"\n=== {n} labels ===")
    print(f"cascade correct : {cascade_ok}")
    print(f"fallback recovered: {recovered}")
    print(f"still no decode : {no_decode}")
    print(f"WRONG           : {wrong}")
    print(f"TOTAL correct   : {total_ok}/{n} = {total_ok / n:.3f}")
    print(f"\ntiming/label: p50 {np.percentile(times,50)*1000:.0f}ms  "
          f"p95 {np.percentile(times,95)*1000:.0f}ms  max {times.max():.1f}s  "
          f"total {times.sum():.0f}s")


if __name__ == "__main__":
    main()
