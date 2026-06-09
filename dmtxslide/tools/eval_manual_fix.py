"""Experiment helper: run every PNG in corpus/wsi_labels/manual_L_fix/ through the
full Reader and report whether it now decodes (after you hand-repair the broken L
finder). Crops are named scan<NNN>.png; ground truth is matched by scan number.

    python -m tools.eval_manual_fix
"""
import csv
import re
from pathlib import Path

import cv2

from dmtxslide.reader import Reader


def main():
    corpus = Path("corpus/wsi_labels")
    # crops are named scanNNN.png, so GT must be keyed by scan number here; the
    # manual_L_fix scans are unique, but scan numbers are NOT globally unique
    # (scan_180 is two slides) — setdefault avoids a silent wrong-row overwrite.
    gt = {}
    for r in csv.DictReader((corpus / "labels.csv").open(newline="")):
        m = re.search(r"scan_(\d+)_", r["file"])
        if m:
            gt.setdefault(m.group(1), r["payload"].encode())
    rd = Reader()
    folder = corpus / "manual_L_fix" / "edited"   # the repaired copies
    files = sorted(folder.glob("*.png"))
    if not files:
        print(f"no edited crops in {folder} (paint + save in tools.paint_l first)")
        return
    decoded = 0
    for f in files:
        scan = re.search(r"scan(\d+)", f.stem).group(1)
        res = rd.read(cv2.imread(str(f)))
        truth = gt.get(scan)
        if res.payload == truth:
            print(f"  {f.name}: DECODED ✓  (stage {res.stage})")
            decoded += 1
        elif res.payload is not None:
            print(f"  {f.name}: !! WRONG decode (does NOT match label) !!")
        else:
            print(f"  {f.name}: still no decode")
    print(f"\n{decoded}/{len(files)} decoded")


if __name__ == "__main__":
    main()
