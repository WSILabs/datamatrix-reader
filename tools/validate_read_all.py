"""read_all must not lose any DataMatrix vs read(): on the WSI corpus every label has one
DataMatrix; confirm read_all finds the GT payload in all of them.

    .venv/bin/python -m tools.validate_read_all
"""
import csv
import glob
import re
from pathlib import Path

import cv2

from datamatrix_reader.reader import Reader

CORPUS = Path("corpus/wsi_labels")


def main():
    gt = {r["file"]: r["payload"].encode()
          for r in csv.DictReader((CORPUS / "labels.csv").open(newline=""))}
    rd = Reader()
    ok = wrong = miss = 0
    for f in sorted(CORPUS.glob("*.png")):
        truth = gt.get(f.name)
        res = rd.read_all(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
        pls = res.payloads
        if truth in pls:
            ok += 1
        elif pls:
            wrong += 1
            print(f"  {f.name}: GT not in {pls}")
        else:
            miss += 1
    n = ok + wrong + miss
    print(f"read_all: {ok}/{n} labels' GT DataMatrix found; missed={miss}, GT-absent-but-decoded={wrong}")


if __name__ == "__main__":
    main()
