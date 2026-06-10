"""Regression guard for the 28 decodable corpus/pathology_samples (pseudo-GT = the value
the current Reader decodes). Run before/after register changes; the count must not drop.

    .venv/bin/python -m tools.validate_pathology
"""
import glob
import os
import numpy as np
import cv2
from PIL import Image
from dmtxslide.reader import Reader


def load(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return g if g is not None else np.array(Image.open(p).convert("L"))


def main():
    rd = Reader()
    ok = 0
    files = sorted(glob.glob("corpus/pathology_samples/*"))
    for p in files:
        g = load(p)
        if g is None:
            continue
        if rd.read(g).payload is not None:
            ok += 1
    print(f"pathology_samples decoded: {ok}/{len(files)}  (baseline 28; must not drop)")


if __name__ == "__main__":
    main()
