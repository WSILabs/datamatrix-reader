"""Auto-label the real WSI corpus and add it to the YOLO training set, so the detector
learns what real (faint / low-contrast) codes look like — no manual boxing. The box for
each label comes from read()'s decoded quad (raw zxing.position / cascade / autoreg), so
every one of the 404 labels gets a tight ground-truth box for free.

The real IMAGES are PHI and stay local (/tmp dataset, gitignored); only the trained
detection weights ship — and a detector learns "a code is here", never the payload.

    .venv/bin/python -m tools.add_real_to_dataset
"""
import glob
from pathlib import Path

import cv2
import numpy as np

from dmtxslide.reader import Reader

ROOT = Path("/tmp/dm_yolo")
CORPUS = "corpus/wsi_labels"


def main():
    rd = Reader()
    (ROOT / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ROOT / "labels" / "train").mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    for i, f in enumerate(sorted(glob.glob(f"{CORPUS}/*.png"))):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        r = rd.read(g)
        if r.quad is None:                          # undecoded -> no trustworthy box
            skipped += 1
            continue
        H, W = g.shape
        xs, ys = r.quad[:, 0], r.quad[:, 1]
        cx, cy = (xs.min() + xs.max()) / 2 / W, (ys.min() + ys.max()) / 2 / H
        bw, bh = (xs.max() - xs.min()) / W, (ys.max() - ys.min()) / H
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < bw < 1 and 0 < bh < 1):
            skipped += 1
            continue
        name = f"real_{i:04d}"
        cv2.imwrite(str(ROOT / "images" / "train" / f"{name}.png"), g)
        (ROOT / "labels" / "train" / f"{name}.txt").write_text(
            "0 %.6f %.6f %.6f %.6f\n" % (cx, cy, bw, bh))
        added += 1
    print(f"added {added} real labels to {ROOT}/(images|labels)/train  (skipped {skipped})")


if __name__ == "__main__":
    main()
