"""2-fold decode comparison on the same images:
  1. zxing raw      -> plain gray -> zxing-cpp (single pass)
  2. zxing cascade  -> Reader: raw zxing, then upscale2x+CLAHE on a miss

Fold 2 minus fold 1 is the value of the poor-print fallback. Run on labelled
sets (synth/corpus) for correct rate; pathology has no truth, so report
decode-hits + cross-decoder agreement.

    python -m tools.compare_backends --synth --per-cell 1 --budget 250
    python -m tools.compare_backends --corpus corpus/wsi_labels --budget 250
    python -m tools.compare_backends --pathology corpus/pathology_samples
"""
from __future__ import annotations

import argparse
import csv
import statistics as stats
import time
from pathlib import Path

import cv2
import zxingcpp

from datamatrix_reader.reader import Reader
from bench.harness import _iter_synth, _payload_pool
from datamatrix_reader.synth import strata

_DM = zxingcpp.BarcodeFormat.DataMatrix
_reader = Reader()

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_corpus(root):
    """Load (truth_bytes, bgr_image) pairs from a labelled corpus.

    Accepts both layouts: <root>/images/* + <root>/labels.csv (BarBeR style),
    or a flat <root>/*.png + <root>/labels.csv (wsi_labels style). Only images
    with a labels.csv row are returned."""
    root = Path(root)
    labels = {}
    lp = root / "labels.csv"
    if lp.exists():
        with lp.open(newline="") as f:
            for row in csv.DictReader(f):
                labels[row["file"]] = row["payload"].encode()
    img_dir = root / "images"
    if not img_dir.is_dir():
        img_dir = root
    items = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None and p.name in labels:
            items.append((labels[p.name], img))
    return items


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def fold_zxing_raw(img, budget):
    res = zxingcpp.read_barcodes(_gray(img), formats=_DM)
    return res[0].bytes if res else None


def fold_zxing_cascade(img, budget):
    return _reader.read(img, budget_ms=budget).payload


FOLDS = [("zxing raw", fold_zxing_raw),
         ("zxing cascade", fold_zxing_cascade)]


def _pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def run_labelled(items, budget):
    n = len(items)
    res = {name: {"correct": 0, "t": []} for name, _ in FOLDS}
    for truth, img in items:
        for name, fn in FOLDS:
            t0 = time.perf_counter()
            got = fn(img, budget)
            res[name]["t"].append((time.perf_counter() - t0) * 1000)
            res[name]["correct"] += (got == truth)
    print(f"{'fold':<22}{'correct':>10}{'rate':>8}{'p50ms':>9}{'p95ms':>9}")
    for name, _ in FOLDS:
        c, ts = res[name]["correct"], res[name]["t"]
        print(f"{name:<22}{c:>6}/{n:<3}{c/n:>8.2f}{_pct(ts,.5):>9.1f}{_pct(ts,.95):>9.1f}")


def run_pathology(root, budget):
    items = []
    for p in sorted(Path(root).iterdir()):
        if p.is_file():
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is not None:
                items.append(img)
    n = len(items)
    hits = {name: 0 for name, _ in FOLDS}
    payloads = []
    for img in items:
        got = {name: fn(img, budget) for name, fn in FOLDS}
        for name in hits:
            hits[name] += got[name] is not None
        payloads.append(got)
    # agreement among folds that returned something, per image
    agree = sum(1 for g in payloads
                if len({v for v in g.values() if v}) == 1 and any(g.values()))
    print(f"no ground truth (web images) -> decode-HITS, not correctness  (n={n})")
    for name, _ in FOLDS:
        print(f"  {name:<22}{hits[name]:>3}/{n}")
    print(f"  images where all decoders that fired agreed on payload: {agree}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--corpus", type=str, default=None)
    ap.add_argument("--pathology", type=str, default=None)
    ap.add_argument("--budget", type=float, default=250.0)
    args = ap.parse_args()

    if args.pathology:
        print(f"=== pathology: {args.pathology} ===")
        run_pathology(args.pathology, args.budget)
        return

    if args.corpus:
        items = load_corpus(args.corpus)
        print(f"=== corpus: {args.corpus} (n={len(items)}) ===")
    else:
        items = [(truth, img) for _, truth, img in _iter_synth(args.per_cell)]
        print(f"=== synth per_cell={args.per_cell} (n={len(items)}) ===")
    run_labelled(items, args.budget)


if __name__ == "__main__":
    main()
