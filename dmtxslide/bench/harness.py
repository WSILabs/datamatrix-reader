"""Benchmark harness.

Objective is generalisation, not a single corpus number. We report:
  * overall correct-decode rate and latency p50/p95 (the $20-reader yardstick);
  * read rate stratified by each axis -> the WORST stratum, which is how a
    source-agnostic reader is actually judged;
  * a per-stage breakdown (found vs decoded) so you tune the right stage.

Synthetic strata are the optimisation surface; a real corpus dir (with
labels.csv) is run the same way as confirmation.

Usage:
    python -m bench.harness --synth --per-cell 2 --budget 250 --out runs/a.json
    python -m bench.harness --corpus corpus --out runs/real.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as stats
import time
from collections import defaultdict
from pathlib import Path

import cv2

from dmtxslide.reader import Reader
from dmtxslide.synth import strata
from dmtxslide.validate import AcceptAny


def _payload_pool() -> list[bytes]:
    # Mixed-length accession formats so the benchmark spans real symbol sizes
    # (~12x12 .. 24x24 modules); payload length drives the symbol dimension.
    fmts = [
        lambda n: f"S{n % 25:02d}-{n:04d}-B",            # short
        lambda n: f"S25-{n:05d}-A{n % 9}",               # medium
        lambda n: f"PCAA{n:08d} B{n % 4}-{n % 9}",       # long
        lambda n: f"GDC-{n % 9:02d}-{n:06d}, CASE",      # longer
    ]
    return [fmts[n % len(fmts)](n).encode() for n in range(1, 40)]


def _iter_synth(per_cell: int):
    for stratum, truth, img in strata(_payload_pool(), per_cell=per_cell):
        yield stratum, truth, img


def _iter_corpus(root: Path):
    labels = {}
    lp = root / "labels.csv"
    if lp.exists():
        with lp.open() as f:
            for row in csv.DictReader(f):
                labels[row["file"]] = row["payload"].encode()
    for p in sorted((root / "images").glob("*")):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        yield {"file": p.name}, labels.get(p.name), img


def run(items, budget_ms: float):
    reader = Reader(validator=AcceptAny())
    rows = []
    for stratum, truth, img in items:
        r = reader.read(img, budget_ms=budget_ms)
        correct = bool(r.ok and (truth is None or r.payload == truth))
        # best stage reached across candidates (for the found-vs-decoded split)
        found = any(t.found for _, res in r.candidate_traces for t in res.trace)
        rows.append({
            "stratum": stratum, "correct": correct, "decoded": r.ok,
            "found": found, "rung": r.rung, "ms": r.elapsed_ms,
            "truth": truth.decode() if truth else None,
            "got": r.payload.decode("latin1") if r.payload else None,
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ms = sorted(r["ms"] for r in rows)
    def pct(p):
        return ms[min(len(ms) - 1, int(p * len(ms)))] if ms else 0.0
    overall = {
        "n": n,
        "correct_rate": sum(r["correct"] for r in rows) / n if n else 0,
        "found_rate": sum(r["found"] for r in rows) / n if n else 0,
        "p50_ms": pct(0.50), "p95_ms": pct(0.95),
        "max_ms": ms[-1] if ms else 0.0,
    }
    # stratified rates per axis value — BOTH correct and found, so each weak
    # stratum is diagnosable: found-limited (localization) vs decoded-limited
    # (sampling/RS). The found-vs-correct split is the metric that matters.
    by_correct: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    by_found: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r["stratum"].items():
            by_correct[k][str(v)].append(r["correct"])
            by_found[k][str(v)].append(r["found"])
    strat = {ax: {val: sum(c) / len(c) for val, c in vals.items()}
             for ax, vals in by_correct.items()}
    strat_found = {ax: {val: sum(c) / len(c) for val, c in vals.items()}
                   for ax, vals in by_found.items()}
    worst = None
    for ax, vals in strat.items():
        for val, rate in vals.items():
            if worst is None or rate < worst[2]:
                worst = (ax, val, rate)
    worst_stratum = None
    if worst:
        ax, val, rate = worst
        fr = strat_found[ax][val]
        # localization gap = codes never found; sampling gap = found-but-undecoded
        loc_gap, samp_gap = 1.0 - fr, fr - rate
        worst_stratum = {
            "axis": ax, "value": val, "rate": rate, "found_rate": fr,
            "limited_by": "localization" if loc_gap >= samp_gap else "sampling",
        }
    return {"overall": overall, "stratified": strat,
            "stratified_found": strat_found, "worst_stratum": worst_stratum}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--corpus", type=str, default=None)
    ap.add_argument("--budget", type=float, default=250.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.corpus:
        items = list(_iter_corpus(Path(args.corpus)))
    else:
        items = list(_iter_synth(args.per_cell))

    t0 = time.perf_counter()
    rows = run(items, args.budget)
    summary = summarize(rows)
    summary["wall_s"] = time.perf_counter() - t0

    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
