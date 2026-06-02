"""Compare two harness runs and surface which strata moved.

A config change is only an improvement if it lifts a weak stratum without
regressing others — adding a cascade rung is justified by Pareto, not by mean.

Usage:  python -m bench.report runs/before.json runs/after.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: str) -> dict:
    return json.loads(Path(p).read_text())["summary"]


def main():
    if len(sys.argv) != 3:
        print("usage: python -m bench.report BEFORE.json AFTER.json")
        raise SystemExit(2)
    a, b = load(sys.argv[1]), load(sys.argv[2])

    print(f"{'metric':<16}{'before':>10}{'after':>10}{'delta':>10}")
    for k in ("correct_rate", "found_rate", "p50_ms", "p95_ms", "max_ms"):
        va, vb = a["overall"][k], b["overall"][k]
        print(f"{k:<16}{va:>10.3f}{vb:>10.3f}{vb - va:>+10.3f}")

    print("\nstratum regressions / gains (correct rate):")
    for ax in a["stratified"]:
        for val in a["stratified"][ax]:
            va = a["stratified"][ax][val]
            vb = b["stratified"].get(ax, {}).get(val)
            if vb is None:
                continue
            d = vb - va
            flag = "  <-- REGRESSED" if d < -1e-9 else ""
            if abs(d) > 1e-9:
                print(f"  {ax}={val:<14}{va:.3f} -> {vb:.3f} ({d:+.3f}){flag}")

    print(f"\nworst stratum  before: {a['worst_stratum']}")
    print(f"worst stratum  after : {b['worst_stratum']}")


if __name__ == "__main__":
    main()
