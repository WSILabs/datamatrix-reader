"""Dump representative synthetic images to disk for visual inspection.

The harness generates synth images in memory and discards them; this writes a
stratified, full-resolution sample (the exact render->degrade path the
benchmark uses) plus a contact sheet, so you can see what the reader is graded
on.

    python -m tools.dump_synth_samples --out runs/synth_samples
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from datamatrix_reader.synth import render, degrade, DegradeParams, AXES

SUBS = dict(zip(["white", "yellow", "pink", "green", "blue", "laser"],
                AXES["substrate"]))


def _name(p: DegradeParams, subname: str) -> str:
    return (f"mpx{p.module_px}_blur{p.blur_sigma}_ink{p.ink_gain}"
            f"_rot{p.rotation_deg}_{subname}_crowd{int(p.quiet_crowd)}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/synth_samples")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    payload = b"PCAA00028208 B1-2"
    grid = render(payload)

    sub_w, ink_w = SUBS["white"]
    base = dict(blur_sigma=0.0, ink_gain=0, rotation_deg=0.0,
                substrate_bgr=sub_w, print_bgr=ink_w, quiet_crowd=True)

    samples: list[tuple[str, np.ndarray]] = []

    def add(p: DegradeParams, subname: str):
        img = degrade(grid, p, rng)
        fn = _name(p, subname)
        cv2.imwrite(str(out / fn), img)
        samples.append((fn, img))

    # sweep module_px (others easy)
    for m in AXES["module_px"]:
        add(DegradeParams(module_px=m, **base), "white")
    # sweep substrate at a mid module size
    for name, (sub, ink) in SUBS.items():
        add(DegradeParams(module_px=3.0, blur_sigma=0.0, ink_gain=0,
                          rotation_deg=0.0, substrate_bgr=sub, print_bgr=ink,
                          quiet_crowd=True), name)
    # sweep blur and ink_gain (the other weak strata)
    for b in AXES["blur_sigma"]:
        add(DegradeParams(module_px=3.0, blur_sigma=b, ink_gain=0,
                          rotation_deg=0.0, substrate_bgr=sub_w, print_bgr=ink_w,
                          quiet_crowd=True), "white")
    for g in AXES["ink_gain"]:
        add(DegradeParams(module_px=3.0, blur_sigma=0.0, ink_gain=g,
                          rotation_deg=0.0, substrate_bgr=sub_w, print_bgr=ink_w,
                          quiet_crowd=True), "white")

    # contact sheet
    cell = 260
    cols = 5
    rows = (len(samples) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 40, np.uint8)
    for i, (fn, img) in enumerate(samples):
        h, w = img.shape[:2]
        s = (cell - 28) / max(h, w)
        thumb = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(i, cols)
        yo, xo = r * cell, c * cell
        yh, xw = thumb.shape[:2]
        sheet[yo + 4:yo + 4 + yh, xo + 4:xo + 4 + xw] = thumb
        cv2.putText(sheet, fn[:-4], (xo + 4, yo + cell - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
    cv2.imwrite(str(out / "_contact_sheet.png"), sheet)
    print(f"wrote {len(samples)} samples + _contact_sheet.png to {out}/")


if __name__ == "__main__":
    main()
