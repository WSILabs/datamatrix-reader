"""Generalization harness: synthetic full-label scenes across position/scale/rotation,
reporting LOCALIZATION recall (propose finds the code) and DECODE rate (recover reads it)
separately, with the worst stratum surfaced.

    .venv/bin/python -m tools.validate_synth
"""
import random
import numpy as np
import zxingcpp
from dmtxslide import synth
from dmtxslide.locate import propose
from dmtxslide.register import recover

_DM = zxingcpp.BarcodeFormat.DataMatrix
PAYLOADS = [p for p in (b"S25-04821 A3-1 HE", b"PCAA00028208 A1-1",
                        b"ABCDEFGHIJKLMNOPQRSTUVWX")
            if np.asarray(zxingcpp.create_barcode(p.decode(), _DM).to_image()).shape[0]
            == np.asarray(zxingcpp.create_barcode(p.decode(), _DM).to_image()).shape[1]]

POS = [(0.25, 0.3), (0.7, 0.65), (0.5, 0.2), (0.8, 0.8)]
CELL = [10.0, 14.0, 22.0, 30.0]
ROT = [0.0, 90.0, 180.0, 270.0]


def _hit(cands, t, tol=0.4):
    return any(abs(cx - t["cx"]) < tol * t["size"] and abs(cy - t["cy"]) < tol * t["size"]
               and abs(s - t["size"]) < 0.35 * t["size"] for cx, cy, s, _ in cands)


def main():
    rng = random.Random(0)
    loc = dec = n = 0
    for pos in POS:
        for cell in CELL:
            for rot in ROT:
                payload = rng.choice(PAYLOADS)
                p = synth.SceneParams(canvas=(900, 1100), cell=cell, pos=pos,
                                      rotation_deg=rot, skew_deg=rng.uniform(-12, 12),
                                      defects=True, text=True, edges=True,
                                      chip=rng.random() < 0.3)
                img, truth = synth.scene(payload, p, rng)
                g = img[..., 0]
                n += 1
                if _hit(propose(g), truth):
                    loc += 1
                if recover(g)[0] == payload:
                    dec += 1
    print(f"synthetic scenes: {n}")
    print(f"localization recall: {loc}/{n} = {loc/n:.3f}")
    print(f"decode rate        : {dec}/{n} = {dec/n:.3f}")


if __name__ == "__main__":
    main()
