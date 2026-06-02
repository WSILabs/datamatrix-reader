"""Synthetic, distribution-controlled test data — the PRIMARY optimisation
surface for a source-agnostic reader.

We render known payloads with libdmtx's own encoder, then push them through a
parametric degradation model spanning the axes you want to generalise across:
module size, blur, ink gain/dropout, label colour, rotation, noise. This gives
arbitrarily many perfectly-labelled samples from printers/colours/sizes you
don't own. Your real corpus is the reality check on this model, never the
thing you tune against.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import binding


@dataclass(frozen=True)
class DegradeParams:
    module_px: float = 8.0       # target pixels per module (size/resolution axis)
    blur_sigma: float = 0.0      # gaussian blur (focus / ink bleed)
    ink_gain: int = 0            # dilation iters of dark ink (dot gain)
    dropout: float = 0.0         # fraction of module pixels punched out
    rotation_deg: float = 0.0
    noise_sigma: float = 0.0
    substrate_bgr: tuple = (255, 255, 255)
    print_bgr: tuple = (0, 0, 0)
    jpeg_quality: int = 0        # 0 = none
    quiet_crowd: bool = False    # encroach text on the quiet zone


def render(payload: bytes) -> np.ndarray:
    """Clean 1px-module grid (0/255)."""
    return binding.encode(payload, module_size=1, margin=2)


def degrade(grid: np.ndarray, p: DegradeParams, rng: random.Random) -> np.ndarray:
    """Clean grid -> a realistic BGR capture, exercising the colour path."""
    img = cv2.resize(grid, None, fx=p.module_px, fy=p.module_px,
                     interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0

    if p.ink_gain:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark = cv2.erode(img, k, iterations=p.ink_gain)  # grow dark modules
        img = dark
    if p.dropout > 0:
        mask = (np.random.default_rng(rng.randint(0, 1 << 30))
                .random(img.shape) < p.dropout)
        img[mask] = 1.0  # punch holes (under-inking)

    # colourise: print colour where dark, substrate where light
    sub = np.array(p.substrate_bgr, np.float32) / 255.0
    ink = np.array(p.print_bgr, np.float32) / 255.0
    color = img[..., None] * sub + (1 - img[..., None]) * ink

    if p.quiet_crowd:
        h, w, _ = color.shape
        cv2.putText(color, "S25-0001", (2, h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    h / 90.0, tuple(map(float, ink)), 1, cv2.LINE_AA)

    if p.rotation_deg:
        h, w = color.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), p.rotation_deg, 1.0)
        color = cv2.warpAffine(color, M, (w, h), borderValue=tuple(map(float, sub)))

    color = (color * 255).clip(0, 255).astype(np.uint8)
    if p.blur_sigma > 0:
        color = cv2.GaussianBlur(color, (0, 0), p.blur_sigma)
    if p.noise_sigma > 0:
        color = (color.astype(np.float32)
                 + np.random.default_rng(rng.randint(0, 1 << 30))
                 .normal(0, p.noise_sigma, color.shape)).clip(0, 255).astype(np.uint8)
    if p.jpeg_quality:
        ok, enc = cv2.imencode(".jpg", color,
                               [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        if ok:
            color = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # pad so localization has context around the code
    return cv2.copyMakeBorder(color, 40, 40, 40, 40, cv2.BORDER_CONSTANT,
                              value=tuple(map(int, np.array(p.substrate_bgr))))


# axis grids — each value becomes a reporting stratum
AXES = {
    "module_px": [3.0, 4.0, 6.0, 10.0],
    "blur_sigma": [0.0, 0.8, 1.6],
    "ink_gain": [0, 1, 2],
    "substrate": [((255, 255, 255), (0, 0, 0)),       # white / black
                  ((180, 230, 255), (20, 20, 20)),     # yellow stock
                  ((230, 200, 255), (40, 20, 40))],    # pink stock
    "rotation_deg": [0.0, 7.0],
}


def strata(payloads: list[bytes], *, seed: int = 0, per_cell: int = 1):
    """Yield (stratum:dict, payload:bytes, image:np.ndarray) across the grid.

    Stratified so the harness can report read rate per axis value and surface
    the *worst* stratum — the real measure of source-agnosticism.
    """
    rng = random.Random(seed)
    keys = list(AXES)
    for combo in itertools.product(*[AXES[k] for k in keys]):
        cell = dict(zip(keys, combo))
        (sub, ink) = cell.pop("substrate")
        for _ in range(per_cell):
            payload = rng.choice(payloads)
            p = DegradeParams(module_px=cell["module_px"],
                              blur_sigma=cell["blur_sigma"],
                              ink_gain=cell["ink_gain"],
                              rotation_deg=cell["rotation_deg"],
                              substrate_bgr=sub, print_bgr=ink,
                              quiet_crowd=True)
            img = degrade(render(payload), p, rng)
            stratum = {**cell, "substrate": f"{sub}"}
            yield stratum, payload, img
