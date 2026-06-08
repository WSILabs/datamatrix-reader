"""Synthetic, distribution-controlled test data — the PRIMARY optimisation
surface for a source-agnostic reader.

We render known payloads with zxing-cpp's encoder, then push them through a
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

import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix


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
    """Clean 1px-module DataMatrix grid (0/255), encoded by zxing-cpp.

    Passing bytes yields a tight 1-pixel-per-module bitmap with zxing's quiet
    zone; synth.degrade adds its own scaling/border/crowding on top."""
    return np.asarray(zxingcpp.create_barcode(payload, _DM).to_image()).copy()


# Accession-style strings used to crowd the quiet zone, modelled on real
# pathology labels (see decoded BarBeR/pathology samples).
ACCESSION_SAMPLES = [
    "S25-04821 A3", "PCAA00028208", "B1-2  H&E", "GDC-04-123456",
    "370956.1/10 PAS", "Smith, W", "2025-06-02", "BLOCK 2  L3",
]


def crowd_quiet_zone(code: np.ndarray, substrate_bgr, ink_bgr,
                     rng, margin_frac: float = 0.6) -> np.ndarray:
    """Place the code on a substrate canvas with accession text crowding the
    quiet-zone margin on two sides.

    Models the real pathology failure: text encroaches on the code's quiet zone
    (stressing *localization*), while the code's own modules stay pristine — so
    the code is re-stamped after the text is drawn, guaranteeing no module is
    overwritten regardless of how far the text overshoots.
    """
    h, w = code.shape[:2]
    m = max(12, int(margin_frac * max(h, w)))
    canvas = np.full((h + 2 * m, w + 2 * m, 3), substrate_bgr, np.uint8)
    y0, x0 = m, m  # centred -> equal margins on every side
    ink = tuple(int(c) for c in ink_bgr)
    fs = max(0.3, m / 40.0)
    th = max(1, int(round(fs * 1.6)))
    # bottom margin: a line encroaching up toward the code
    cv2.putText(canvas, rng.choice(ACCESSION_SAMPLES),
                (x0, y0 + h + int(m * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, ink, th, cv2.LINE_AA)
    # right margin: a short second-side label
    cv2.putText(canvas, rng.choice(ACCESSION_SAMPLES)[:6],
                (x0 + w + int(m * 0.1), y0 + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, ink, th, cv2.LINE_AA)
    # re-stamp the code so its modules are pristine even if text overshot
    canvas[y0:y0 + h, x0:x0 + w] = code
    return canvas


def degrade(grid: np.ndarray, p: DegradeParams, rng: random.Random) -> np.ndarray:
    """Clean grid -> a realistic BGR capture, exercising the colour path."""
    img = cv2.resize(grid, None, fx=p.module_px, fy=p.module_px,
                     interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0

    if p.ink_gain:
        # Dot gain keyed to module pitch: thicken dark modules by a fraction of
        # a module, so it models the same physical print artifact at any
        # resolution. A fixed-pixel kernel annihilates low-px codes (a solid
        # blob no reader can read) — unrealistic and an unfair benchmark sample.
        rad = int(round(p.ink_gain * 0.12 * p.module_px))
        if rad >= 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
            img = cv2.erode(img, k, iterations=1)  # grow dark (ink) regions
    if p.dropout > 0:
        mask = (np.random.default_rng(rng.randint(0, 1 << 30))
                .random(img.shape) < p.dropout)
        img[mask] = 1.0  # punch holes (under-inking)

    # colourise: print colour where dark, substrate where light
    sub = np.array(p.substrate_bgr, np.float32) / 255.0
    ink = np.array(p.print_bgr, np.float32) / 255.0
    color = img[..., None] * sub + (1 - img[..., None]) * ink

    if p.quiet_crowd:
        cu = (color * 255).clip(0, 255).astype(np.uint8)
        cu = crowd_quiet_zone(cu, p.substrate_bgr, p.print_bgr, rng)
        color = cu.astype(np.float32) / 255.0

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


# axis grids — each value becomes a reporting stratum. Calibrated to the
# pathology slide/cassette domain from decoded real samples: low pixels/module,
# saturated colour stock plus low-contrast laser etch, modest rotation (flat
# capture). Every value is kept non-degenerate by tests/test_synth.py.
AXES = {
    "module_px": [2.0, 3.0, 4.5, 8.0],
    "blur_sigma": [0.0, 0.8, 1.6],
    "ink_gain": [0, 1, 2],
    "substrate": [((255, 255, 255), (10, 10, 10)),     # white
                  ((60, 230, 255), (10, 10, 10)),      # saturated yellow stock
                  ((200, 150, 255), (10, 10, 10)),     # pink stock
                  ((130, 230, 150), (10, 10, 10)),     # green stock
                  ((255, 200, 130), (10, 10, 10)),     # blue stock
                  ((200, 200, 200), (120, 120, 120))], # low-contrast laser etch
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
