"""L-defect-tolerant finder-registration recovery for damaged DataMatrix codes.

When the printed finder/timing is unreliable (top timing row prints half-height, the
finder L is glare-/chip-/drip-broken) the data modules are usually still intact — the
only missing piece is the module-grid registration. So we localize the code, register a
grid, REPAINT the canonical finder/timing, and let zxing do ECC200 + Reed-Solomon
(ECC-validated → a wrong fit fails safe, never mis-reads).

Localization takes the UNION of three complementary detectors (each catches what the
others miss; none alone exceeds ~4/7 on the WSI residual):
  • gradient anisotropy — min(|Sobel_x|, |Sobel_y|): a 2D module grid has gradient in
    BOTH directions; ANY straight edge (slide rim, label border, glass-chip boundary)
    has it in one direction only, so min≈0 there → rejected by construction. Smooth
    glare is low-gradient → a hole, not a centroid pull.
  • dark-ink extent — precise center when the finder L is intact.
  • data-region texture — works when the finder L is broken/obscured.

The L is found INSIDE each candidate by the solid-side test (l_orientations): the two
adjacent edge strips reading ~90-100% dark are the finder L; the ~50% strips are timing.
That orders the 4 orientations (loose gate; ECC stays the arbiter) and tolerates a
~90%-intact L.

`recover()` is the Reader entry point: it crops to the consensus code-ROI (a spatial
prior — these labels place the code in the upper-left) and upscales, which puts the small
full-label code at the scale `decode_auto` expects. The ROI/size priors are calibrated to
the Grundium WSI label format; the detectors and L-test themselves are format-agnostic.
Speed is not optimized (fallback-only; runs on a cascade miss); likely ports to C later.
"""
from __future__ import annotations

import cv2
import numpy as np
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
SIZES = (22, 18, 20, 24)            # candidate square ECC200 sizes, common-first

# Consensus code-ROI as fractions of (W, H): the WSI label places the DataMatrix in the
# upper-left; from the 350-readable-code position study (x≈0.20±0.05, y≈0.32±0.06). The
# box is generous (~1.6× the code) so small per-label shifts stay inside it.
ROI_FRAC = (0.03, 0.36, 0.09, 0.55)   # x0, x1, y0, y1


def _zxing(img: np.ndarray) -> bytes | None:
    res = zxingcpp.read_barcodes(np.ascontiguousarray(img), formats=_DM)
    return res[0].bytes if res else None


def _kernel(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


# ---- DataMatrix border repaint ---------------------------------------------------

def border_mask(M: int) -> np.ndarray:
    """True where an M×M ECC200 square symbol's finder/timing module is DARK. Left col +
    bottom row solid (L finder); top row dark at even cols, right col dark at odd rows."""
    r = np.arange(M)[:, None]
    c = np.arange(M)[None, :]
    return ((c == 0) | (r == M - 1)
            | ((r == 0) & (c % 2 == 0))
            | ((c == M - 1) & (r % 2 == 1)))


def render_symbol(grid: np.ndarray, M: int, quiet: int = 2) -> np.ndarray:
    """Clean 1px/module image from an M×M sampled grid (True=dark): overwrite the border
    with the canonical finder/timing (repairing the damage), keep the interior data, add a
    quiet zone, upscale 8× for reliable zxing detection."""
    dark = grid.astype(bool).copy()
    bm = border_mask(M)
    isb = np.zeros((M, M), bool)
    isb[0, :] = isb[-1, :] = isb[:, 0] = isb[:, -1] = True
    dark[isb] = bm[isb]
    sym = np.where(dark, 0, 255).astype(np.uint8)
    sym = cv2.copyMakeBorder(sym, quiet, quiet, quiet, quiet,
                             cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(sym, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)


# ---- grid sampling ---------------------------------------------------------------

def sample_fast(gray, cx, cy, cell, M, deg) -> np.ndarray:
    """Sample an M×M module grid in one warpAffine (Minv maps output module (j,i) ->
    image (X,Y) at the module centre). Dark via p10/p90 midpoint."""
    t = np.radians(deg)
    cos, sin = np.cos(t), np.sin(t)
    off = (0.5 - M / 2) * cell
    Minv = np.array([
        [cell * cos, -cell * sin, cx + off * (cos - sin)],
        [cell * sin,  cell * cos, cy + off * (sin + cos)],
    ], np.float32)
    cells = cv2.warpAffine(gray, Minv, (M, M),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
    thr = (np.percentile(cells, 10) + np.percentile(cells, 90)) / 2.0
    return cells < thr


# ---- detectors (each returns (cx, cy, extent_px, angle_deg) or None) -------------

def _texture(gray, box):
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, box)
    var = cv2.boxFilter(g * g, -1, box) - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def detect_dark_region(gray):
    """Dark-ink extent, gated by texture-overlap (rejects the solid slide edge: dark but
    no internal texture). Precise center when the finder L is intact."""
    tex = (_texture(gray, (15, 15)) > 0).astype(np.uint8)
    tex = cv2.morphologyEx(tex, cv2.MORPH_OPEN, _kernel(9))
    dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _kernel(15))
    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    best = None
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < 60 or h < 60 or max(w, h) / max(1.0, min(w, h)) > 1.4:
            continue
        if cv2.contourArea(c) / max(1.0, w * h) < 0.5:
            continue
        m = np.zeros((H, W), np.uint8); cv2.drawContours(m, [c], -1, 1, -1)
        if tex[m > 0].mean() < 0.25:
            continue
        area = w * h
        if best is None or area > best[0]:
            best = (area, (cx, cy, (w + h) / 2.0, ang))
    return None if best is None else best[1]


def detect_data_region(gray):
    """High-texture DATA region. L-defect tolerant — independent of the printed finder."""
    std = _texture(gray, (13, 13)).astype(np.uint8)
    tex = cv2.threshold(std, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    tex = cv2.morphologyEx(tex, cv2.MORPH_OPEN, _kernel(7))     # drop thin text
    tex = cv2.morphologyEx(tex, cv2.MORPH_CLOSE, _kernel(21))   # fill the data region
    cnts, _ = cv2.findContours(tex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < 60 or h < 60 or max(w, h) / max(1.0, min(w, h)) > 1.6:
            continue
        area = w * h
        if best is None or area > best[0]:
            best = (area, (cx, cy, (w + h) / 2.0, ang))
    return None if best is None else best[1]


def detect_area(gray):
    """Gradient-anisotropy region: min(|Sobel_x|, |Sobel_y|) is high only on a 2D module
    grid, near-zero on any straight edge (slide rim / label border / glass-chip boundary)
    and on smooth glare — so those are rejected by construction."""
    gx = cv2.boxFilter(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), -1, (31, 31))
    gy = cv2.boxFilter(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), -1, (31, 31))
    dens = np.minimum(gx, gy)
    dn = (dens / (dens.max() + 1e-9) * 255).astype(np.uint8)
    th = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, _kernel(31))     # fill chip hole / gaps
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, _kernel(9))       # drop specks
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < 60 or h < 60 or max(w, h) / max(1.0, min(w, h)) > 1.5:
            continue
        area = w * h
        if best is None or area > best[0]:
            best = (area, (cx, cy, (w + h) / 2.0, ang))
    return None if best is None else best[1]


def l_orientations(grid):
    """Rank the 4 rotations by L-solidity. Each entry (oriented_grid, l, timing): l = mean
    dark of the two arms that would be the finder L (left col + bottom row); timing = mean
    dark of the other two. Best-first → the top puts the most-solid L at left+bottom."""
    out = []
    for k in range(4):
        g = np.rot90(grid, k)
        out.append((g, (g[:, 0].mean() + g[-1, :].mean()) / 2.0,
                    (g[0, :].mean() + g[:, -1].mean()) / 2.0))
    return sorted(out, key=lambda e: -e[1])


def decode_auto(gray):
    """Detect (union of 3 detectors) + register + find-L + repaint-border + decode an
    already-isolated, crop-scale grayscale image. Returns (payload, params) or
    (None, None). ECC-validated → never returns a wrong payload."""
    regions = [r for r in (detect_area(gray),
                           detect_dark_region(gray),
                           detect_data_region(gray)) if r]
    for cx, cy, te, ang in regions:
        for M in SIZES:
            # extent may under-shoot (finder excluded) or over-shoot (merged); bracket wide
            for cell in np.arange(te / (M + 3), te / (M - 1), 0.5):
                for ddeg in np.arange(-3, 3.01, 1.0):
                    for dcx in np.arange(-1.5, 1.51, 0.375) * cell:
                        for dcy in np.arange(-1.5, 1.51, 0.375) * cell:
                            grid = sample_fast(gray, cx + dcx, cy + dcy, cell, M, ang + ddeg)
                            for g, lsc, _ in l_orientations(grid):
                                if lsc < 0.6:                  # skip non-L grids (sorted desc)
                                    break
                                try:
                                    p = _zxing(render_symbol(g, M))
                                except cv2.error:
                                    continue
                                if p is not None:
                                    return p, dict(M=M, cell=round(float(cell), 2),
                                                   deg=round(float(ang + ddeg), 2),
                                                   cx=round(float(cx + dcx), 1),
                                                   cy=round(float(cy + dcy), 1),
                                                   Lsolid=round(float(lsc), 2))
    return None, None


def recover(gray: np.ndarray, upscale: int = 2) -> bytes | None:
    """Reader fallback: crop to the consensus code-ROI and upscale (putting the small
    full-label code at the crop scale decode_auto expects), then decode_auto. Returns the
    payload bytes or None. ECC-validated → safe."""
    H, W = gray.shape
    x0f, x1f, y0f, y1f = ROI_FRAC
    crop = gray[int(y0f * H):int(y1f * H), int(x0f * W):int(x1f * W)]
    if crop.size == 0:
        return None
    up = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    payload, _ = decode_auto(up)
    return payload
