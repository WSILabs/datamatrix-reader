"""L-defect-tolerant finder-registration recovery for damaged DataMatrix codes.

When the printed finder/timing is unreliable (top timing row prints half-height, the
finder L is glare-/chip-/drip-broken) the data modules are usually still intact — the
only missing piece is the module-grid registration. So we localize the code, register a
grid, REPAINT the canonical finder/timing, and let zxing do ECC200 + Reed-Solomon
(ECC-validated → a wrong fit fails safe, never mis-reads).

Localization takes the UNION of two complementary detectors (ablation confirmed
detect_dark_region adds 0 unique recoveries over the pair below):
  • gradient anisotropy — min(|Sobel_x|, |Sobel_y|): a 2D module grid has gradient in
    BOTH directions; ANY straight edge (slide rim, label border, glass-chip boundary)
    has it in one direction only, so min≈0 there → rejected by construction. Smooth
    glare is low-gradient → a hole, not a centroid pull.
  • data-region texture — works when the finder L is broken/obscured.

detect_dark_region remains defined (importable for ablation harnesses) but is NOT
called from decode_auto.

The L is found INSIDE each candidate by the solid-side test (l_orientations): the two
adjacent edge strips reading ~90-100% dark are the finder L; the ~50% strips are timing.
That orders the 4 orientations (loose gate; ECC stays the arbiter) and tolerates a
~90%-intact L.

`recover()` is the Reader entry point: it uses `locate.propose` to find candidate code
regions ANYWHERE on the label at any scale, normalizes each candidate to a canonical
size, and runs `decode_auto` on it. This is format-agnostic — the code can be anywhere
on the label, not just the upper-left. The detectors and L-test themselves are also
format-agnostic.

The registration search (`_brute_region`) iterates hypotheses most-likely-FIRST
(center-out from the detection estimate) so early-exit hits the probable registration
before the unlikely extremes. Coverage is identical to a full sweep — only the iteration
order changes. ECC-validated → never mis-reads.
Speed is not optimized (fallback-only; runs on a cascade miss); likely ports to C later.
"""
from __future__ import annotations

import cv2
import numpy as np
import zxingcpp

from .locate import propose

_DM = zxingcpp.BarcodeFormat.DataMatrix
SIZES = (22, 18, 20, 24)            # candidate square ECC200 sizes, common-first

CANON = 470          # normalized code side (px); decode_auto's grid-search cell range is calibrated at this scale
_MARGIN = 0.6        # crop margin around a proposal, as a fraction of its size


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


def _outward(values, center):
    """Same values, ordered by distance from `center` (most-likely-first) so early-exit
    hits the probable registration before the unlikely extremes. Coverage is unchanged."""
    return sorted(values, key=lambda v: abs(v - center))


def _brute_region(gray, cx, cy, te, ang):
    """Exhaustive registration search for ONE region, ordered most-likely-FIRST: the
    detection's own pitch (te/M), angle (0 offset) and center (0 offset) are tried first,
    expanding outward. Identical coverage to a full sweep -> recall unchanged; only the
    iteration order (hence early-exit speed) differs. ECC-validated -> never mis-reads."""
    for M in SIZES:
        cell0 = te / M
        for cell in _outward(np.arange(te / (M + 3), te / (M - 1), 0.5), cell0):
            for ddeg in _outward(np.arange(-3, 3.01, 1.0), 0.0):
                offs = _outward(np.arange(-1.5, 1.51, 0.375) * cell, 0.0)
                for dcx in offs:
                    for dcy in offs:
                        grid = sample_fast(gray, cx + dcx, cy + dcy, cell, M, ang + ddeg)
                        for g, lsc, _ in l_orientations(grid):
                            if lsc < 0.6:
                                break
                            try:
                                p = _zxing(render_symbol(g, M))
                            except cv2.error:
                                continue
                            if p is not None:
                                return p, (cx + dcx, cy + dcy, cell * M, ang + ddeg)
    return None, None


def decode_auto(gray):
    """Detect (union of 2 detectors) + register + repaint-border + decode an isolated,
    crop-scale grayscale image. Returns (payload, reg) where reg=(cx, cy, side, deg) is
    the code square in `gray`'s coords, or (None, None). ECC-validated.

    Texture-only: `gray` is an already-isolated crop (the detector / YOLO box did the
    localization, including any slide-edge rejection), so the gradient detector's edge
    rejection is redundant here, and it occasionally over-segments and burns a full failed
    search (~4s) on faint codes that aren't repairable anyway. Texture is the reliable,
    precise region (ablation: 7/7 alone). (detect_area stays available for callers that
    work on a non-isolated frame.)"""
    regions = [r for r in (detect_data_region(gray),) if r]
    for cx, cy, te, ang in regions:
        p, reg = _brute_region(gray, cx, cy, te, ang)
        if p is not None:
            return p, reg
    return None, None


def _square_quad(cx, cy, side, deg):
    """The 4 corners of a square of `side` centred at (cx, cy) and rotated by `deg`
    degrees, using the SAME rotation convention as sample_fast (img = center + R @ local
    with R = [[cos, -sin], [sin, cos]]). Returns a (4, 2) float32 array (TL, TR, BR, BL)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    h = side / 2.0
    loc = np.array([[-h, -h], [h, -h], [h, h], [-h, h]], np.float32)
    R = np.array([[c, -s], [s, c]], np.float32)
    return (loc @ R.T + np.array([cx, cy], np.float32)).astype(np.float32)


def _unmap_quad(quad, tf):
    """Map a (4, 2) quad from upscaled-crop coords to original-image coords. tf=(x0,y0,f)."""
    x0, y0, f = tf
    return (np.asarray(quad, np.float32) / f + np.array([x0, y0], np.float32)).astype(np.float32)


def _normalize(gray, cx, cy, size):
    """Crop a window around a proposal and scale so the code is ~CANON px. Returns
    (upscaled_crop, tf) where tf=(x0, y0, f) maps a crop pixel (u, v) back to original
    coords as (x0 + u/f, y0 + v/f). Returns (None, None) on an empty crop."""
    half = int(size * (0.5 + _MARGIN))
    y0 = max(0, int(cy) - half)
    y1 = min(gray.shape[0], int(cy) + half)
    x0 = max(0, int(cx) - half)
    x1 = min(gray.shape[1], int(cx) + half)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    f = CANON / max(1.0, size)
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC), (x0, y0, f)


_yolo = None    # lazy detector singleton: None=untried, False=unavailable, else YoloDetector


def _detector():
    """Return a YoloDetector if its ONNX model is present and onnxruntime imports, else
    None (Reader then uses the classical proposer). Tried once, cached."""
    global _yolo
    if _yolo is None:
        try:
            from .detect import YoloDetector, DEFAULT_MODEL
            _yolo = YoloDetector() if DEFAULT_MODEL.exists() else False
        except Exception:
            _yolo = False
    return _yolo or None


def _decode_region(gray, cx, cy, size):
    """Normalize a candidate region, then the format gate: a readable DataMatrix is the
    fast path; a readable QR/Aztec is NOT a DataMatrix so we skip the repair; otherwise
    it's a (likely damaged) DataMatrix -> repair. Returns (payload, quad) in ORIGINAL
    image coords, or (None, None)."""
    up, tf = _normalize(gray, cx, cy, size)
    if up is None:
        return None, None
    from .detect import format_gate
    payload, fmt, pos = format_gate(up)
    if fmt == "DataMatrix":
        return payload, _unmap_quad(pos, tf)            # gate fast-path
    if fmt is not None:
        return None, None                               # QR/Aztec etc. — not a DataMatrix
    pl, reg = decode_auto(up)                            # nothing read -> damaged DM -> repair
    if pl is None:
        return None, None
    return pl, _unmap_quad(_square_quad(*reg), tf)


def recover(gray):
    """Reader fallback: detect candidate regions (YOLO when its model is installed, else
    the classical proposer), gate each by format, repair damaged DataMatrix codes. Returns
    (payload, quad) with quad the 4 corners in ORIGINAL image coords, or (None, None).
    ECC-validated; classical proposer is the recall safety net when YOLO finds nothing."""
    det = _detector()
    if det is not None:
        for cx, cy, size, _conf in det.detect(gray):
            pl, quad = _decode_region(gray, cx, cy, size)
            if pl is not None:
                return pl, quad
    for cx, cy, size, _ in propose(gray):
        pl, quad = _decode_region(gray, cx, cy, size)
        if pl is not None:
            return pl, quad
    return None, None


def _quad_center(quad):
    return float(np.asarray(quad)[:, 0].mean()), float(np.asarray(quad)[:, 1].mean())


def decode_all(gray, skip=()):
    """Find EVERY DataMatrix region the detector surfaces, plus non-DM 2D hints. Returns
    (dm, other, undecoded): dm/other are lists of (payload, quad, format, stage) (quads in
    ORIGINAL image coords); undecoded is the count of detected regions that yielded no code
    (a possible faint code -> the caller may want the full-frame cascade). Each region:
    format-gate -> a readable DataMatrix (gate fast-path) or a readable QR/Aztec (hint, NOT
    repaired) or, if nothing reads, repair a damaged DataMatrix. `skip` is a list of (cx,cy)
    centers already found (e.g. by a prior raw pass) — regions near them are skipped, so we
    don't re-decode or waste a repair on a code/decoy the caller already has."""
    from .detect import format_gate
    det = _detector()
    cands = (det.detect(gray) if det is not None
             else [(cx, cy, s, 0.0) for cx, cy, s, _ in propose(gray)])
    dm, other, undecoded = [], [], 0
    seen = list(skip)
    for cx, cy, size, *_ in cands:
        if any(abs(cx - sx) < 0.5 * size and abs(cy - sy) < 0.5 * size for sx, sy in seen):
            continue
        up, tf = _normalize(gray, cx, cy, size)
        if up is None:
            continue
        payload, fmt, pos = format_gate(up)
        if fmt == "DataMatrix":
            dm.append((payload, _unmap_quad(pos, tf), "DataMatrix", "gate"))
            seen.append((cx, cy))
        elif fmt is not None:
            other.append((payload, _unmap_quad(pos, tf), fmt, "detector"))
        else:
            pl, reg = decode_auto(up)               # nothing read -> damaged DM -> repair
            if pl is not None:
                dm.append((pl, _unmap_quad(_square_quad(*reg), tf), "DataMatrix", "autoreg"))
                seen.append((cx, cy))
            else:
                undecoded += 1          # a detected region we couldn't decode -> maybe faint
    return dm, other, undecoded
