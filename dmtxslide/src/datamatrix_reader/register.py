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

# The cascade's 4x-upscale (u4) stages cost ~3x a u2 stage (16MP vs 4MP full-frame zxing
# scan) and only help UNDER-SAMPLED symbols — zxing needs ~3-5 px/module, so u4 earns its
# keep only once native px/module drops toward the floor (small/dense/low-res codes). When
# the detector localizes a comfortably-oversampled code we skip u4 (full-frame retained).
_EST_MODULES = 24    # nominal modules across a symbol incl. quiet zone (M=22 dominates this corpus)
_PXMOD_GATE = 6.0    # run u4 only when est. px/module (detector size / _EST_MODULES) is below this


def _needs_u4(region_sizes):
    """Whether the cascade should run the costly 4x-upscale (u4) stages. True when no region
    was localized (blind safety net) or any region is small enough that even u2 leaves it
    under-sampled (est. px/module < _PXMOD_GATE); False when every region is oversampled."""
    return (not region_sizes
            or any(s / _EST_MODULES < _PXMOD_GATE for s in region_sizes))


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
    # Dark via p10/p90 midpoint. np.partition picks both cutpoints in one partial sort —
    # np.percentile's per-call machinery dominated the brute-force (~35% of decode_auto).
    flat = cells.reshape(-1)
    n = flat.size
    lo, hi = n // 10, n * 9 // 10
    part = np.partition(flat, (lo, hi))
    thr = (float(part[lo]) + float(part[hi])) / 2.0
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


def _square_from_coverage(shape, contour, ang):
    """Best-fit square to a region contour, robust to NARROW protrusions (e.g. ink dripping
    below the finder L). Axis-align the filled contour, then on each axis keep only the span
    whose coverage exceeds half the plateau: the drip's partial-width rows/cols fall below and
    are clipped, the full-width data rows/cols are kept. Returns (cx, cy, side, ang) in the
    original frame, or None if the span is degenerate. A clean square clips to itself (no-op),
    so this only moves the start where an asymmetric protrusion was dragging center/extent."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    cv2.drawContours(m, [contour], -1, 1, -1)
    mr = cv2.warpAffine(m, cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0), (w, h))

    def span(cov):
        cov = cov.astype(np.float32)
        strong = cov[cov > 0.5 * cov.max()]
        if strong.size == 0:
            return None
        idx = np.where(cov > 0.5 * float(np.median(strong)))[0]
        return (int(idx[0]), int(idx[-1])) if idx.size else None

    xs, ys = span(mr.sum(0)), span(mr.sum(1))
    if xs is None or ys is None:
        return None
    cxr, cyr = (xs[0] + xs[1]) / 2.0, (ys[0] + ys[1]) / 2.0
    side = ((xs[1] - xs[0]) + (ys[1] - ys[0])) / 2.0
    p = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -ang, 1.0) @ np.array([cxr, cyr, 1.0])
    return float(p[0]), float(p[1]), float(side), ang


def detect_data_region(gray):
    """High-texture DATA region. L-defect tolerant — independent of the printed finder.
    Returns (cx, cy, te, ang, ocx, ocy, oside): the texture region's minAreaRect (cx, cy, te)
    is the SEARCH-RANGE anchor; (ocx, ocy, oside) is a coverage-clip best-fit SQUARE used only
    to ORDER the search most-likely-first. The clip is robust to a narrow protrusion — e.g. ink
    dripping below the finder L — that drags the rect's center down and inflates its extent. The
    ranges stay anchored to the rect so coverage (hence recall) is unchanged; only the iteration
    order benefits from the cleaner estimate. Falls back to the rect for ordering too if the
    clip is degenerate."""
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
            best = (area, c, (cx, cy, (w + h) / 2.0, ang))
    if best is None:
        return None
    rcx, rcy, rte, ang = best[2]
    sq = _square_from_coverage(gray.shape, best[1], ang)
    ocx, ocy, oside = (sq[0], sq[1], sq[2]) if sq is not None else (rcx, rcy, rte)
    return rcx, rcy, rte, ang, ocx, ocy, oside


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
    """Yield the 4 rotations best-L-first as (oriented_grid, l, timing): l = mean dark of
    the two arms that would be the finder L (left col + bottom row of the rotated grid);
    timing = mean dark of the other two. The 4 scores are derived from the grid's 4 border
    means (no per-rotation rotation needed to RANK), and each rotated grid is materialised
    lazily as it's yielded — so the caller's early break (l < 0.6) skips the unused rot90s.
    Equivalent to the eager rot90-all-4 version (verified)."""
    T, B = grid[0, :].mean(), grid[-1, :].mean()
    L, Rt = grid[:, 0].mean(), grid[:, -1].mean()
    sc = [((L + B) / 2.0, (T + Rt) / 2.0), ((T + L) / 2.0, (Rt + B) / 2.0),
          ((Rt + T) / 2.0, (B + L) / 2.0), ((B + Rt) / 2.0, (L + T) / 2.0)]
    for k in sorted(range(4), key=lambda k: -sc[k][0]):
        yield np.rot90(grid, k), sc[k][0], sc[k][1]


def _outward(values, center):
    """Same values, ordered by distance from `center` (most-likely-first) so early-exit
    hits the probable registration before the unlikely extremes. Coverage is unchanged."""
    return sorted(values, key=lambda v: abs(v - center))


def _fft_pitch(gray, lo, hi):
    """Dominant module pitch (px) in [lo, hi] via the FFT power-spectrum peak (Hanning window
    + parabolic sub-pixel refinement) of gradient-energy row/col profiles. Validated ~95%
    correct at M-selection on this corpus (vs zxing extra['Version']); used ONLY to order the
    symbol-size search, so a wrong estimate costs speed, never recall. Returns None if the band
    is empty. (Plain autocorrelation was biased ~1.5px low and misordered M — the FFT peak is
    unbiased; see the autoreg/cascade memory.)"""
    lo, hi = max(1, int(lo)), int(hi)
    if hi <= lo:
        return None
    g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), max(gray.shape) / 20.0)      # drop low-freq shading
    pitches = []
    for prof in (np.abs(np.diff(g, axis=0)).mean(axis=1),
                 np.abs(np.diff(g, axis=1)).mean(axis=0)):
        prof = prof - prof.mean()
        N = prof.size
        P = np.abs(np.fft.rfft(prof * np.hanning(N))) ** 2
        klo, khi = max(1, int(N / hi)), min(len(P) - 2, int(N / lo))
        if khi <= klo:
            continue
        k = klo + int(np.argmax(P[klo:khi + 1]))
        d = P[k - 1] - 2 * P[k] + P[k + 1]                          # parabolic peak refine
        pitches.append(N / (k + (0.5 * (P[k - 1] - P[k + 1]) / d if d else 0.0)))
    return float(np.mean(pitches)) if pitches else None


def _brute_region(gray, cx, cy, te, ang, ocx=None, ocy=None, oside=None):
    """Exhaustive registration search for ONE region, ordered most-likely-FIRST. The search
    RANGES are anchored to the rect (cx, cy, te) — identical coverage to a full sweep, so recall
    is unchanged. The ORDER expands outward from the coverage-clip estimate (ocx, ocy, oside)
    when given (its pitch oside/M and center seed the most-likely hypotheses, drip-corrected),
    else from the rect itself. The symbol SIZE is tried FFT-estimate-first (the measured pitch
    picks the likely M), then by closeness to that estimate with ties broken common-first (SIZES
    order). Only iteration order — hence early-exit speed — differs. ECC-validated -> never
    mis-reads."""
    ocx = cx if ocx is None else ocx
    ocy = cy if ocy is None else ocy
    pitch = _fft_pitch(gray, oside / (max(SIZES) + 1), oside / (min(SIZES) - 1)) if oside else None
    sizes = sorted(SIZES, key=lambda M: abs(M - oside / pitch)) if pitch else SIZES
    for M in sizes:
        cell0 = (oside / M) if oside else (te / M)
        for cell in _outward(np.arange(te / (M + 3), te / (M - 1), 0.5), cell0):
            for ddeg in _outward(np.arange(-3, 3.01, 1.0), 0.0):
                base = np.arange(-1.5, 1.51, 0.375) * cell
                offs_x = _outward(base, ocx - cx)
                offs_y = _outward(base, ocy - cy)
                for dcx in offs_x:
                    for dcy in offs_y:
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
    for cx, cy, te, ang, ocx, ocy, oside in regions:
        p, reg = _brute_region(gray, cx, cy, te, ang, ocx, ocy, oside)
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


def recover(gray):
    """Reader fallback shim over the unified pipeline: returns (payload, quad) of the first
    DataMatrix, or (None, None). ECC-validated."""
    dm, _ = _collect(gray, first_only=True)
    return (dm[0][0], dm[0][1]) if dm else (None, None)


def _quad_center(quad):
    return float(np.asarray(quad)[:, 0].mean()), float(np.asarray(quad)[:, 1].mean())


def _pos_quad(p):
    return np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                     [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]],
                    np.float32)


def _collect(gray, first_only=False, fallback=True):
    """The one read pipeline. Returns (dm, other): lists of (payload, quad, format, stage),
    quads in ORIGINAL image coords. `first_only` stops at the first DataMatrix (the read()
    fast path). Passes, cheapest-first, with the cascade BEFORE the expensive repair so a
    faint code (which the cascade decodes) never wastes a failed reconstruction:
      1 raw multi-decode (all 2D) on the full frame
      2 detector regions -> format gate (cheap); undecoded regions deferred
      3 (conditional) full-frame preprocessing cascade — runs only if a region went
        undecoded or nothing was found — the safety net for faint codes no detector localizes
      4 grid-repair the regions still undecoded (broken-border codes the cascade missed)."""
    from .detect import format_gate, _2D
    from .preprocess import STAGES, STAGE_SCALE
    dm, other, seen = [], [], []

    def _take(payload, quad, fmt, stage):
        c = (float(quad[:, 0].mean()), float(quad[:, 1].mean()))
        side = float(np.linalg.norm(quad[0] - quad[1]))
        if any(abs(c[0] - sx) < 0.6 * side and abs(c[1] - sy) < 0.6 * side for sx, sy in seen):
            return False
        dm.append((payload, quad, fmt, stage))
        seen.append(c)
        return True

    # Pass 1: every directly-decodable 2D code on the full frame.
    for r in zxingcpp.read_barcodes(np.ascontiguousarray(gray), formats=_2D):
        q = _pos_quad(r.position)
        if r.format.name == "DataMatrix":
            if _take(r.bytes, q, "DataMatrix", "raw") and first_only:
                return dm, other
        else:
            other.append((r.bytes, q, r.format.name, "raw"))
    if not fallback:
        return dm, other

    # Pass 2: detector regions -> gate (cheap). Undecoded regions deferred to Pass 3/4.
    det = _detector()
    cands = (det.detect(gray) if det is not None
             else [(cx, cy, s, 0.0) for cx, cy, s, _ in propose(gray)])
    undec = []
    for cx, cy, size, *_ in cands:
        if any(abs(cx - sx) < 0.5 * size and abs(cy - sy) < 0.5 * size for sx, sy in seen):
            continue
        up, tf = _normalize(gray, cx, cy, size)
        if up is None:
            continue
        payload, fmt, pos = format_gate(up)
        if fmt == "DataMatrix":
            if _take(payload, _unmap_quad(pos, tf), "DataMatrix", "gate") and first_only:
                return dm, other
        elif fmt is not None:
            other.append((payload, _unmap_quad(pos, tf), fmt, "detector"))
        else:
            undec.append((cx, cy, size, up, tf))

    # Pass 3 (conditional): full-frame cascade for faint codes. Skip the costly u4 stages
    # when every localized region is comfortably oversampled (run them only for small/dense
    # regions, or when we have no localization at all — the blind safety net).
    if undec or not dm:
        run_u4 = _needs_u4([size for _, _, size, _, _ in undec])
        for name, transform in STAGES:
            if not run_u4 and name.startswith("thick_u4"):
                continue
            try:
                out = transform(gray)
            except cv2.error:
                continue
            for r in zxingcpp.read_barcodes(np.ascontiguousarray(out), formats=_DM):
                q = _pos_quad(r.position) / STAGE_SCALE[name]
                if _take(r.bytes, q, "DataMatrix", name) and first_only:
                    return dm, other

    # Pass 4: repair the still-undecoded regions (broken-border codes the cascade missed).
    for cx, cy, size, up, tf in undec:
        if any(abs(cx - sx) < 0.6 * size and abs(cy - sy) < 0.6 * size for sx, sy in seen):
            continue                          # the cascade already got this region
        pl, reg = decode_auto(up)
        if pl is not None:
            if _take(pl, _unmap_quad(_square_quad(*reg), tf), "DataMatrix", "autoreg") and first_only:
                return dm, other
    return dm, other
