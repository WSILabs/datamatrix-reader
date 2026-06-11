"""Scale/position-robust localization: propose candidate square DataMatrix regions
anywhere on a label, at any bounded scale, via a texture-density image pyramid.

`propose(gray) -> [(cx, cy, size, angle), ...]` ranked best-first (most code-like first),
in NATIVE pixel coordinates. The repair core (register.py) refines + decodes each.
"""
from __future__ import annotations

import cv2
import numpy as np

# pyramid scales: a code's native cell of ~7-35px lands near the canonical ~10px at
# some level. Tune if the scale range widens. (See spec: bounded ~0.3-1.5x of canonical.)
PYRAMID_SCALES = (1.0, 0.7, 0.5, 0.35)
_MIN_SIDE = 40          # reject sub-40px blobs at a level
_MAX_AR = 1.3           # square-ish
_MIN_FILL = 0.4
# Fine morphology pass at scale=1.0: smaller kernels isolate code sub-blobs in busy labels
# where the code texture merges into a larger blob under the coarse (7,21) pass.
_FINE_OPEN = 5    # px kernel side
_FINE_CLOSE = 15  # px kernel side


def _kernel(n):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def _density(gray):
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (13, 13))
    var = cv2.boxFilter(g * g, -1, (13, 13)) - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def _level_candidates(gray, scale, open_k=7, close_k=21):
    lvl = (gray if scale == 1.0 else
           cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    dens = _density(lvl)
    dn = (dens / (dens.max() + 1e-9) * 255).astype(np.uint8)
    th = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, _kernel(open_k))    # drop thin text
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, _kernel(close_k))  # fill module grid
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < _MIN_SIDE or h < _MIN_SIDE:
            continue
        if max(w, h) / max(1.0, min(w, h)) > _MAX_AR:
            continue
        fill = cv2.contourArea(c) / max(1.0, w * h)
        if fill < _MIN_FILL:
            continue
        m = np.zeros(lvl.shape, np.uint8); cv2.drawContours(m, [c], -1, 1, -1)
        strength = float(fill * dens[m > 0].mean())          # density-weighted squareness
        out.append((cx / scale, cy / scale, (w + h) / 2.0 / scale, ang, strength))
    return out


def _dedup(cands):
    """Merge proposals whose centers are within half the smaller size (same code found at
    adjacent pyramid levels); keep the strongest."""
    cands = sorted(cands, key=lambda c: -c[4])
    kept = []
    for cx, cy, size, ang, strength in cands:
        if any(abs(cx - kx) < 0.5 * min(size, ks) and abs(cy - ky) < 0.5 * min(size, ks)
               for kx, ky, ks, _, _ in kept):
            continue
        kept.append((cx, cy, size, ang, strength))
    return kept


def propose(gray):
    cands = []
    for sc in PYRAMID_SCALES:
        cands.extend(_level_candidates(gray, sc))
    # Fine-morphology pass at native scale: catches sub-blobs in busy labels where the
    # code texture merges with surrounding text under the coarse (7,21) pass.
    cands.extend(_level_candidates(gray, 1.0, open_k=_FINE_OPEN, close_k=_FINE_CLOSE))  # (recomputes the native-scale density; negligible on this fallback-only path)
    return [(cx, cy, size, ang) for cx, cy, size, ang, _ in _dedup(cands)]
