"""Fast, decoder-free localization.

The reason libdmtx can take *seconds* is its Hough-style region search over a
full high-res image, especially on misses. We refuse to pay that repeatedly:
locate the code ONCE here, cheaply, on a downscaled gradient map, rectify a
small crop, and let the cascade run its many attempts on that small ROI where
libdmtx's own search is near-instant. Localize-once, decode-many.

This stage makes no decode decision, so it cannot mis-route; the cascade's
valid-decode gate is the only arbiter. We return an *ordered* list of
candidates and always append a downscaled whole-frame fallback.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    crop: np.ndarray      # rectified grayscale ROI, quiet-zone padded
    score: float          # higher = more code-like
    source: str           # "blob" | "fallback"


def _downscale(gray: np.ndarray, long_side: int = 900) -> tuple[np.ndarray, float]:
    h, w = gray.shape
    s = long_side / max(h, w)
    if s >= 1.0:
        return gray, 1.0
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return small, s


def _quiet_pad(crop: np.ndarray, modules_guess: int = 4) -> np.ndarray:
    """Pad a synthetic quiet zone. Direct-print labels crowd text against the
    code; libdmtx region detection wants a clear border."""
    pad = max(8, crop.shape[0] // 12)
    border = int(np.median(crop[[0, -1], :].ravel()))  # assume border ~ background
    return cv2.copyMakeBorder(crop, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=border)


def localize(gray: np.ndarray, *, max_candidates: int = 3) -> list[Candidate]:
    """Return code-region candidates, best first, plus a whole-frame fallback."""
    small, scale = _downscale(gray)

    # Dense high-frequency texture == module grid. Gradient magnitude, threshold,
    # close into blobs, rank by squareness + fill.
    gx = cv2.Scharr(small, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(small, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blob = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands: list[Candidate] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < (small.shape[0] * small.shape[1]) * 0.0008:
            continue
        rect = cv2.minAreaRect(c)
        (cx, cy), (rw, rh), ang = rect
        if min(rw, rh) < 12:
            continue
        squareness = min(rw, rh) / max(rw, rh)          # DataMatrix ~ square
        fill = area / (rw * rh + 1e-6)
        if squareness < 0.45:
            continue
        score = squareness * fill

        # rectify on full-res coordinates
        box = cv2.boxPoints(rect) / scale
        full_rect = (tuple(box.mean(0)),
                     (max(rw, rh) / scale * 1.15, max(rw, rh) / scale * 1.15),
                     ang)
        crop = _rectify(gray, full_rect)
        if crop is None or crop.size == 0:
            continue
        cands.append(Candidate(crop=_quiet_pad(crop), score=score, source="blob"))

    cands.sort(key=lambda c: c.score, reverse=True)
    cands = cands[:max_candidates]

    # Always include a whole-frame fallback (downscaled to keep libdmtx fast).
    fb, _ = _downscale(gray, long_side=1400)
    cands.append(Candidate(crop=fb, score=0.0, source="fallback"))
    return cands


def _rectify(gray: np.ndarray, rect) -> np.ndarray | None:
    (cx, cy), (rw, rh), ang = rect
    rw, rh = int(round(rw)), int(round(rh))
    if rw < 4 or rh < 4:
        return None
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return cv2.getRectSubPix(rot, (rw, rh), (cx, cy))
