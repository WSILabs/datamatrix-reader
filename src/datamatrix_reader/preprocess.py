"""Ordered full-frame preprocessing stages for the Reader's fallback cascade.

Progressive ink-thickening — upscale -> strong CLAHE -> Otsu -> morphological
erosion that GROWS the dark modules (lays down more ink) at escalating strength —
recovers faint / broken-finder DataMatrix codes; a final Sauvola stage catches a
couple more. Pure OpenCV/numpy; the Reader runs these only when raw zxing misses,
in order, taking the first decode.

Stage params are tuned on the wsi_labels corpus (0.926 -> 0.983, WRONG=0). The
principle (progressive ink-thickening for faint codes) generalizes; the exact
numbers may need re-validation on fresh captures.
"""
from __future__ import annotations

import cv2
import numpy as np

CLAHE_TILE = (8, 8)


def _up(g: np.ndarray, f: int) -> np.ndarray:
    return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def _clahe(g: np.ndarray, clip: float) -> np.ndarray:
    return cv2.createCLAHE(clip, CLAHE_TILE).apply(g)


def _otsu(g: np.ndarray) -> np.ndarray:
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _thicken(binary: np.ndarray, iters: int) -> np.ndarray:
    # erode shrinks bright regions -> grows the dark (ink) modules
    return cv2.erode(binary, np.ones((2, 2), np.uint8), iterations=iters)


def _unsharp(g: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(g, 2.5, cv2.GaussianBlur(g, (0, 0), 1.0), -1.5, 0)


def _sauvola(g: np.ndarray, w: int = 41, k: float = 0.15, R: float = 128.0) -> np.ndarray:
    g = g.astype(np.float32)
    m = cv2.boxFilter(g, -1, (w, w))
    s2 = cv2.boxFilter(g * g, -1, (w, w))
    sd = np.sqrt(np.maximum(s2 - m * m, 0))
    return ((g > (m * (1 + k * (sd / R - 1)))) * 255).astype(np.uint8)


def s_clahe(g: np.ndarray) -> np.ndarray:
    return _clahe(_up(g, 2), 2.0)


def _thick(g: np.ndarray, f: int, it: int) -> np.ndarray:
    return _thicken(_otsu(_clahe(_up(g, f), 4.0)), it)


def s_sauv(g: np.ndarray) -> np.ndarray:
    return _thicken(_sauvola(_unsharp(_up(g, 2))), 1)


# Ordered fallback stages: (name, transform: gray -> gray). The Reader tries them
# in order only when prior stages (incl. raw zxing) miss; first decode wins.
# clahe -> progressive ink-thickening (up2 then up4, erode 1..3 iters) -> sauvola.
STAGES = [("clahe", s_clahe)]
STAGES += [(f"thick_u{f}_i{it}", lambda g, f=f, it=it: _thick(g, f, it))
           for f in (2, 4) for it in (1, 2, 3)]
STAGES += [("sauv", s_sauv)]

# Each stage uniformly upscales by this factor (no translation), so a code's position in
# a stage's output maps to original coords by dividing by the factor.
STAGE_SCALE = {"clahe": 2}
STAGE_SCALE.update({f"thick_u{f}_i{it}": f for f in (2, 4) for it in (1, 2, 3)})
STAGE_SCALE["sauv"] = 2
