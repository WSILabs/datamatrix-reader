"""The decode cascade.

Robustness comes from coverage of a strategy *space*, not a point in
parameter space: try spanning representations, return on the first
validated decode. Because the exit is gated on a valid decode there is almost
nothing to overfit — an input is either covered by some rung or it isn't.

Two hard latency rules:
  * every rung's libdmtx call carries a small per-rung timeout;
  * the orchestrator stops the moment a global deadline is hit.
So total time is bounded by construction, and clean codes exit in rung 0
paying almost nothing.

Rung parameters are keyed to the *measured* module pitch, so each rung is
self-calibrating rather than tuned to a printer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import adapt
from .binding import decode_staged
from .validate import AcceptAny, Validator

Transform = Callable[[np.ndarray, float | None], np.ndarray]


def _odd(n: int) -> int:
    n = int(n)
    return n + 1 if n % 2 == 0 else max(3, n)


# --- self-calibrating transforms (module_px-keyed) ----------------------------

def t_raw(g: np.ndarray, mpx: float | None) -> np.ndarray:
    return g

def t_otsu(g: np.ndarray, mpx: float | None) -> np.ndarray:
    _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return o

def t_adaptive(g: np.ndarray, mpx: float | None) -> np.ndarray:
    win = _odd((mpx or 8) * 3)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, win, 5)

def t_scale_adaptive(g: np.ndarray, mpx: float | None) -> np.ndarray:
    g = adapt.normalize_scale(g, mpx, target_px=8.0)
    return t_adaptive(g, 8.0)

def t_close(g: np.ndarray, mpx: float | None) -> np.ndarray:
    """Fill print dropouts (under-inking) at module scale."""
    g = adapt.normalize_scale(g, mpx, target_px=8.0)
    b = t_adaptive(g, 8.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)

def t_open(g: np.ndarray, mpx: float | None) -> np.ndarray:
    """Remove ink-spread satellites / specks at module scale."""
    g = adapt.normalize_scale(g, mpx, target_px=8.0)
    b = t_adaptive(g, 8.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(b, cv2.MORPH_OPEN, k)

def t_unsharp(g: np.ndarray, mpx: float | None) -> np.ndarray:
    """Counter blur / ink bleed, then threshold."""
    g = adapt.normalize_scale(g, mpx, target_px=8.0)
    blur = cv2.GaussianBlur(g, (0, 0), max(1.0, (mpx or 8) / 6))
    sharp = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
    return t_adaptive(sharp, 8.0)


@dataclass(frozen=True)
class Rung:
    name: str
    fn: Transform


DEFAULT_LADDER: list[Rung] = [
    Rung("raw", t_raw),                  # clean codes exit here, ~ms
    Rung("otsu", t_otsu),
    Rung("adaptive", t_adaptive),
    Rung("scale_adaptive", t_scale_adaptive),
    Rung("close", t_close),
    Rung("open", t_open),
    Rung("unsharp", t_unsharp),
]


@dataclass
class RungTrace:
    name: str
    found: bool
    decoded: bool
    valid: bool
    module_px: float | None
    ms: float


@dataclass
class CascadeResult:
    payload: bytes | None
    rung: str | None
    trace: list[RungTrace] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.payload is not None


def run_cascade(crop: np.ndarray, *, validator: Validator = AcceptAny(),
                ladder: list[Rung] = DEFAULT_LADDER,
                rung_timeout_ms: int = 35,
                deadline: float | None = None) -> CascadeResult:
    """Walk the ladder on one candidate crop until a validated decode or the
    deadline. `deadline` is an absolute time.perf_counter() value."""
    trace: list[RungTrace] = []

    # One cheap probe to estimate module pitch, feeding the scale-keyed rungs.
    probe = decode_staged(crop, timeout_ms=rung_timeout_ms)
    mpx = probe.module_px
    if probe.decoded and validator(probe.payload):
        trace.append(RungTrace("probe", True, True, True, mpx, 0.0))
        return CascadeResult(probe.payload, "probe", trace)

    for rung in ladder:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        t0 = time.perf_counter()
        try:
            img = rung.fn(crop, mpx)
            r = decode_staged(img, timeout_ms=rung_timeout_ms)
        except Exception:
            r = None
        ms = (time.perf_counter() - t0) * 1000
        if r is None:
            trace.append(RungTrace(rung.name, False, False, False, mpx, ms))
            continue
        if r.module_px:
            mpx = r.module_px  # refine estimate as we learn it
        valid = bool(r.decoded and validator(r.payload))
        trace.append(RungTrace(rung.name, r.found, r.decoded, valid, r.module_px, ms))
        if valid:
            return CascadeResult(r.payload, rung.name, trace)

    return CascadeResult(None, None, trace)
