"""Public API: Reader.read(image, budget_ms).

zxing-cpp decodes the grayscale image directly (stage "raw"). On a miss, the Reader
runs an ordered ladder of full-frame preprocessing stages (preprocess.STAGES) that
progressively thicken faint ink until the code decodes — recovering poorly-printed
codes (real WSI: 0.926 -> 0.983, validated on that corpus; the stage params may need
re-checking on fresh captures). If the whole cascade still misses, a final
finder-registration fallback (register.recover) localizes the code, repaints the
canonical finder/timing, and decodes — recovering broken-border codes the cascade can't
(WSI: 0.983 -> 1.000, ECC-validated so it never mis-reads). Stages and the fallback run
ONLY on a miss, so p50 stays ~3 ms. `budget_ms` is accepted for call-site compatibility
but IGNORED.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
_2D = (zxingcpp.BarcodeFormat.DataMatrix, zxingcpp.BarcodeFormat.QRCode,
       zxingcpp.BarcodeFormat.Aztec)


@dataclass
class ReadResult:
    payload: bytes | None
    # "raw" | "clahe" | "thick_u{f}_i{it}" | "sauv" | "autoreg" | None
    stage: str | None
    elapsed_ms: float
    quad: np.ndarray | None = None      # (4,2) corners in ORIGINAL image coords, or None

    @property
    def ok(self) -> bool:
        return self.payload is not None

    @property
    def box(self) -> tuple[float, float, float, float] | None:
        """Axis-aligned bounding box (x0, y0, x1, y1) in original-image coords, derived
        from `quad` (None if undecoded). The quad carries orientation; box is the rect."""
        if self.quad is None:
            return None
        xs, ys = self.quad[:, 0], self.quad[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _zxing(gray: np.ndarray) -> bytes | None:
    res = zxingcpp.read_barcodes(np.ascontiguousarray(gray), formats=_DM)
    return res[0].bytes if res else None


def _pos_quad(p) -> np.ndarray:
    return np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                     [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]],
                    np.float32)


def _zxing_pos(gray: np.ndarray):
    """(payload_bytes, (4,2) corner array) or (None, None)."""
    res = zxingcpp.read_barcodes(np.ascontiguousarray(gray), formats=_DM)
    if not res:
        return None, None
    return res[0].bytes, _pos_quad(res[0].position)


@dataclass
class Code:
    """One found 2D code. `payload` is the decode (None only for an undecoded hint);
    `quad` (4,2) and `box` are ORIGINAL-image coords; `format` is 'DataMatrix'/'QRCode'/
    'Aztec'; `stage` is how it was found ('raw'|'gate'|'autoreg'|'detector')."""
    payload: bytes | None
    quad: np.ndarray
    format: str
    stage: str

    @property
    def box(self) -> tuple[float, float, float, float]:
        xs, ys = self.quad[:, 0], self.quad[:, 1]
        return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


@dataclass
class ReadAllResult:
    datamatrix: list[Code]      # every DataMatrix found
    other_2d: list[Code]        # non-DM 2D codes (QR/Aztec) as routable hints
    elapsed_ms: float

    @property
    def payloads(self) -> list[bytes]:
        return [c.payload for c in self.datamatrix]


class Reader:
    def read(self, image: np.ndarray, budget_ms: float = 250.0,
             fallback: bool = True) -> ReadResult:
        from .register import _collect
        t0 = time.perf_counter()
        dm, _ = _collect(_gray(image), first_only=True, fallback=fallback)
        ms = (time.perf_counter() - t0) * 1000
        if dm:
            payload, quad, _fmt, stage = dm[0]
            return ReadResult(payload, stage, ms, quad=quad)
        return ReadResult(None, None, ms)

    def read_all(self, image: np.ndarray, fallback: bool = True) -> ReadAllResult:
        from .register import _collect
        t0 = time.perf_counter()
        dm, other = _collect(_gray(image), first_only=False, fallback=fallback)
        ms = (time.perf_counter() - t0) * 1000
        return ReadAllResult([Code(*d) for d in dm], [Code(*o) for o in other], ms)
