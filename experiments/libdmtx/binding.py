"""Thin, typed wrapper over the compiled libdmtx shim.

`decode_staged` is the one primitive everything else is built on. It reports
*where* libdmtx failed, the region geometry, and respects a hard timeout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._dmtx import ffi, lib  # type: ignore


@dataclass(frozen=True)
class StageResult:
    found: bool          # region located
    decoded: bool        # region located AND matrix decoded
    payload: bytes | None
    symbol_rows: int     # 0 if not found
    symbol_cols: int
    bbox: tuple[int, int, int, int] | None   # (xmin, ymin, xmax, ymax) in px
    polarity: int        # +1 dark-on-light, -1 light-on-dark, 0 unknown

    @property
    def module_px(self) -> float | None:
        """Estimated pixels-per-module from region bbox + symbol size.

        The single most diagnostic number per image. Works even on a decode
        failure, as long as the region was *found*.
        """
        if not self.found or self.bbox is None or self.symbol_cols <= 0:
            return None
        x0, y0, x1, y1 = self.bbox
        span = max(abs(x1 - x0), abs(y1 - y0))
        n = max(self.symbol_rows, self.symbol_cols)
        return span / n if n else None


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("decode_staged expects a single-channel uint8 image")
    return np.ascontiguousarray(img, dtype=np.uint8)


def decode_staged(gray: np.ndarray, *, timeout_ms: int = 60,
                  edge_thresh: int = 0, out_cap: int = 256) -> StageResult:
    """Run the two libdmtx stages on a grayscale image with a hard timeout.

    timeout_ms bounds the region search. Keep it small (tens of ms) per
    cascade rung; the orchestrator sums these into the global budget.
    edge_thresh=0 leaves the libdmtx default (do not tune it to a corpus).
    """
    g = _as_gray_u8(gray)
    h, w = g.shape
    res = ffi.new("StageResult *")
    out = ffi.new("unsigned char[]", out_cap)
    rc = lib.dtmx_decode_staged(
        ffi.cast("unsigned char *", g.ctypes.data),
        w, h, timeout_ms, edge_thresh, out, out_cap, res,
    )
    if rc != 0:
        raise RuntimeError(f"dtmx_decode_staged failed (rc={rc})")
    payload = bytes(ffi.buffer(out, res.data_len)) if res.decoded else None
    # libdmtx uses a y-up convention; flip y into numpy/OpenCV (y-down) image
    # coords so bbox is usable for cropping/overlays. (min/max swap under flip.)
    # module_px is unaffected — it only uses span magnitude.
    bbox = ((res.bmin_x, h - 1 - res.bmax_y, res.bmax_x, h - 1 - res.bmin_y)
            if res.found else None)
    return StageResult(
        found=bool(res.found), decoded=bool(res.decoded), payload=payload,
        symbol_rows=res.symbol_rows, symbol_cols=res.symbol_cols,
        bbox=bbox, polarity=res.polarity,
    )


def encode(payload: bytes, *, module_size: int = 1, margin: int = 2) -> np.ndarray:
    """Render a payload to a clean grayscale module bitmap (0/255)."""
    data = ffi.new("unsigned char[]", payload)
    cap = 4096 * 4096
    out = ffi.new("unsigned char[]", cap)
    ow, oh = ffi.new("int *"), ffi.new("int *")
    rc = lib.dtmx_encode(data, len(payload), module_size, margin, out, cap, ow, oh)
    if rc != 0:
        raise RuntimeError(f"dtmx_encode failed (rc={rc})")
    w, h = ow[0], oh[0]
    return np.frombuffer(ffi.buffer(out, w * h), dtype=np.uint8).reshape(h, w).copy()
