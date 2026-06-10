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

from .preprocess import STAGES, STAGE_SCALE
from .register import recover

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
    def read_all(self, image: np.ndarray, fallback: bool = True) -> ReadAllResult:
        """All DataMatrix codes on the image (each with quad+box, original coords), plus
        non-DM 2D codes (QR/Aztec) as tagged hints. Raw multi-decode first; then the
        detector (YOLO or classical) surfaces additional/damaged codes via gate+repair."""
        from .register import decode_all, _quad_center
        t0 = time.perf_counter()
        gray = _gray(image)
        dm: list[Code] = []
        seen: list[tuple[float, float]] = []
        other: list[Code] = []
        # Pass 1: every directly-decodable 2D code on the full frame (DM + QR + Aztec).
        for r in zxingcpp.read_barcodes(np.ascontiguousarray(gray), formats=_2D):
            q = _pos_quad(r.position)
            fmt = r.format.name
            if fmt == "DataMatrix":
                dm.append(Code(r.bytes, q, "DataMatrix", "raw"))
                seen.append(_quad_center(q))
            else:
                other.append(Code(r.bytes, q, fmt, "raw"))
        # Pass 2: detector regions -> gate/repair for codes raw missed (dedup vs Pass 1).
        if fallback:
            ddm, doth = decode_all(gray)
            for pl, q, fmt, st in ddm:
                cx, cy = _quad_center(q)
                side = float(np.linalg.norm(q[0] - q[1]))
                if any(abs(cx - sx) < 0.6 * side and abs(cy - sy) < 0.6 * side for sx, sy in seen):
                    continue
                dm.append(Code(pl, q, fmt, st))
                seen.append((cx, cy))
            for pl, q, fmt, st in doth:
                other.append(Code(pl, q, fmt, st))
            # Pass 3: full-frame preprocessing cascade — same as read()'s ladder — for any
            # DataMatrix that the raw scan and the detector both missed (faint ink, no crop).
            for name, transform in STAGES:
                try:
                    transformed = transform(gray)
                except cv2.error:
                    continue
                scale = STAGE_SCALE[name]
                for r in zxingcpp.read_barcodes(np.ascontiguousarray(transformed), formats=_DM):
                    q = _pos_quad(r.position) / scale
                    cx, cy = _quad_center(q)
                    side = float(np.linalg.norm(q[0] - q[1]))
                    if any(abs(cx - sx) < 0.6 * side and abs(cy - sy) < 0.6 * side
                           for sx, sy in seen):
                        continue
                    dm.append(Code(r.bytes, q, "DataMatrix", name))
                    seen.append((cx, cy))
        return ReadAllResult(dm, other, (time.perf_counter() - t0) * 1000)

    def read(self, image: np.ndarray, budget_ms: float = 250.0,
             fallback: bool = True) -> ReadResult:
        t0 = time.perf_counter()
        gray = _gray(image)
        payload, quad = _zxing_pos(gray)
        stage = "raw" if payload is not None else None
        if payload is None:
            for name, transform in STAGES:
                try:
                    cand, qpos = _zxing_pos(transform(gray))
                except cv2.error:
                    continue
                if cand is not None:
                    payload, stage = cand, name
                    quad = qpos / STAGE_SCALE[name] if qpos is not None else None  # stage upscales -> back to original
                    break
        if payload is None and fallback:
            cand, qquad = recover(gray)
            if cand is not None:
                payload, stage, quad = cand, "autoreg", qquad
        return ReadResult(payload, stage, (time.perf_counter() - t0) * 1000, quad=quad)
