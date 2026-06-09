"""Public API: Reader.read(image, budget_ms).

zxing-cpp decodes the grayscale image directly (stage "raw"). On a miss, the Reader
runs an ordered ladder of full-frame preprocessing stages (preprocess.STAGES) that
progressively thicken faint ink until the code decodes — recovering poorly-printed
codes (real WSI: 0.926 -> 0.983, validated on that corpus; the stage params may need
re-checking on fresh captures). Stages run ONLY on a miss, so p50 stays ~3 ms.
`budget_ms` is accepted for call-site compatibility but IGNORED.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import zxingcpp

from .preprocess import STAGES

_DM = zxingcpp.BarcodeFormat.DataMatrix


@dataclass
class ReadResult:
    payload: bytes | None
    stage: str | None          # "raw" | "clahe" | "thick_u{f}_i{it}" | "sauv" | None
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return self.payload is not None


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _zxing(gray: np.ndarray) -> bytes | None:
    res = zxingcpp.read_barcodes(np.ascontiguousarray(gray), formats=_DM)
    return res[0].bytes if res else None


class Reader:
    def read(self, image: np.ndarray, budget_ms: float = 250.0) -> ReadResult:
        t0 = time.perf_counter()
        gray = _gray(image)
        payload = _zxing(gray)
        stage = "raw" if payload is not None else None
        if payload is None:
            for name, transform in STAGES:
                try:
                    cand = _zxing(transform(gray))
                except cv2.error:
                    continue           # degenerate image for this transform -> miss
                if cand is not None:
                    payload, stage = cand, name
                    break
        return ReadResult(payload, stage, (time.perf_counter() - t0) * 1000)
