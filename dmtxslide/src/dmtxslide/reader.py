"""Public API: Reader.read(image, budget_ms).

zxing-cpp is the decode engine. Stage "raw" reads the grayscale image directly;
on a miss, stage "clahe" retries on a 2x-upscaled, CLAHE-equalised copy — the
validated recovery for poorly-printed codes (real WSI: 0.87 -> 0.93, with the
preprocessing paid only on the hard tail). `budget_ms` is accepted for call-site
compatibility but IGNORED — zxing is fast and uncancellable here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
UPSCALE = 2
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)


@dataclass
class ReadResult:
    payload: bytes | None
    stage: str | None          # "raw" | "clahe" | None
    elapsed_ms: float

    @property
    def ok(self) -> bool:
        return self.payload is not None


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _zxing(gray: np.ndarray) -> bytes | None:
    res = zxingcpp.read_barcodes(gray, formats=_DM)
    return res[0].bytes if res else None


class Reader:
    def read(self, image: np.ndarray, budget_ms: float = 250.0) -> ReadResult:
        t0 = time.perf_counter()
        gray = _gray(image)
        payload = _zxing(gray)
        stage = "raw" if payload is not None else None
        if payload is None:
            up = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE,
                            interpolation=cv2.INTER_CUBIC)
            enhanced = cv2.createCLAHE(CLAHE_CLIP, CLAHE_TILE).apply(up)
            payload = _zxing(enhanced)
            stage = "clahe" if payload is not None else None
        return ReadResult(payload, stage, (time.perf_counter() - t0) * 1000)
