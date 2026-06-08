from __future__ import annotations
import csv, shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def decide(reads: dict[str, bytes | None]) -> tuple[str, list[bytes]]:
    """Classify per-decoder reads for one image.

    ("auto", [payload]) when the distinct non-None reads number exactly 1
    (all decoders that fired agree, or a single decoder fired). Otherwise
    ("queue", candidates) where candidates is the sorted distinct reads
    (empty when nothing read, >=2 on disagreement)."""
    vals = sorted({v for v in reads.values() if v is not None})
    if len(vals) == 1:
        return ("auto", vals)
    return ("queue", vals)
