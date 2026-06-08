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


def payload_to_text(b: bytes) -> str:
    try:
        return b.decode("ascii")
    except UnicodeDecodeError:
        return b.decode("latin-1")


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                labels[row["file"]] = row["payload"]
    return labels


def save_labels(path: Path, labels: dict[str, str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "payload"])
        for name in sorted(labels):
            w.writerow([name, labels[name]])
