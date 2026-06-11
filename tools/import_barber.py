"""Adapter: BarBeR dataset -> a datamatrix_reader --corpus directory of DataMatrix codes.

BarBeR (https://universe... / VGG VIA annotations) bundles 12 public barcode
datasets. We keep only **DataMatrix** regions that carry a decoded `String`, so
the result is a real-capture confirmation corpus with payload ground truth that
drops straight onto `bench.harness --corpus`.

Selection rule: an image is included iff its DataMatrix regions resolve to a
*single* distinct payload (the common case: one code, or several shots of the
same code). Images with multiple *different* DataMatrix payloads are dropped,
because the harness keys one truth per filename — the count is logged, not
silently swallowed.

Reads images and annotations directly from the zip; nothing else needs to be
extracted first.

    python -m tools.import_barber \
        --zip corpus/public/BarBeR_Dataset.zip \
        --out corpus/barber
"""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

ANNOT_DIR = "BarBeR - Dataset/Annotations/VIA/"
IMAGES_DIR = "BarBeR - Dataset/dataset/images/"


def _valid(s) -> bool:
    return s is not None and str(s).strip() not in ("", "-1")


def collect_datamatrix(zf: zipfile.ZipFile) -> dict[str, dict]:
    """filename -> {payloads:set, ppes:list} for every image with a DataMatrix."""
    by_file: dict[str, dict] = defaultdict(lambda: {"payloads": set(), "ppes": []})
    for name in zf.namelist():
        if not (name.startswith(ANNOT_DIR) and name.endswith(".json")):
            continue
        data = json.loads(zf.read(name))
        meta = data.get("_via_img_metadata", data)
        for entry in meta.values():
            if not isinstance(entry, dict):
                continue
            fn = entry.get("filename")
            for r in entry.get("regions", []):
                ra = r.get("region_attributes", {})
                if str(ra.get("Type", "")).lower() != "datamatrix":
                    continue
                if not _valid(ra.get("String")):
                    continue
                by_file[fn]["payloads"].add(str(ra["String"]))
                try:
                    ppe = float(ra.get("PPE"))
                    if ppe > 0:
                        by_file[fn]["ppes"].append(ppe)
                except (TypeError, ValueError):
                    pass
    return by_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="corpus/public/BarBeR_Dataset.zip")
    ap.add_argument("--out", default="corpus/barber")
    args = ap.parse_args()

    out = Path(args.out)
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as zf:
        by_file = collect_datamatrix(zf)
        # map basename -> zip entry for fast extraction
        img_entries = {Path(n).name: n for n in zf.namelist()
                       if n.startswith(IMAGES_DIR) and not n.endswith("/")}

        kept, dropped_multi, missing = [], [], []
        for fn, info in sorted(by_file.items()):
            if len(info["payloads"]) != 1:
                dropped_multi.append(fn)
                continue
            entry = img_entries.get(fn)
            if entry is None:
                missing.append(fn)
                continue
            (img_out / fn).write_bytes(zf.read(entry))
            kept.append((fn, next(iter(info["payloads"])), info["ppes"]))

    labels = out / "labels.csv"
    with labels.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "payload"])
        for fn, payload, _ in kept:
            w.writerow([fn, payload])

    ppe_n = sum(1 for _, _, p in kept if p)
    print(f"DataMatrix images found:        {len(by_file)}")
    print(f"  kept (single payload):        {len(kept)} -> {img_out}/")
    print(f"  dropped (multi-payload):      {len(dropped_multi)}")
    print(f"  missing image in zip:         {len(missing)}")
    print(f"  with measured PPE:            {ppe_n}/{len(kept)}")
    print(f"labels.csv written:             {labels}")
    if dropped_multi:
        print(f"  dropped files: {', '.join(dropped_multi[:10])}"
              + (" ..." if len(dropped_multi) > 10 else ""))


if __name__ == "__main__":
    main()
