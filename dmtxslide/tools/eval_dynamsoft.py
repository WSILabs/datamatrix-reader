"""A/B the Dynamsoft Barcode Reader (commercial, 30-day trial) against the
shipped zxing cascade on the real WSI labels — the decoder-capability test.

We isolated the failure mode as faint printing that degrades the finder/timing
pattern (zxing + libdmtx both fail to even locate ~25 codes; no preprocessing
recovers them). Dynamsoft advertises damaged-symbol / low-quality recovery, so
the decisive question is: does it read the codes the zxing cascade misses,
especially the ones your labels marked broken_finder?

License (trial key from the Dynamsoft portal), in priority order:
  --license <key>  |  env DYNAMSOFT_LICENSE  |  file .dynamsoft_license (gitignored)

    python -m tools.eval_dynamsoft --corpus corpus/wsi_labels
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2

from datamatrix_reader.reader import Reader


def _license(arg: str | None) -> str:
    if arg:
        return arg
    if os.environ.get("DYNAMSOFT_LICENSE"):
        return os.environ["DYNAMSOFT_LICENSE"]
    f = Path(".dynamsoft_license")
    if f.exists():
        return f.read_text().strip()
    raise SystemExit("No license: pass --license, set DYNAMSOFT_LICENSE, or write "
                     ".dynamsoft_license")


def _load_truth(corpus: Path) -> dict[str, bytes]:
    out = {}
    with (corpus / "labels.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            out[r["file"]] = r["payload"].encode()
    return out


def _load_defects(corpus: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    p = corpus / "defects.csv"
    if p.exists():
        with p.open(newline="") as f:
            for r in csv.DictReader(f):
                out[r["file"]] = [d for d in r["defects"].split(";") if d]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--license", default=None)
    args = ap.parse_args()
    corpus = Path(args.corpus)

    import dynamsoft_barcode_reader_bundle as dbr
    code, msg = dbr.LicenseManager.init_license(_license(args.license))
    print(f"license init: code={code} ({msg})")
    if code != 0 and "successfully" not in msg.lower():
        print("  WARNING: license may be invalid; results below may be empty.")
    router = dbr.CaptureVisionRouter()
    template = dbr.EnumPresetTemplate.PT_READ_BARCODES_READ_RATE_FIRST.value

    def dyn_read(path: Path) -> bytes | None:
        res = router.capture(str(path), template)
        items = res.get_items() if res else []
        for it in items:
            b = it.get_bytes()
            if b:
                return bytes(b)
        return None

    truth = _load_truth(corpus)
    defects = _load_defects(corpus)
    reader = Reader()

    dyn_ok = zx_ok = 0
    zx_misses, dyn_gets_zx_miss = [], []
    for name, t in truth.items():
        img = cv2.imread(str(corpus / name))
        zx = reader.read(img).payload == t
        zx_ok += zx
        d = dyn_read(corpus / name) == t
        dyn_ok += d
        if not zx:
            zx_misses.append(name)
            if d:
                dyn_gets_zx_miss.append(name)

    n = len(truth)
    print(f"\n=== n={n} ===")
    print(f"zxing cascade : {zx_ok}/{n} = {zx_ok/n:.3f}")
    print(f"dynamsoft     : {dyn_ok}/{n} = {dyn_ok/n:.3f}")
    print(f"\nof the {len(zx_misses)} zxing-cascade misses, dynamsoft reads "
          f"{len(dyn_gets_zx_miss)}  -> combined ceiling "
          f"{zx_ok + len(dyn_gets_zx_miss)}/{n} = {(zx_ok + len(dyn_gets_zx_miss))/n:.3f}")

    # does dynamsoft crack the broken-finder ones? (PHI-safe: counts only)
    bf = [m for m in zx_misses if "broken_finder" in defects.get(m, [])]
    bf_got = [m for m in bf if m in dyn_gets_zx_miss]
    faint = [m for m in zx_misses if "faint" in defects.get(m, [])]
    faint_got = [m for m in faint if m in dyn_gets_zx_miss]
    print(f"\nby your defect labels (among zxing misses):")
    print(f"  broken_finder: dynamsoft reads {len(bf_got)}/{len(bf)}")
    print(f"  faint:         dynamsoft reads {len(faint_got)}/{len(faint)}")


if __name__ == "__main__":
    main()
