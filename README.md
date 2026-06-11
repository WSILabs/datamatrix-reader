# datamatrix-reader

![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)
![Decoder](https://img.shields.io/badge/decoder-zxing--cpp-orange?style=flat-square)
![Detector](https://img.shields.io/badge/detector-YOLOv8%20%C2%B7%20ONNX-9cf?style=flat-square)
[![Tests](https://github.com/WSILabs/datamatrix-reader/actions/workflows/tests.yml/badge.svg)](https://github.com/WSILabs/datamatrix-reader/actions/workflows/tests.yml)

An **adaptive, source-agnostic DataMatrix reader** designed and tested on pathology slide labels — built on **zxing-cpp**, and optimized for the typical DataMatrix on a pathology slide imaged with a whole-slide scanner. It has been benchmarked on real clinical slides against libdmtx, raw (unassisted) zxing-cpp, and the commercial Dynamsoft Barcode Reader:

| decoder | reads | rate |
|---|---:|---:|
| **this library** (full reader) | **404 / 404** | **100%** |
| Dynamsoft (commercial reader) ¹ | ~397 / 404 | 98.3% |
| zxing-cpp (raw decode) | 350 / 404 | 86.6% |
| libdmtx (raw decode) ² | ~292 / 407 | ~72% |

*Real Grundium WSI slide labels (PHI — not shipped). ¹ Dynamsoft Barcode Reader 30-day trial, used only as a ceiling benchmark (not a dependency); measured before this library's finder/timing repair existed. ² earlier run on a slightly different basis (n=407, decode-hits rather than ground-truth-correct).*

The decisive gap is the **broken-finder / damaged-timing tail**: this library reconstructs the canonical finder ("L") and timing pattern from the intact data modules and hands the clean symbol to zxing's Reed–Solomon stage — reading codes that even the commercial decoder misses, and never guessing a payload (every decode is ECC-validated).

Note that this library is optimized for typical WSI-imaged slide labels: well- and evenly-lit captures, roughly **1200×850 px** label crops, near-cardinal orientation (any 90° rotation; in-plane skew within ~±15°), and reasonably-sized modules (**≳5 px each** — ~9 px/module on the WSI corpus). It has been hardened for the real defects whole-slide scanners actually produce: faint or over-inked print, horizontal line-printing defects, glare, and especially the damaged finder ("L") and timing patterns that zxing alone struggles with. It is not optimized for oblique, heavily skewed or perspective-warped, heavily obscured, or poorly-lit barcodes.

It does two jobs:

1. **Single-code decode** — read the one DataMatrix on a slide label (the common case).
2. **DataMatrix engine** — inside a larger label reader: return *every* DataMatrix
   (with location), surface QR/Aztec as tagged hints, and stay in its lane (no OCR /
   1D / layout — that's the host's job).

Every decode is **ECC200/Reed–Solomon validated**, so a bad fit fails closed — the
reader never mis-reads.

## How it works

A miss-driven ladder: cheap things first, expensive recovery only when needed.

```mermaid
flowchart TD
    img(["image"]) --> raw["zxing raw decode"]
    raw -->|"miss"| yolo["YOLO detector<br/>localizes the code(s)"]
    yolo --> gate{"format-gate<br/>zxing on tight crop"}
    gate -->|"undecoded / faint"| casc["ink-thickening cascade<br/>CLAHE+upscale → Otsu → grow ink → Sauvola"]
    casc -->|"broken finder / timing"| rep["registration repair<br/>sample grid → repaint L/timing → zxing ECC"]

    raw -->|"hit · p50 ~4 ms"| done(["✓ decoded · ECC-validated"])
    gate -->|"DataMatrix"| done
    casc -->|"hit"| done
    rep -->|"recovered"| done
    gate -->|"QR / Aztec"| hint(["tagged non-DM hint"])
    rep -->|"unrecoverable"| none(["✗ no decode"])

    classDef ok fill:#1f883d,stroke:#1a7f37,color:#ffffff
    classDef tag fill:#bf8700,stroke:#9a6700,color:#ffffff
    classDef bad fill:#cf222e,stroke:#a40e26,color:#ffffff
    class done ok
    class hint tag
    class none bad
```

- **zxing-first.** On real captures a modern decoder beats hand-tuned preprocessing;
  preprocessing only runs on a miss, so the fast path stays a few milliseconds.
- **Ink-thickening cascade** (`preprocess.py`) recovers faint / poorly-printed codes by
  progressively laying down ink at escalating strength. The costly 4×-upscale stages are
  gated on measured pixels-per-module, so well-sampled codes skip them.
- **Registration repair** (`register.py`) handles broken *borders*: when the finder L or
  timing is damaged but the data modules survive, it reconstructs the module grid, repaints
  the canonical finder/timing, and lets zxing do the ECC — recovering codes a decoder alone
  can't, without ever guessing a payload.
- **Learned detector** (`detect.py`): a YOLOv8-nano ONNX model (run via onnxruntime, no
  torch at inference) localizes DataMatrix and rejects look-alikes (QR/Aztec/cassette mesh).
  It **ships in the wheel**; if it or onnxruntime is absent, the reader falls back to a
  classical texture/gradient proposer (`locate.py`).

## Install

The learned detector ships with the package and its runtime (onnxruntime) is a core
dependency, so a plain install gives you the full reader — no extras required.

```bash
# pip — from source, or straight from git
pip install -e .
pip install "datamatrix-reader @ git+https://github.com/WSILabs/datamatrix-reader"

# uv — drop-in, much faster; same package and pyproject, nothing special needed
uv pip install -e .

# conda / anaconda — make the env, then pip-install into it (the standard pattern)
conda create -n dmr python=3.11 && conda activate dmr && pip install -e .
```

Core deps (numpy, opencv-python-headless, zxing-cpp, onnxruntime) install automatically.
Extras: `[yolo-train]` = ultralytics (detector training/export, dev only); `[tools]` =
pillow (GUI helper tools). (`[yolo]` is kept as a no-op alias — the detector is now core.)

## Usage

```python
import cv2
from datamatrix_reader.reader import Reader

reader = Reader()

# 1) single code — the fast path
res = reader.read(cv2.imread("label.png"))      # BGR or grayscale
if res.ok:
    print(res.payload, res.stage, res.box)      # bytes, how it was found, (x0,y0,x1,y1)

# 2) every 2D code on a mixed label
out = reader.read_all(cv2.imread("label.png"))
for c in out.datamatrix:                        # each: payload, quad (4x2), box, format, stage
    print("DM", c.payload, c.box)
for c in out.other_2d:                          # QR/Aztec as routable hints (payload may be None)
    print("hint", c.format, c.box)
```

- `read()` → `ReadResult(payload, stage, elapsed_ms, quad)` with `.ok` and `.box`
  (axis-aligned bbox derived from the quad). `stage` is how it decoded:
  `raw` · `gate` · `clahe` · `thick_u{f}_i{it}` · `sauv` · `autoreg`.
- `read_all()` → `ReadAllResult(datamatrix, other_2d, elapsed_ms)` with `.payloads`.
  Each entry is a `Code(payload, quad, format, stage)` with `.box`. Quads/boxes are in
  **original-image coordinates**.

**Validating payloads** is an application concern — the reader returns whatever decodes.
`validate.py` offers helpers you apply yourself:

```python
from datamatrix_reader.validate import RegexValidator
ok = RegexValidator(r'^[A-Z]\d{2}-\d{5}-[A-Z]\d$')
accession = res.payload if (res.ok and ok(res.payload)) else None
```

## Design principles

- **Robustness from coverage, not tuned constants.** A single pipeline bakes in
  assumptions some unseen input breaks. The ladder spans the variation instead, and every
  rung exits only on a *validated* decode — so there's little to overfit.
- **Latency bounded by construction.** The fast path is raw zxing; the cascade and repair
  run only on a miss. On the validated WSI corpus, p50 ≈ 4 ms and the worst broken-border
  label finishes in ~0.2 s.
- **Synthetic for generalization, real for confirmation.** Tune against parametric
  synthetic scenes (`synth.py`) that bracket the real failure axes (scale, rotation, ink
  gain/dropout, clutter); treat the real corpus as held-out confirmation, never the tuning
  target. Judge by the *worst* stratum, not the mean.

## Layout

```
src/datamatrix_reader/
  reader.py       public API: Reader.read / read_all -> ReadResult / ReadAllResult
  register.py     unified pipeline (_collect) + registration repair + region detectors
  preprocess.py   ink-thickening cascade stages (px/module-gated upscale)
  detect.py       YOLO detector (onnxruntime) + format gate; classical fallback
  locate.py       decoder-free region proposer (used when the model is absent)
  synth.py        parametric synthetic label scenes for generalization testing
  validate.py     optional payload validators (application-layer)
  models/dm_yolo.onnx   the shipped detector weights
tools/            validation harnesses (validate_full / read_all / synth / pathology),
                  viz_search (animate the registration search), YOLO train/export,
                  smoke_install (verify a built wheel ships + loads the model)
tests/            PHI-free unit tests
```

## Validation

```bash
pip install -e .
python -m pytest -q                       # unit tests
python -m tools.validate_full             # read() over the real WSI corpus (correctness + timing)
python -m tools.validate_read_all         # read_all() multi-code coverage
python -m tools.validate_synth            # synthetic localization + decode rate
```

The real WSI corpus lives outside the package (it's PHI and is not committed); only code and
the detector *weights* ship. Validated: **WSI 404/404, WRONG = 0**, p50 ~4 ms.
