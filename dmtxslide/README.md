# datamatrix-reader

An **adaptive, source-agnostic DataMatrix reader** for pathology slide labels —
built on **zxing-cpp**, hardened for the real, degraded codes that whole-slide
scanners (e.g. Grundium) actually capture: faint or over-inked print, glare,
half-printed timing rows, and chipped finder patterns.

It does two jobs:

1. **Single-code decode** — read the one DataMatrix on a slide label (the common case).
2. **DataMatrix engine** — inside a larger label reader: return *every* DataMatrix
   (with location), surface QR/Aztec as tagged hints, and stay in its lane (no OCR /
   1D / layout — that's the host's job).

Every decode is **ECC200/Reed–Solomon validated**, so a bad fit fails closed — the
reader never mis-reads.

## How it works

A miss-driven ladder: cheap things first, expensive recovery only when needed.

```
image ─▶ zxing raw decode ──────────────────────────────── hit ▶ done  (p50 ~4 ms)
            │ miss
            ▼
         YOLO detector localizes the code(s) ─▶ format-gate (zxing on the tight crop):
            │                                     DataMatrix ▶ done · QR/Aztec ▶ tagged hint
            │ undecoded / faint
            ▼
         ink-thickening cascade (full frame): CLAHE+upscale ▶ Otsu ▶ progressive
            │                                   erosion (grow ink) ▶ Sauvola  ─▶ hit ▶ done
            │ broken finder/timing
            ▼
         registration repair: localize ▶ sample the module grid ▶ REPAINT the canonical
            finder/timing ▶ hand the clean symbol to zxing  ─▶ hit ▶ done
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

```bash
# from source (this repo); [yolo] pulls onnxruntime for the learned detector
pip install -e ".[yolo]"

# or straight from git
pip install "datamatrix-reader[yolo] @ git+https://github.com/WSILabs/datamatrix-reader"
```

Core deps (numpy, opencv-python-headless, zxing-cpp) install automatically. Extras:
`[yolo]` = onnxruntime (runtime detector), `[yolo-train]` = ultralytics (training/export,
dev only), `[tools]` = pillow (GUI helper tools).

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
pip install -e ".[yolo]"
python -m pytest -q                       # unit tests
python -m tools.validate_full             # read() over the real WSI corpus (correctness + timing)
python -m tools.validate_read_all         # read_all() multi-code coverage
python -m tools.validate_synth            # synthetic localization + decode rate
```

The real WSI corpus lives outside the package (it's PHI and is not committed); only code and
the detector *weights* ship. Validated: **WSI 404/404, WRONG = 0**, p50 ~4 ms.
