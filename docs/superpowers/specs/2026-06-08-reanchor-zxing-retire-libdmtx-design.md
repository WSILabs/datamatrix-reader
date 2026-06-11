# Re-anchor on zxing-cpp, retire libdmtx — design

**Date:** 2026-06-08
**Status:** approved (brainstorming) → ready for implementation plan
**Supersedes the founding thesis:** "wrap libdmtx with adaptive preprocessing."

## Problem

Real-data benchmarks settled the decoder question: zxing-cpp reads the real
Grundium WSI labels at 0.87 (correct) vs preprocess+libdmtx 0.78, and ~50× faster.
Its 54 misses are poorly-printed (low-contrast / small / noisy) codes, and a
**reader-side** fallback — retry zxing on a 2×-upscaled + CLAHE image — recovers 24
of them (0.87 → 0.93), validated against ground truth on the real misses (not synth).
"Improve the printing" is never an acceptable fix; the reader must cope.

So: make zxing-cpp the sole decode **and** encode engine, delete the libdmtx native
shim and the preprocessing front-end it fed, and collapse `Reader` to a small
2-stage zxing cascade.

## Goals

1. `Reader.read()` = raw zxing, then on miss zxing-on-(upscale2×+CLAHE). ~0.93 on
   real WSI labels, p50 ~3 ms (preprocessing paid only on the ~13% hard tail).
2. Remove libdmtx entirely: the cffi shim, its build step, and the native dep.
3. Keep the synthetic benchmark working by re-encoding via zxing's writer.
4. Repurpose `compare_backends` to measure the cascade's gain (zxing-raw vs cascade).
5. Full test suite green; no new third-party library (zxing-cpp already a dep, just
   promoted from optional to core; OpenCV/numpy already core).

## Non-goals

- Recovering the ~27/404 codes that no method we tried can read (grid
  reconstruction / super-resolution — a separate, harder problem).
- Adding libdmtx fallback stages (they add a native library for +3 codes — rejected).
- Re-validating/expanding the synth degradation model (out of scope; port only).

## Architecture

zxing-cpp is the only barcode engine. The libdmtx layer (`binding`, `_build_dmtx`,
`_dmtx.*.so`) and the decoder-agnostic preprocessing front-end (`cascade`,
`localize`, `adapt`) are **deleted** (git history retains them; not archived in-tree).

```
Reader.read(image):
    gray = grayscale(image)
    hit  = zxing_decode(gray)                       # stage "raw"  (~3 ms, 0.87)
    if hit is None:
        prep = CLAHE(clip=2.0, tile=8)( upscale2x(gray) )
        hit  = zxing_decode(prep)                   # stage "clahe" (recovers +24)
    return ReadResult(payload=hit, stage=..., elapsed_ms=...)
```

## Components

### `src/datamatrix_reader/reader.py` — rewrite

- `ReadResult` dataclass: `payload: bytes | None`, `stage: str | None`
  (`"raw"` | `"clahe"` | `None`), `elapsed_ms: float`; `.ok` property = `payload is
  not None`. Drop `rung`, `candidate_idx`, `candidate_traces`.
- `class Reader`: no constructor args (drop `validator`/`ladder`/`rung_timeout_ms`).
  `read(self, image: np.ndarray, budget_ms: float = 250.0) -> ReadResult`.
  `budget_ms` is retained for call-site compatibility (bench/tests pass it) but is
  **ignored** — zxing is fast and uncancellable here. Document it as accepted-but-
  unused so callers don't break; do not add deadline logic.
- Decode helper: `zxingcpp.read_barcodes(img, formats=DataMatrix)`; return
  `res[0].bytes if res else None`. Grayscale via the existing OpenCV idiom.
- Constants at module top: `UPSCALE = 2`, `CLAHE_CLIP = 2.0`, `CLAHE_TILE = (8, 8)`
  (the validated values).

### Deletions

`src/datamatrix_reader/binding.py`, `_build_dmtx.py`, `_dmtx.cpython-312-darwin.so`,
`cascade.py`, `localize.py`, `adapt.py`. Remove their imports everywhere.

### `src/datamatrix_reader/synth.py` — port the encoder

Replace `binding.encode(payload, module_size=1, margin=2)` (returns a 1px/module
matrix the synth then scales/degrades) with a zxing-writer encoder that yields the
**same shape of input**: a 1-pixel-per-module **square** DataMatrix matrix (uint8,
0/255), with a known margin synth can control.

Porting requirements (the real work — zxing's writer differs from libdmtx's):
- zxing `create_barcode(...).to_image()` / `write_barcode_to_image` renders with a
  built-in quiet zone and may choose a **rectangular** symbol (observed 12×26 for a
  17-char payload) whereas libdmtx defaulted to square. The port must:
  1. produce **1 pixel per module** (writer scale = 1, or downsample the rendered
     bitmap back to module resolution),
  2. **strip the quiet zone** to a bare data matrix, then re-add synth's own margin,
  3. **force a square symbol** (writer option if available; else pick square sizes /
     reject rectangular) so synth's geometry axes (module_px, crowding) stay valid.
- Encapsulate as a small `synth._encode(payload) -> np.ndarray` so `render()` and the
  tests share one path.

### `tools/compare_backends.py` — repoint

Drop `fold_libdmtx_raw` and `fold_preprocess_libdmtx` (and the `binding` import).
New `FOLDS`:
- `"zxing raw"` — `zxingcpp.read_barcodes(gray)[0]` (unchanged logic).
- `"zxing cascade"` — `Reader().read(img).payload` (the new 2-stage reader).

The tool now answers "how much does the CLAHE/upscale fallback add on this corpus?"
(expected on WSI: ~0.866 raw → ~0.926 cascade). `--synth`/`--corpus`/`--pathology`
modes and `load_corpus` unchanged.

### `bench/harness.py`

`Reader(validator=AcceptAny())` → `Reader()`. `reader.read(img, budget_ms=...)`
unchanged. Remove any now-dead validator import.

### `pyproject.toml`

- `[build-system] requires`: drop `cffi`.
- `dependencies`: drop `cffi`; **add `zxing-cpp>=2.0`** (promoted from the extra);
  keep `numpy`, `opencv-python-headless`.
- Delete `[project.optional-dependencies] compare` (zxing-cpp is now core, so the
  extra is empty). Update any `pip install -e ".[compare]"` references in
  docs/HANDOFF to plain `pip install -e .`.
- Remove `[tool.setuptools.package-data] datamatrix_reader = ["*.so"]` and the
  `_build_dmtx` post-install note. Fresh clones no longer need libdmtx headers or a
  build step.

### Tests

- `tests/test_synth.py`: rides the re-encoded `synth.render()` + new `Reader`.
  Expectation holds only if zxing decodes the zxing-encoded synth codes (it must).
- `tests/test_harness.py`: `binding.encode()` symbol-size check (line ~14) repoints
  to `synth._encode()`; the `run()` smoke test uses the new `Reader`.
- `tests/test_label_gt.py`, `tests/test_compare_corpus.py`: untouched (no libdmtx).
- The synth-decode tests double as the regression that the new reader decodes clean
  codes; add (if absent) a direct reader test: a zxing-encoded known payload decodes
  back via `Reader().read()` (stage `"raw"`), and a degraded one via stage `"clahe"`.

## Data flow

`bench`/`compare_backends`/`label_gt` → `Reader.read(bgr_img)` → grayscale →
zxing → (on miss) upscale+CLAHE → zxing → `ReadResult`. Encoding path (synth/tests)
→ `synth._encode(payload)` → zxing writer → square 1px/module matrix → degraded by
synth → fed back through `Reader`.

## Error handling

- zxing returns `[]` on no-decode → `payload=None`, `stage=None`, `.ok=False`.
- Empty/garbage image → zxing returns `[]`; no crash.
- Stage 2 preprocessing failure (e.g. tiny image) is caught; treated as no-decode.
- `synth._encode` on an unencodable payload raises (tests use valid payloads).

## Testing strategy

1. Unit: `Reader` decodes a known zxing-encoded payload (stage raw); a degraded
   variant recovers via stage clahe; a blank image returns `.ok == False`.
2. `synth._encode` round-trips: encode → `Reader().read()` returns the payload;
   output is square, binary, 1px/module.
3. Full `pytest tests/` green after the port.
4. Acceptance (manual, not CI): `compare_backends --corpus corpus/wsi_labels`
   reproduces ~0.866 (raw) vs ~0.926 (cascade); p50 ~3 ms.

## Risks / notes

- **Synth encoder parity** is the main risk (square vs rectangular, quiet zone, 1px/
  module). Isolated in `synth._encode` with a round-trip test as the guard.
- `is_valid_payload`/`validate.py` accession validation is unaffected and stays a
  caller-side concern (not folded into `Reader`).
- Memories/HANDOFF referencing `binding`, the cffi shim, the y-flipped bbox, and the
  libdmtx build steps become **stale** after this lands — update them (the
  bbox-y-flip note is obsolete once `binding.py` is gone).
- zxing-cpp must expose a usable writer in the pinned version (verified: 2.x has
  `create_barcode`/`write_barcode_to_image`).
