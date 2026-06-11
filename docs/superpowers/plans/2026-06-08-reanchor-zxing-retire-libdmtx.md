# Re-anchor on zxing-cpp, retire libdmtx — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make zxing-cpp the sole decode+encode engine: rewrite `Reader` as a 2-stage zxing cascade (raw → upscale2×+CLAHE on miss), delete the libdmtx shim and preprocessing front-end, port `synth` to zxing's writer, and repurpose `compare_backends`.

**Architecture:** `Reader.read()` calls zxing on the grayscale image; on a miss it retries zxing on a 2×-upscaled CLAHE-equalised copy (validated to lift real-WSI accuracy 0.87→0.93, with preprocessing paid only on the ~13% hard tail). `synth` re-encodes test codes via `zxingcpp.create_barcode(...).to_image()`. The native cffi shim (`binding`/`_build_dmtx`/`.so`) and the decoder-agnostic front-end (`cascade`/`localize`/`adapt`) are deleted.

**Tech Stack:** Python 3.12, zxing-cpp (decode+encode, promoted to a core dep), OpenCV (grayscale/upscale/CLAHE), numpy.

Spec: `docs/superpowers/specs/2026-06-08-reanchor-zxing-retire-libdmtx-design.md`

**Empirically pre-verified (so each task stays green):**
- zxing `read_barcodes` already runs try_rotate/try_downscale/try_invert by default.
- `create_barcode(payload_bytes, DataMatrix).to_image()` → 2D uint8 0/255, 1px/module; payload pool yields ≥3 distinct shapes; round-trips through `read_barcodes`.
- All synth axes (incl. module_px=2 and ink_gain=max@min-module) decode under the new reader, with **both** the old libdmtx encoder and the new zxing encoder — so no AXES recalibration, and the Task 1→2 transient is safe.

---

## File Structure

- **Rewrite** `src/datamatrix_reader/reader.py` — 2-stage zxing `Reader` + `ReadResult` (no libdmtx, no cascade/localize/adapt).
- **Create** `tests/test_reader.py` — unit tests for the new reader.
- **Modify** `bench/harness.py` — `Reader()` (drop `validator=AcceptAny()`), drop the `AcceptAny` import.
- **Modify** `src/datamatrix_reader/synth.py` — `render()` encodes via zxing instead of `binding.encode`.
- **Modify** `tests/test_harness.py` — symbol-size check uses `synth.render` instead of `binding.encode`.
- **Modify** `tools/compare_backends.py` — `FOLDS` = `zxing raw` + `zxing cascade`; drop libdmtx folds/import.
- **Delete** `src/datamatrix_reader/{binding.py,_build_dmtx.py,_dmtx.cpython-312-darwin.so,cascade.py,localize.py,adapt.py}`.
- **Modify** `pyproject.toml` — drop cffi + `.so` packaging + `[compare]`; add `zxing-cpp` as a core dep.
- **Modify** `HANDOFF.md` — install/build commands.

Run everything with `.venv/bin/python` from `/Volumes/Ext/GitHub/datamatrix-reader/datamatrix_reader`.

---

### Task 1: Rewrite `Reader` as the zxing cascade

**Files:**
- Rewrite: `src/datamatrix_reader/reader.py`
- Create: `tests/test_reader.py`
- Modify: `bench/harness.py:31,78`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reader.py`:

```python
import cv2, numpy as np, zxingcpp
from datamatrix_reader import reader as R
from datamatrix_reader.reader import Reader

_DM = zxingcpp.BarcodeFormat.DataMatrix

def _encoded(payload, scale=8):
    grid = np.asarray(zxingcpp.create_barcode(payload, _DM).to_image())
    return cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def test_reads_clean_code_stage_raw():
    r = Reader().read(_encoded(b"S25-04821-A3"))
    assert r.payload == b"S25-04821-A3"
    assert r.stage == "raw"
    assert r.ok and r.elapsed_ms >= 0

def test_blank_image_is_not_ok():
    r = Reader().read(np.full((120, 120), 255, np.uint8))
    assert r.payload is None and r.stage is None and not r.ok

def test_accepts_bgr_and_gray():
    gray = _encoded(b"S25-04821-A3")
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert Reader().read(bgr).payload == b"S25-04821-A3"

def test_falls_back_to_clahe_stage(monkeypatch):
    # deterministically force stage 1 to miss, stage 2 to hit; verifies the
    # fallback wiring AND that the cv2 upscale+CLAHE path runs without error.
    calls = {"n": 0}
    def fake_zxing(gray):
        calls["n"] += 1
        return None if calls["n"] == 1 else b"RECOVERED"
    monkeypatch.setattr(R, "_zxing", fake_zxing)
    r = Reader().read(np.full((50, 50), 255, np.uint8))
    assert r.payload == b"RECOVERED" and r.stage == "clahe"
    assert calls["n"] == 2  # tried raw, then clahe
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_reader.py -q`
Expected: FAIL — the current `ReadResult` has no `stage` attribute and `Reader` takes different args (AttributeError / TypeError).

- [ ] **Step 3: Rewrite `src/datamatrix_reader/reader.py`**

Replace the ENTIRE file with:

```python
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
```

- [ ] **Step 4: Fix the `bench/harness.py` consumer**

In `bench/harness.py`, change line 78 from:

```python
    reader = Reader(validator=AcceptAny())
```
to:
```python
    reader = Reader()
```

And delete the now-unused import (line 31):
```python
from datamatrix_reader.validate import AcceptAny
```

- [ ] **Step 5: Run the new tests + full suite**

Run: `.venv/bin/python -m pytest tests/test_reader.py -q && .venv/bin/python -m pytest tests/ -q`
Expected: `test_reader.py` 4 passed; full suite green (synth/harness tests pass — the old libdmtx encoder still feeds the new zxing reader, verified safe).

- [ ] **Step 6: Commit**

```bash
git add src/datamatrix_reader/reader.py tests/test_reader.py bench/harness.py
git commit -m "feat(reader): rewrite Reader as zxing 2-stage cascade (raw -> upscale+CLAHE)"
```

---

### Task 2: Port `synth.render` to the zxing encoder

**Files:**
- Modify: `src/datamatrix_reader/synth.py:20,37-39`

- [ ] **Step 1: Add a focused test for the encoder shape/round-trip**

Append to `tests/test_synth.py`:

```python
import zxingcpp
from datamatrix_reader.reader import Reader

def test_render_is_1px_module_binary_and_round_trips():
    grid = synth.render(b"S25-04821-A3")
    assert grid.ndim == 2 and grid.dtype == np.uint8
    assert set(np.unique(grid)).issubset({0, 255})
    # 1px/module is tiny; upscale and confirm it decodes back to the payload
    big = cv2.resize(grid, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    assert Reader().read(big).payload == b"S25-04821-A3"

def test_render_payload_pool_spans_multiple_symbol_sizes():
    from bench.harness import _payload_pool
    sizes = {synth.render(p).shape for p in _payload_pool()}
    assert len(sizes) >= 3, f"only {len(sizes)} symbol size(s): {sizes}"
```

(Add `import cv2` to the top of `tests/test_synth.py` if not already present.)

- [ ] **Step 2: Run to verify the new tests pass against the CURRENT (libdmtx) render**

Run: `.venv/bin/python -m pytest tests/test_synth.py -k "render_is_1px or payload_pool_spans" -q`
Expected: PASS (the libdmtx encoder already satisfies these invariants — this locks the contract before we swap the implementation).

- [ ] **Step 3: Swap the encoder implementation**

In `src/datamatrix_reader/synth.py`:

Replace the import (line 20):
```python
from . import binding
```
with:
```python
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
```

Replace `render` (lines 37-39):
```python
def render(payload: bytes) -> np.ndarray:
    """Clean 1px-module grid (0/255)."""
    return binding.encode(payload, module_size=1, margin=2)
```
with:
```python
def render(payload: bytes) -> np.ndarray:
    """Clean 1px-module DataMatrix grid (0/255), encoded by zxing-cpp.

    Passing bytes yields a tight 1-pixel-per-module bitmap with zxing's quiet
    zone; synth.degrade adds its own scaling/border/crowding on top."""
    return np.asarray(zxingcpp.create_barcode(payload, _DM).to_image()).copy()
```

- [ ] **Step 4: Run the synth tests + full suite**

Run: `.venv/bin/python -m pytest tests/test_synth.py -q && .venv/bin/python -m pytest tests/ -q`
Expected: all green. (Pre-verified: every AXES value and the ink_gain@min-module case decode with the zxing-encoded codes through the new reader.)

- [ ] **Step 5: Commit**

```bash
git add src/datamatrix_reader/synth.py tests/test_synth.py
git commit -m "feat(synth): encode test codes with zxing-cpp writer (drop libdmtx encoder)"
```

---

### Task 3: Repoint `test_harness.py` off `binding`

**Files:**
- Modify: `tests/test_harness.py:7,14`

- [ ] **Step 1: Update the symbol-size test to use `synth.render`**

In `tests/test_harness.py`:

Delete the import (line 7):
```python
from datamatrix_reader import binding
```
And change the body of `test_payload_pool_spans_multiple_symbol_sizes` (line 14) from:
```python
    sizes = {binding.encode(p, module_size=1, margin=2).shape for p in _payload_pool()}
```
to:
```python
    sizes = {render(p).shape for p in _payload_pool()}
```
(`render` is already imported on line 8: `from datamatrix_reader.synth import DegradeParams, degrade, render`.)

- [ ] **Step 2: Run the harness tests + full suite**

Run: `.venv/bin/python -m pytest tests/test_harness.py -q && .venv/bin/python -m pytest tests/ -q`
Expected: all green. `binding` is now imported nowhere except `tools/compare_backends.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_harness.py
git commit -m "test(harness): use synth.render for symbol-size check (drop binding import)"
```

---

### Task 4: Repurpose `compare_backends` to zxing-raw vs zxing-cascade

**Files:**
- Modify: `tools/compare_backends.py` (module docstring, imports, FOLDS)

- [ ] **Step 1: Update the module docstring (lines 1-13)**

Replace the top docstring block with:

```python
"""2-fold decode comparison on the same images:
  1. zxing raw      -> plain gray -> zxing-cpp (single pass)
  2. zxing cascade  -> Reader: raw zxing, then upscale2x+CLAHE on a miss

Fold 2 minus fold 1 is the value of the poor-print fallback. Run on labelled
sets (synth/corpus) for correct rate; pathology has no truth, so report
decode-hits + cross-decoder agreement.

    python -m tools.compare_backends --synth --per-cell 1 --budget 250
    python -m tools.compare_backends --corpus corpus/wsi_labels --budget 250
    python -m tools.compare_backends --pathology corpus/pathology_samples
"""
```

- [ ] **Step 2: Drop the libdmtx import**

Delete line 25:
```python
from datamatrix_reader import binding
```

- [ ] **Step 3: Replace the fold functions and `FOLDS`**

Replace the three fold functions and the `FOLDS` list (the `fold_preprocess_libdmtx` / `fold_libdmtx_raw` / `fold_zxing` defs and `FOLDS = [...]`) with:

```python
def fold_zxing_raw(img, budget):
    res = zxingcpp.read_barcodes(_gray(img), formats=_DM)
    return res[0].bytes if res else None


def fold_zxing_cascade(img, budget):
    return _reader.read(img, budget_ms=budget).payload


FOLDS = [("zxing raw", fold_zxing_raw),
         ("zxing cascade", fold_zxing_cascade)]
```

(Keep the existing `_gray` helper, `_reader = Reader()`, and `_DM`. The `_reader` line stays; `from datamatrix_reader.reader import Reader` stays.)

- [ ] **Step 4: Verify it loads and runs**

Run:
```bash
.venv/bin/python -c "import tools.compare_backends as m; print([n for n,_ in m.FOLDS])"
.venv/bin/python -m pytest tests/test_compare_corpus.py -q
```
Expected: prints `['zxing raw', 'zxing cascade']`; corpus-loader tests pass.

- [ ] **Step 5: Smoke-run on synth (fast, proves both folds execute)**

Run: `.venv/bin/python -m tools.compare_backends --synth --per-cell 1 --budget 250 2>&1 | tail -4`
Expected: a table with `zxing raw` and `zxing cascade` rows, n=432, no crash.

- [ ] **Step 6: Commit**

```bash
git add tools/compare_backends.py
git commit -m "feat(compare_backends): 2-fold zxing-raw vs zxing-cascade (drop libdmtx folds)"
```

---

### Task 5: Delete the libdmtx shim and preprocessing front-end

**Files:**
- Delete: `src/datamatrix_reader/binding.py`, `_build_dmtx.py`, `_dmtx.cpython-312-darwin.so`, `cascade.py`, `localize.py`, `adapt.py`

- [ ] **Step 1: Confirm nothing still imports them**

Run:
```bash
grep -rnE "import (binding|cascade|localize|adapt|_build_dmtx)|from \.(binding|cascade|localize|adapt)|_dmtx" src tools bench tests | grep -v "\.pyc"
```
Expected: **no output** (all references removed in Tasks 1-4). If anything prints, fix that file before deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm src/datamatrix_reader/binding.py src/datamatrix_reader/_build_dmtx.py \
       src/datamatrix_reader/_dmtx.cpython-312-darwin.so \
       src/datamatrix_reader/cascade.py src/datamatrix_reader/localize.py src/datamatrix_reader/adapt.py
```

- [ ] **Step 3: Verify import + full suite still green**

Run:
```bash
.venv/bin/python -c "import datamatrix_reader.reader, datamatrix_reader.synth, datamatrix_reader.validate; print('import ok')"
.venv/bin/python -m pytest tests/ -q
```
Expected: `import ok`; full suite green.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: delete libdmtx shim (binding/_build_dmtx/.so) + cascade/localize/adapt"
```

---

### Task 6: Update `pyproject.toml` and `HANDOFF.md`; reinstall clean

**Files:**
- Modify: `pyproject.toml`
- Modify: `HANDOFF.md`

- [ ] **Step 1: Rewrite `pyproject.toml`**

Replace the ENTIRE file with:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "datamatrix_reader"
version = "0.0.1"
description = "Source-agnostic DataMatrix reader for slide labels (zxing-cpp core)"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24", "opencv-python-headless>=4.8", "zxing-cpp>=2.0"]

[tool.setuptools.packages.find]
where = ["src"]
```

(Removed: `cffi` from build + runtime, the `[project.optional-dependencies] compare` extra, `[tool.setuptools.package-data] *.so`, and the `_build_dmtx` build note.)

- [ ] **Step 2: Update install commands in `HANDOFF.md`**

In `HANDOFF.md`, change the environment-rebuild commands so they no longer install the `[compare]` extra or build the cffi shim. Replace the `pip install -e ".[compare]" setuptools pytest` line with:
```bash
.venv/bin/pip install -e . pytest
```
And delete the two "Native cffi shim" lines (the `CPATH=... python -m datamatrix_reader._build_dmtx` build command and the `mv datamatrix_reader/_dmtx.*.so ...` line) plus the `brew install libdmtx` native-dep note — they no longer apply. Add a one-line note: `# zxing-cpp is a pip wheel; no native build step.`

- [ ] **Step 3: Reinstall from the rewritten metadata and verify clean**

Run:
```bash
.venv/bin/pip install -e . pytest 2>&1 | tail -3
.venv/bin/python -c "import cffi" 2>&1 | tail -1   # expect ModuleNotFoundError is fine; cffi no longer required
.venv/bin/python -c "import datamatrix_reader.reader, zxingcpp; print('install ok')"
.venv/bin/python -m pytest tests/ -q
```
Expected: install succeeds; `install ok`; full suite green. (cffi may still be present from before — that's harmless; the point is it's no longer a declared dependency.)

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml HANDOFF.md
git commit -m "build: drop cffi/libdmtx; promote zxing-cpp to a core dependency"
```

---

### Task 7: Acceptance — the cascade reproduces the measured gain (manual verification)

**Files:** none (verification only)

- [ ] **Step 1: Run the repurposed comparison on the real WSI corpus**

Run: `.venv/bin/python -m tools.compare_backends --corpus corpus/wsi_labels --budget 250 2>&1 | tail -6`
Expected: two rows on n=404 — `zxing raw` ≈ **0.866** (350/404) and `zxing cascade` ≈ **0.926** (374/404), with the cascade's p50 still a few ms. This confirms the new `Reader` delivers the validated 0.87→0.93 lift end-to-end.

- [ ] **Step 2: Confirm the synth bench still runs through the new reader**

Run: `.venv/bin/python -m bench.harness --synth --per-cell 1 --budget 250 --out runs/reanchor_check.json 2>&1 | tail -3`
Expected: completes without error and writes the summary (overall correct/found rates print).

- [ ] **Step 3: No commit** (runs/ is gitignored; this task only verifies).

---

## Notes for the implementer

- Work on a feature branch (not `main`): `git checkout -b feat/reanchor-zxing` before Task 1.
- `tools/` and `bench/` have no `__init__.py`; they're imported via the `-m` runner / pytest rootdir. Don't add packaging.
- Decoder returns are `bytes` (`barcode.bytes`); keep that — bench/compare/tests compare bytes.
- The `Reader` `budget_ms` parameter is intentionally accepted-but-ignored; do not add deadline logic (zxing is fast/uncancellable). Keeping the parameter avoids touching every call site.
- After this lands, these memories are stale and should be updated: the libdmtx build steps, the `bbox y-flip` note (obsolete — `binding.py` is gone), `[compare]`-extra references. (Out of plan scope; flag to the user.)
