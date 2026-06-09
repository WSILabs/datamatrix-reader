# Progressive Ink-Thickening Cascade Implementation Plan

> **STATUS: IMPLEMENTED (commit e4a66f6).** Final design evolved during exploration:
> the fixed `ink1/ink2` ladder below became an escalating thickening loop
> (`clahe → thick_u{2,4}_i{1,2,3} → sauv`), verified **0.983 (397/404), WRONG=0**.
> Ignore any "0.988" (a retracted double-count error). preprocess.py + reader.py
> implement it; 40 tests pass.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-frame, fallback-only ink-thickening stages to `Reader` so it recovers faint DataMatrix codes — 0.926 → **0.975** on the real WSI labels, open/no-commercial, zero false decodes.

**Architecture:** A new `preprocess.py` holds the ordered stage transforms (`clahe`, `ink1`, `ink2`), each `upscale → CLAHE → Otsu → erode(N)` escalating how much it grows the dark modules. `Reader.read()` runs raw zxing, then iterates the stages only on a miss; first decode wins; `ReadResult.stage` names it. Fallback-only, so p50 stays ~3 ms.

**Tech Stack:** Python 3.12, OpenCV (resize/CLAHE/threshold/erode), zxing-cpp, numpy.

Spec: `docs/superpowers/specs/2026-06-08-ink-thickening-cascade-design.md`

**Pre-verified end-to-end (so the plan carries real numbers):** the exact ladder `raw → clahe → ink1 → ink2` reads **394/404 = 0.975, WRONG=0**; stage mix `raw 350 / clahe 24 / ink1 16 / ink2 4`; latency p50 3.2 ms, p95 40 ms, max 97 ms.

---

## File Structure

- **Create** `src/dmtxslide/preprocess.py` — stage transforms + ordered `STAGES` list (pure OpenCV/numpy, no zxing).
- **Create** `tests/test_preprocess.py` — unit tests for the transforms.
- **Modify** `src/dmtxslide/reader.py` — import `STAGES`, iterate them on miss, drop the inline CLAHE constants/logic; `ReadResult.stage` values expand.
- **Modify** `tests/test_reader.py` — keep existing tests, add ink-stage routing tests.

Run everything with `.venv/bin/python` from `/Volumes/Ext/GitHub/datamatrix-reader/dmtxslide`. Branch first: `git checkout -b feat/ink-thickening-cascade`.

---

### Task 1: `preprocess.py` — the stage ladder

**Files:**
- Create: `src/dmtxslide/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_preprocess.py`:

```python
import numpy as np
from dmtxslide import preprocess as pp


def test_stages_named_and_ordered():
    assert [name for name, _ in pp.STAGES] == ["clahe", "ink1", "ink2"]


def test_each_stage_returns_2d_uint8():
    g = np.full((40, 40), 128, np.uint8)
    for name, fn in pp.STAGES:
        out = fn(g)
        assert out.ndim == 2 and out.dtype == np.uint8, name


def test_thicken_grows_dark_region():
    # binary: white field with a small black (ink) square; thicken must add ink
    b = np.full((20, 20), 255, np.uint8)
    b[8:12, 8:12] = 0
    out = pp._thicken(b, 1)
    assert (out == 0).sum() > (b == 0).sum()


def test_ink_stages_binarize_to_two_levels():
    g = (np.random.default_rng(0).integers(0, 255, (40, 40))).astype(np.uint8)
    for name in ("ink1", "ink2"):
        fn = dict(pp.STAGES)[name]
        assert set(np.unique(fn(g))).issubset({0, 255}), name
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_preprocess.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmtxslide.preprocess'`.

- [ ] **Step 3: Create `src/dmtxslide/preprocess.py`**

```python
"""Ordered, full-frame preprocessing stages for the Reader's fallback cascade.

Each stage thickens the dark ink progressively (upscale -> strong CLAHE -> Otsu ->
morphological erosion that GROWS the dark modules), recovering faint / broken-finder
DataMatrix codes. Pure OpenCV/numpy; the Reader runs these only when raw zxing misses.

Stage params are tuned on the wsi_labels corpus; the *principle* (progressive
ink-thickening) generalizes, the exact numbers may need re-validation on fresh data.
"""
from __future__ import annotations

import cv2
import numpy as np

CLAHE_TILE = (8, 8)


def _up(g: np.ndarray, f: int) -> np.ndarray:
    return cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def _clahe(g: np.ndarray, clip: float) -> np.ndarray:
    return cv2.createCLAHE(clip, CLAHE_TILE).apply(g)


def _otsu(g: np.ndarray) -> np.ndarray:
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _thicken(binary: np.ndarray, iters: int) -> np.ndarray:
    # erode shrinks bright regions -> grows the dark (ink) modules
    return cv2.erode(binary, np.ones((2, 2), np.uint8), iterations=iters)


def s_clahe(g: np.ndarray) -> np.ndarray:
    return _clahe(_up(g, 2), 2.0)


def s_ink1(g: np.ndarray) -> np.ndarray:
    return _thicken(_otsu(_clahe(_up(g, 2), 4.0)), 1)


def s_ink2(g: np.ndarray) -> np.ndarray:
    return _thicken(_otsu(_clahe(_up(g, 4), 4.0)), 2)


# ordered fallback stages: (name, transform: gray -> gray). Tried in order, only
# when prior stages (incl. raw zxing) miss; first decode wins.
STAGES = [("clahe", s_clahe), ("ink1", s_ink1), ("ink2", s_ink2)]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_preprocess.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dmtxslide/preprocess.py tests/test_preprocess.py
git commit -m "feat(preprocess): progressive ink-thickening stage ladder"
```

---

### Task 2: Wire the ladder into `Reader`

**Files:**
- Modify: `src/dmtxslide/reader.py`
- Modify: `tests/test_reader.py`

- [ ] **Step 1: Add the ink-stage routing tests**

In `tests/test_reader.py`, the existing tests are: `test_reads_clean_code_stage_raw`, `test_blank_image_is_not_ok`, `test_accepts_bgr_and_gray`, `test_falls_back_to_clahe_stage` (monkeypatches `R._zxing`). Keep them all. Append:

```python
def test_falls_back_through_ink_stages(monkeypatch):
    # raw, clahe, ink1 all miss; ink2 hits -> stage "ink2", stages run in order
    seq = iter([None, None, None, b"P"])
    monkeypatch.setattr(R, "_zxing", lambda g: next(seq))
    r = Reader().read(np.full((60, 60), 255, np.uint8))
    assert r.payload == b"P" and r.stage == "ink2"


def test_stage_transform_error_is_treated_as_miss(monkeypatch):
    # a stage that raises must not crash read(); it's skipped like a miss
    import dmtxslide.preprocess as pp
    boom = [("clahe", lambda g: (_ for _ in ()).throw(cv2.error("x"))),
            ("ink1", pp.s_ink1), ("ink2", pp.s_ink2)]
    monkeypatch.setattr(R, "STAGES", boom)
    monkeypatch.setattr(R, "_zxing", lambda g: None)   # everything misses
    r = Reader().read(np.full((60, 60), 255, np.uint8))
    assert r.payload is None and r.stage is None
```

(Add `import cv2` to the top of `tests/test_reader.py` if not already present.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_reader.py -q`
Expected: FAIL — `test_falls_back_through_ink_stages` (current reader only has a single hard-coded clahe stage, never reaches "ink2") and `test_stage_transform_error_is_treated_as_miss` (no `R.STAGES`, no try/except).

- [ ] **Step 3: Rewrite `src/dmtxslide/reader.py`**

Replace the ENTIRE file with:

```python
"""Public API: Reader.read(image, budget_ms).

zxing-cpp decodes the grayscale image directly (stage "raw"). On a miss, the Reader
runs an ordered ladder of full-frame preprocessing stages (see preprocess.STAGES)
that progressively thicken faint ink until the code decodes — recovering
poorly-printed codes (real WSI: 0.926 -> 0.975, validated on that corpus; the stage
params may need re-checking on fresh captures). Stages run ONLY on a miss, so p50
stays ~3 ms. `budget_ms` is accepted for call-site compatibility but IGNORED.
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
    stage: str | None          # "raw" | "clahe" | "ink1" | "ink2" | None
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
```

Notes:
- `STAGES` is imported at module scope, so the test's `monkeypatch.setattr(R, "STAGES", ...)` rebinds the module global the loop reads.
- `_zxing` now wraps input in `np.ascontiguousarray` (the ink stages return non-contiguous slices/views in some OpenCV paths).

- [ ] **Step 4: Run reader tests + full suite**

Run: `.venv/bin/python -m pytest tests/test_reader.py -q && .venv/bin/python -m pytest tests/ -q`
Expected: all reader tests pass (incl. the two new ones); full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/dmtxslide/reader.py tests/test_reader.py
git commit -m "feat(reader): run ink-thickening fallback ladder on zxing miss"
```

---

### Task 3: Acceptance — confirm the end-to-end number (verification only)

**Files:** none.

- [ ] **Step 1: Run the ground-truthed comparison on the WSI labels**

Run: `.venv/bin/python -m tools.compare_backends --corpus corpus/wsi_labels --budget 250 2>&1 | tail -6`
Expected: `zxing cascade` ≈ **0.975** (394/404) vs `zxing raw` ≈ 0.866 (350/404), cascade p50 a few ms.

- [ ] **Step 2: Confirm zero false decodes + latency profile**

Run:
```bash
.venv/bin/python - <<'PY'
import csv, cv2, time
from pathlib import Path
from collections import Counter
from dmtxslide.reader import Reader
corpus = Path("corpus/wsi_labels")
GT = {r["file"]: r["payload"].encode() for r in csv.DictReader((corpus/"labels.csv").open(newline=""))}
rd = Reader(); ok=wrong=0; by=Counter(); ts=[]
for n,t in GT.items():
    t0=time.perf_counter(); r=rd.read(cv2.imread(str(corpus/n))); ts.append((time.perf_counter()-t0)*1000)
    if r.payload==t: ok+=1; by[r.stage]+=1
    elif r.payload is not None: wrong+=1
ts.sort()
print(f"{ok}/{len(GT)}={ok/len(GT):.3f} WRONG={wrong} stages={dict(by)} p50={ts[len(ts)//2]:.1f} p95={ts[int(.95*len(ts))]:.1f} max={ts[-1]:.1f}")
PY
```
Expected: `394/404=0.975 WRONG=0 stages={'raw':350,'clahe':24,'ink1':16,'ink2':4} p50~3 p95~40 max~100`. **`WRONG=0` is the gate** — if any false decode appears, stop and revisit (do not ship a reader that mis-reads accessions).

- [ ] **Step 3: No commit** (verification only).

---

## Notes for the implementer

- Work on branch `feat/ink-thickening-cascade` (do not implement on `main`).
- `bench/harness.py` and `tools/compare_backends.py` consume `Reader` via `.read().payload`/`.ok`/`.stage` — the API is unchanged, so they need no edits; `compare_backends`'s `zxing cascade` fold now measures the ladder automatically.
- Decoder returns are `bytes`; keep that boundary.
- Do NOT add a `deep`/speed flag, a crop/localizer stage, or a false-positive guard — all explicitly out of scope (see spec). The ladder always runs as fallback; ECC is the FP guard.
- After this lands, update memory `[[clahe-upscale-fallback-recovers-poor-codes]]` and the resume pointer (the reader is a 4-stage ladder now, ~0.975), and the `Reader` docstring already records the wsi_labels-baseline caveat.
