# DataMatrix Grid-Reconstruction Decoder Implementation Plan

> **OUTCOME: BUILT & SHELVED.** Tasks 1–5 done; the Task-5 corpus gate failed
> (2/404 on real WSI, recovers 0 residual — `localize` doesn't transfer to real
> labels). Task 7 (integration) skipped per the gate. Code on branch
> `feat/grid-decode` (shelved, not merged). See [[grid-reconstruction-shelved]].

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone `griddecode.decode(image)` that recovers broken-finder DataMatrix codes the ink cascade misses, by reconstructing a clean module matrix and delegating layout + Reed–Solomon to zxing — validated on the full 404 corpus with WRONG=0 as the gate.

**Architecture:** Localize the code's square → brute-force over standard sizes × 4 rotations → sample each N×N module center → overwrite the border with a *correct* finder/timing → render a pristine 1px/module image → `zxing.read` (does ECC200 layout + RS, and validates). Build only the front-end; zxing is the back-end.

**Tech Stack:** Python 3.12, OpenCV (threshold/contours/perspective-warp), zxing-cpp, numpy.

Spec: `docs/superpowers/specs/2026-06-09-grid-reconstruction-decoder-design.md`

**Scope note:** v1 handles **square sizes ≤ 26** (`[10,12,14,16,18,20,22,24,26]`) — these have a single data region (1-module finder/timing border, `(N-2)×(N-2)` data interior, no internal alignment patterns). Larger symbols are deferred.

**Spike-first:** Task 1 proves the render→zxing keystone; Task 5 is a hard decision gate on the real corpus. If either fails, stop/pivot before building more.

---

## File Structure

- **Create** `src/dmtxslide/griddecode.py` — `decode()` + helpers `localize`, `perspective_warp`, `sample_modules`, `render_symbol`, `_zxing`, constant `SQUARE_SIZES`.
- **Create** `tests/test_griddecode.py` — unit tests (render round-trip, localize, sample, decode).
- **Create** `tools/eval_griddecode.py` — corpus validation harness (the Task 5 gate).
- **Modify** (Task 7, gated) `src/dmtxslide/reader.py` — add `griddecode` as the final fallback stage.

Run with `.venv/bin/python` from `/Volumes/Ext/GitHub/datamatrix-reader/dmtxslide`. Branch: `git checkout -b feat/grid-decode`.

---

### Task 1: `render_symbol` + the render→zxing keystone

**Files:**
- Create: `src/dmtxslide/griddecode.py`
- Create: `tests/test_griddecode.py`

- [ ] **Step 1: Write the failing keystone test**

The keystone: a correct data interior, wrapped in our constructed border, renders to an image zxing decodes. We get a real code's true module matrix from zxing's writer (`create_barcode(...).to_image()` is the N×N module image), strip its border to the `(N-2)×(N-2)` data, hand that to `render_symbol`, and confirm the round-trip.

```python
# tests/test_griddecode.py
import numpy as np
import zxingcpp
from dmtxslide import griddecode as gd

_DM = zxingcpp.BarcodeFormat.DataMatrix

def _true_matrix(payload):
    # zxing renders 1px/module: dark module = 0, light = 255. Return NxN bool (True=dark).
    img = np.asarray(zxingcpp.create_barcode(payload, _DM).to_image())
    return img == 0

def test_render_symbol_round_trips_via_zxing():
    m = _true_matrix(b"S25-04821-A3")
    N = m.shape[0]
    assert m.shape[0] == m.shape[1] and N <= 26      # square, in v1 scope
    data = m[1:N-1, 1:N-1]                            # the (N-2)x(N-2) data interior
    img = gd.render_symbol(data, N)                  # our border + this data
    assert gd._zxing(img) == b"S25-04821-A3"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmtxslide.griddecode'`.

- [ ] **Step 3: Create `src/dmtxslide/griddecode.py` with `render_symbol`, `_zxing`, `SQUARE_SIZES`**

```python
"""Grid-reconstruction decoder: reconstruct a clean DataMatrix module matrix from a
degraded image and let zxing do the ECC200 layout + Reed-Solomon (ECC-validated).

Front-end only. v1: square sizes <= 26 (single data region). See the design spec.
"""
from __future__ import annotations

import cv2
import numpy as np
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
SQUARE_SIZES = [10, 12, 14, 16, 18, 20, 22, 24, 26]


def _zxing(img: np.ndarray) -> bytes | None:
    res = zxingcpp.read_barcodes(np.ascontiguousarray(img), formats=_DM)
    return res[0].bytes if res else None


def _border_mask(N: int) -> np.ndarray:
    """True where an ECC200 square symbol's finder/timing module is DARK.
    N is even. Left col + bottom row solid (the L); top row + right col timing."""
    r = np.arange(N)[:, None]
    c = np.arange(N)[None, :]
    return ((c == 0) | (r == N - 1)
            | ((r == 0) & (c % 2 == 0))
            | ((c == N - 1) & (r % 2 == 1)))


def render_symbol(data: np.ndarray, N: int) -> np.ndarray:
    """Render an N×N symbol image (uint8, dark=0/light=255) from the (N-2)×(N-2) data
    interior (bool, True=dark), overwriting the border with a correct finder/timing.
    Adds a 2-module quiet zone and upscales 8× so zxing can detect it."""
    dark = _border_mask(N).copy()
    dark[1:N - 1, 1:N - 1] = data
    sym = np.where(dark, 0, 255).astype(np.uint8)
    sym = cv2.copyMakeBorder(sym, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(sym, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -q`
Expected: 1 passed. (If it FAILS, the border rule is wrong — `_border_mask` is the suspect; the round-trip is the arbiter.)

- [ ] **Step 5: Commit**

```bash
git add src/dmtxslide/griddecode.py tests/test_griddecode.py
git commit -m "feat(griddecode): render_symbol + render->zxing keystone"
```

---

### Task 2: `localize` — find the code's square

**Files:**
- Modify: `src/dmtxslide/griddecode.py`
- Modify: `tests/test_griddecode.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_griddecode.py
import cv2

def _code_on_canvas(payload, scale=10, pad=60, angle=0):
    m = (~_true_matrix(payload)).astype(np.uint8) * 255   # dark=0
    big = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((big.shape[0] + 2 * pad, big.shape[1] + 2 * pad), 255, np.uint8)
    canvas[pad:pad + big.shape[0], pad:pad + big.shape[1]] = big
    if angle:
        M = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), angle, 1)
        canvas = cv2.warpAffine(canvas, M, canvas.shape[::-1], borderValue=255)
    return canvas

def test_localize_finds_square_quad():
    img = _code_on_canvas(b"S25-04821-A3", scale=10, pad=60)
    quad = gd.localize(img)
    assert quad is not None and quad.shape == (4, 2)
    # the quad's bounding box should roughly cover the code region (~ centre of canvas)
    xs, ys = quad[:, 0], quad[:, 1]
    assert 40 < xs.min() < 90 and 40 < ys.min() < 90

def test_localize_blank_returns_none():
    assert gd.localize(np.full((200, 200), 255, np.uint8)) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -k localize -q`
Expected: FAIL — `AttributeError: module 'dmtxslide.griddecode' has no attribute 'localize'`.

- [ ] **Step 3: Implement `localize`**

```python
# add to griddecode.py
def localize(gray: np.ndarray) -> np.ndarray | None:
    """Return the 4 corners (float32, shape (4,2)) of the largest square-ish dark blob,
    or None. Targets the code's overall extent, tolerant of a broken finder."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for cont in cnts:
        rect = cv2.minAreaRect(cont)
        (w, h) = rect[1]
        if w < 20 or h < 20:
            continue
        ar = max(w, h) / max(1.0, min(w, h))
        fill = cv2.contourArea(cont) / max(1.0, w * h)
        if ar > 1.4 or fill < 0.35:           # not square / not filled enough
            continue
        area = w * h
        if best is None or area > best[0]:
            best = (area, cv2.boxPoints(rect))
    return None if best is None else best[1].astype(np.float32)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -k localize -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/dmtxslide/griddecode.py tests/test_griddecode.py
git commit -m "feat(griddecode): localize the code's square quad"
```

---

### Task 3: `perspective_warp` + `sample_modules`

**Files:**
- Modify: `src/dmtxslide/griddecode.py`
- Modify: `tests/test_griddecode.py`

- [ ] **Step 1: Write the failing test**

A clean code, localized and warped, then sampled at its true N, must reproduce the true module matrix (allowing a tiny error budget for edge modules).

```python
# append to tests/test_griddecode.py
def test_warp_and_sample_recover_true_matrix():
    payload = b"S25-04821-A3"
    N = _true_matrix(payload).shape[0]
    img = _code_on_canvas(payload, scale=12, pad=60)
    quad = gd.localize(img)
    warp = gd.perspective_warp(img, quad, gd.SIDE)   # _code_on_canvas returns grayscale
    grid = gd.sample_modules(warp, N)            # NxN bool, True=dark
    true = _true_matrix(payload)
    # orientation of the warp is arbitrary; the true matrix matches under some rot90
    matches = [np.mean(np.rot90(grid, k) == true) for k in range(4)]
    assert max(matches) > 0.95                   # >95% modules correct in best orientation
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -k warp_and_sample -q`
Expected: FAIL — no `SIDE` / `perspective_warp` / `sample_modules`.

- [ ] **Step 3: Implement `perspective_warp` + `sample_modules`**

```python
# add to griddecode.py
SIDE = 480   # warped square buffer size (px); divisible by all SQUARE_SIZES' typical cells


def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL for a stable warp."""
    s = quad.sum(1)
    d = np.diff(quad, axis=1).ravel()
    return np.array([quad[np.argmin(s)], quad[np.argmin(d)],
                     quad[np.argmax(s)], quad[np.argmax(d)]], dtype=np.float32)


def perspective_warp(gray: np.ndarray, quad: np.ndarray, side: int) -> np.ndarray:
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(_order_quad(quad), dst)
    return cv2.warpPerspective(g, M, (side, side))


def sample_modules(warp: np.ndarray, N: int) -> np.ndarray:
    """Sample an N×N grid of module centres from the square `warp`. Each module is the
    mean of a centred window; dark/light by Otsu over the N×N means. Returns bool
    (True = dark module)."""
    cell = warp.shape[0] / N
    win = max(1, int(cell * 0.5))
    means = np.empty((N, N), np.float32)
    for i in range(N):
        for j in range(N):
            cy, cx = int((i + 0.5) * cell), int((j + 0.5) * cell)
            y0, x0 = max(0, cy - win // 2), max(0, cx - win // 2)
            means[i, j] = warp[y0:y0 + win, x0:x0 + win].mean()
    thr = cv2.threshold(means.astype(np.uint8), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    return means < thr                            # dark modules are below threshold
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -k warp_and_sample -q`
Expected: PASS (best-orientation module agreement > 95%).

- [ ] **Step 5: Commit**

```bash
git add src/dmtxslide/griddecode.py tests/test_griddecode.py
git commit -m "feat(griddecode): perspective warp + module-center sampling"
```

---

### Task 4: `decode` — brute-force orchestration

**Files:**
- Modify: `src/dmtxslide/griddecode.py`
- Modify: `tests/test_griddecode.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_griddecode.py
def test_decode_clean_code():
    img = _code_on_canvas(b"S25-04821-A3", scale=12, pad=60)
    assert gd.decode(img) == b"S25-04821-A3"

def test_decode_blank_is_none():
    assert gd.decode(np.full((200, 200), 255, np.uint8)) is None

def test_decode_recovers_erased_finder():
    # erase the solid L finder (left col + bottom row of the rendered code) -> the ink
    # cascade can't locate it, but grid reconstruction should.
    payload = b"S25-04821-A3"
    img = _code_on_canvas(payload, scale=12, pad=60)
    # paint over the left and bottom edges of the code region (the L finder)
    import numpy as np
    ys, xs = np.where(img < 128)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    img[y0:y1 + 1, x0:x0 + 14] = 255          # erase left finder column band
    img[y1 - 13:y1 + 1, x0:x1 + 1] = 255      # erase bottom finder row band
    assert gd.decode(img) == payload
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -k decode -q`
Expected: FAIL — no `gd.decode`.

- [ ] **Step 3: Implement `decode`**

```python
# add to griddecode.py
def decode(image: np.ndarray) -> bytes | None:
    """Reconstruct + decode a (possibly broken-finder) DataMatrix. Returns the payload
    bytes or None. ECC-validated, so it never returns a wrong payload."""
    g = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    quad = localize(g)
    if quad is None:
        return None
    warp = perspective_warp(g, quad, SIDE)
    for N in SQUARE_SIZES:
        grid = sample_modules(warp, N)
        for k in range(4):
            data = np.rot90(grid, k)[1:N - 1, 1:N - 1]
            try:
                payload = _zxing(render_symbol(data, N))
            except cv2.error:
                continue
            if payload is not None:
                return payload
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -q`
Expected: all griddecode tests pass — including `test_decode_recovers_erased_finder` (the capability proof: a code with its finder erased decodes via reconstruction).

- [ ] **Step 5: Commit**

```bash
git add src/dmtxslide/griddecode.py tests/test_griddecode.py
git commit -m "feat(griddecode): brute-force decode (size x rotation, ECC-validated)"
```

---

### Task 5: Corpus validation — the DECISION GATE (verification only)

**Files:**
- Create: `tools/eval_griddecode.py`

- [ ] **Step 1: Write the corpus harness**

```python
# tools/eval_griddecode.py
"""Validate griddecode on the full real corpus: decode rate, WRONG (must be 0),
and recovery of the cascade-residual. The ship gate for the grid decoder."""
import csv
from pathlib import Path
import cv2
from dmtxslide import griddecode as gd
from dmtxslide.reader import Reader

def main():
    corpus = Path("corpus/wsi_labels")
    GT = {r["file"]: r["payload"].encode()
          for r in csv.DictReader((corpus / "labels.csv").open(newline=""))}
    rd = Reader()
    ok = wrong = 0
    cascade_resid = recovered_resid = 0
    for n, t in GT.items():
        img = cv2.imread(str(corpus / n))
        cascade_hit = rd.read(img).payload == t
        p = gd.decode(img)
        if p == t:
            ok += 1
        elif p is not None:
            wrong += 1
        if not cascade_hit:
            cascade_resid += 1
            if p == t:
                recovered_resid += 1
    n = len(GT)
    print(f"griddecode standalone: {ok}/{n} = {ok/n:.3f}   WRONG={wrong}")
    print(f"of {cascade_resid} cascade-residual, griddecode recovers {recovered_resid}")
    print("GATE: ship iff WRONG==0 and recovered_resid>=1")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the gate**

Run: `.venv/bin/python -m tools.eval_griddecode 2>&1 | tail -4`
Expected: prints decode rate, `WRONG=`, and residual recovered.

**DECISION GATE — read the output and decide:**
- **WRONG > 0** → STOP. A grid decoder that mis-reads accessions is unshippable; do not integrate. Report and reassess (likely the orientation/size brute force is accepting a wrong-but-ECC-passing matrix — investigate, but ECC makes this very unlikely).
- **WRONG == 0 and recovered_resid >= 1** → PASS. Proceed to Task 6/7.
- **WRONG == 0 and recovered_resid == 0** → the decoder is safe but adds nothing on this corpus. STOP integration; record the honest finding (residual stays human-flag). Keep the module + synthetic tests as the capability, revisit on fresh data.

- [ ] **Step 3: Commit the harness**

```bash
git add tools/eval_griddecode.py
git commit -m "feat(eval_griddecode): full-corpus validation gate for the grid decoder"
```

---

### Task 6: Synthetic finder-erasure stress (breadth)

**Files:**
- Modify: `tests/test_griddecode.py`

Only proceed if Task 5 PASSED.

- [ ] **Step 1: Add a breadth test over many payloads/sizes**

```python
# append to tests/test_griddecode.py
import pytest

@pytest.mark.parametrize("payload", [
    b"S25-04821-A3", b"1-S-24-34325 G2-1", b"PCAA00028208",
    b"B1-2HE", b"370956", b"X1",
])
def test_decode_recovers_finder_erased_across_payloads(payload):
    img = _code_on_canvas(payload, scale=12, pad=60)
    import numpy as np
    ys, xs = np.where(img < 128)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    img[y0:y1 + 1, x0:x0 + 14] = 255
    img[y1 - 13:y1 + 1, x0:x1 + 1] = 255
    assert gd.decode(img) == payload
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest tests/test_griddecode.py -k across_payloads -q`
Expected: parametrized cases pass — demonstrating the capability generalizes across payloads/sizes, not just one code. (If some sizes fail, note which; a payload that encodes to >26 modules is out of v1 scope — drop it from the list rather than forcing it.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_griddecode.py
git commit -m "test(griddecode): synthetic finder-erasure breadth across payloads"
```

---

### Task 7: Integrate as the Reader's final fallback (gated on Task 5 PASS)

**Files:**
- Modify: `src/dmtxslide/reader.py`
- Modify: `tests/test_reader.py`

Only do this if Task 5 PASSED (WRONG==0 and recovered_resid>=1).

- [ ] **Step 1: Add the routing test**

```python
# append to tests/test_reader.py
def test_grid_stage_runs_last(monkeypatch):
    # all zxing attempts miss; griddecode hits -> stage "grid"
    monkeypatch.setattr(R, "_zxing", lambda g: None)
    import dmtxslide.griddecode as gd
    monkeypatch.setattr(gd, "decode", lambda img: b"VIAGRID")
    r = Reader().read(np.full((80, 80), 255, np.uint8))
    assert r.payload == b"VIAGRID" and r.stage == "grid"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_reader.py -k grid_stage -q`
Expected: FAIL — reader has no grid stage.

- [ ] **Step 3: Wire `griddecode` into `Reader.read`**

In `src/dmtxslide/reader.py`, add the import and a final fallback after the `STAGES` loop. The full `read` method becomes:

```python
from . import griddecode

# ... inside class Reader.read, replace the body after the STAGES loop:
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
                    continue
                if cand is not None:
                    payload, stage = cand, name
                    break
        if payload is None:
            cand = griddecode.decode(gray)      # final fallback: reconstruct the grid
            if cand is not None:
                payload, stage = cand, "grid"
        return ReadResult(payload, stage, (time.perf_counter() - t0) * 1000)
```

Update the `ReadResult.stage` docstring comment to include `"grid"`.

- [ ] **Step 4: Run reader tests + full suite + acceptance**

Run:
```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m tools.compare_backends --corpus corpus/wsi_labels --budget 250 2>&1 | tail -4
```
Expected: full suite green; cascade rate ≥ 0.983 (grid stage adds the recovered residual), and re-confirm WRONG=0 via the Task 5 harness.

- [ ] **Step 5: Commit**

```bash
git add src/dmtxslide/reader.py tests/test_reader.py
git commit -m "feat(reader): griddecode as the final fallback stage"
```

---

## Notes for the implementer

- Branch `feat/grid-decode` off `main`; don't implement on `main`.
- **Tasks 1 and 5 are gates.** If Task 1's round-trip won't pass, `_border_mask` has the wrong finder/timing phase — fix that before anything else. If Task 5 shows WRONG>0 or zero recovery, STOP per its decision rules; don't force Task 7.
- The decoder is ECC-validated end to end: every accepted payload came back through zxing's Reed–Solomon, so a mis-sampled grid fails rather than mis-reads. Preserve that — never return a payload that didn't come from `_zxing`.
- v1 is square sizes ≤26 only. Don't add the ≥32 internal-alignment handling unless a real need appears.
- Decoder returns `bytes` (zxing `.bytes`); keep the bytes boundary consistent with `Reader`.
