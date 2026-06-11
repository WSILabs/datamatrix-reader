# Barcode-Repair Generalization & Efficiency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the DataMatrix finder-registration fallback to recover a damaged code placed *anywhere* on a slide label at *variable scale* and any cardinal orientation, and cut the brute-force registration cost — without regressing WSI 404/404 or WRONG=0.

**Architecture:** A new scale/position-robust front-end (`locate.propose`) finds candidate square code regions via a texture-density image pyramid; each candidate is cropped + scaled to a canonical size where the existing, validated repair core runs. Registration is sped up by a cheap score-guided search with a brute-force backstop so recall cannot regress. Validation uses extended `synth.py` full-label scenes (truth-controlled) plus the WSI and pathology corpora as regression gates.

**Tech Stack:** Python, OpenCV, NumPy, zxing-cpp. No scipy (C-portable). pytest.

**Spec:** `docs/superpowers/specs/2026-06-09-barcode-repair-generalization-design.md`

**Reference (read before starting):**
- `src/datamatrix_reader/register.py` — current repair core: `_zxing`, `_kernel`, `border_mask`, `render_symbol`, `sample_fast`, `_texture`, `detect_dark_region`, `detect_data_region`, `detect_area`, `l_orientations`, `decode_auto`, `recover`, `ROI_FRAC`, `SIZES=(22,18,20,24)`.
- `src/datamatrix_reader/synth.py` — `render(payload)->1px grid`, `degrade(grid,p,rng)`, `DegradeParams`, `crowd_quiet_zone`.
- `tests/test_register.py` — `_square_symbol()` helper returns `(payload_bytes, MxM bool dark grid)` for a square-encoding payload; `_canvas(dark, cell, quiet)` renders it.

---

## File Structure

- **Create `src/datamatrix_reader/locate.py`** — `propose(gray) -> list[(cx, cy, size, angle)]`, pyramid texture-blob proposals (position/scale-robust localization). One responsibility: "where might a square code be."
- **Modify `src/datamatrix_reader/register.py`** — drop `detect_dark_region` from the decode path; add `score_registration`, `register_candidate` (score-guided + brute-force backstop); rewrite `recover` to `propose -> normalize -> register`. `decode_auto` stays (operates at canonical scale) but its detector union becomes texture+gradient.
- **Modify `src/datamatrix_reader/synth.py`** — add `scene(payload, params, rng) -> (bgr_image, truth_dict)` placing a square code anywhere on a label canvas with cardinal rotation+skew and the new confounders (border defects, glass chip, straight edges, text).
- **Create `tests/test_locate.py`** — `propose` localizes off-center/scaled/rotated synthetic scenes.
- **Modify `tests/test_register.py`** — score-guided register agrees with brute-force; keep existing broken-border/edge tests.
- **Modify `tests/test_synth.py`** — `scene` produces non-degenerate, decodable-when-clean scenes with correct truth.
- **Create `tools/validate_synth.py`** — generalization harness: localization recall + decode rate per axis.
- **Create `tools/validate_pathology.py`** — 28 decodable `pathology_samples` pseudo-GT regression.
- **Keep `tools/validate_full.py`** — WSI 404/404 regression gate (unchanged).

---

# PHASE 1 — Generality (anywhere / any scale), low risk

## Task 1: Synthetic full-label scene generator

**Files:**
- Modify: `src/datamatrix_reader/synth.py`
- Test: `tests/test_synth.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_synth.py`:

```python
import random
import numpy as np
import zxingcpp
from datamatrix_reader import synth
from datamatrix_reader.register import _zxing

_DM = zxingcpp.BarcodeFormat.DataMatrix


def _square_payload():
    for t in (b"DMTXSLIDE-SCENE-TEST-01", b"ABCDEFGHIJKLMNOPQRSTUVWX"):
        a = np.asarray(zxingcpp.create_barcode(t.decode(), _DM).to_image())
        if a.shape[0] == a.shape[1]:
            return t
    raise AssertionError("no square payload")


def test_scene_places_code_and_reports_truth():
    rng = random.Random(0)
    payload = _square_payload()
    p = synth.SceneParams(canvas=(900, 700), cell=14, pos=(0.7, 0.3),
                          rotation_deg=90.0, chip=False, edges=False,
                          defects=False, text=True)
    img, truth = synth.scene(payload, p, rng)
    assert img.ndim == 3 and img.shape[0] >= 700
    # truth geometry is inside the canvas and the right rough size
    assert 0 <= truth["cx"] < img.shape[1] and 0 <= truth["cy"] < img.shape[0]
    assert truth["payload"] == payload
    assert abs(truth["size"] - 14 * truth["M"]) < 14 * 2  # size ~ cell*M
    # a CLEAN scene (no defects) must still decode at the placed location via crop
    g = img[..., 0] if img.ndim == 3 else img
    s = int(truth["size"]); cx, cy = int(truth["cx"]), int(truth["cy"])
    crop = g[max(0, cy - s):cy + s, max(0, cx - s):cx + s]
    import cv2
    up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    assert _zxing(up) == payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_synth.py::test_scene_places_code_and_reports_truth -v`
Expected: FAIL — `AttributeError: module 'datamatrix_reader.synth' has no attribute 'SceneParams'`.

- [ ] **Step 3: Implement `SceneParams` + `scene` in `src/datamatrix_reader/synth.py`**

Append:

```python
@dataclass(frozen=True)
class SceneParams:
    canvas: tuple = (900, 700)          # (H, W) label canvas
    cell: float = 14.0                  # px per module (scale axis)
    pos: tuple = (0.5, 0.5)             # code CENTER as (fx, fy) fraction of canvas
    rotation_deg: float = 0.0           # cardinal {0,90,180,270} + skew applied on top
    skew_deg: float = 0.0               # |skew| <= ~20 (slide tolerance)
    substrate_bgr: tuple = (255, 255, 255)
    print_bgr: tuple = (10, 10, 10)
    text: bool = True                   # adjacent accession text clutter
    edges: bool = False                 # straight slide/label edge (dark bar)
    chip: bool = False                  # bright glass-chip blob on the finder
    defects: bool = False               # half-printed top timing + nicked finder
    blur_sigma: float = 0.0
    noise_sigma: float = 0.0


def _square_dark(payload: bytes) -> np.ndarray:
    """MxM bool dark grid for a payload that encodes square; raises if none."""
    a = render(payload)                 # (M+2, M+2) incl 1px quiet
    if a.shape[0] != a.shape[1]:
        raise ValueError("payload does not encode to a square DataMatrix")
    return a[1:-1, 1:-1] < 128


def _apply_border_defects(dark: np.ndarray, rng: random.Random) -> np.ndarray:
    """Model the real WSI border failure: erase the top timing row and nick the
    finder L. Data interior untouched."""
    M = dark.shape[0]
    d = dark.copy()
    d[0, :] = False                     # top timing row prints half-height -> sampled white
    d[:, -1] = False                    # right timing col damaged
    i = rng.randint(1, M - 5)
    d[i:i + 3, 0] = False               # chip nick on the left finder arm
    return d


def scene(payload: bytes, p: SceneParams, rng: random.Random):
    """Render a square code onto a label canvas at p.pos / p.cell / rotation, with
    optional confounders. Returns (bgr_uint8, truth) where truth = {payload, cx, cy,
    size, angle, M}. Truth geometry is the code's center, side length, and net angle."""
    H, W = p.canvas
    dark = _square_dark(payload)
    if p.defects:
        dark = _apply_border_defects(dark, rng)
    M = dark.shape[0]
    # render code tile (BGR) at p.cell px/module, on its own substrate
    grid = np.where(dark, 0, 255).astype(np.uint8)
    tile = cv2.resize(grid, None, fx=p.cell, fy=p.cell, interpolation=cv2.INTER_NEAREST)
    sub = np.array(p.substrate_bgr, np.float32) / 255.0
    ink = np.array(p.print_bgr, np.float32) / 255.0
    tnorm = tile.astype(np.float32) / 255.0
    tile_bgr = ((tnorm[..., None] * sub + (1 - tnorm[..., None]) * ink) * 255).astype(np.uint8)
    side = tile_bgr.shape[0]
    # canvas
    canvas = np.full((H, W, 3), p.substrate_bgr, np.uint8)
    if p.edges:                          # a dark slide/label edge bar down one side
        canvas[:, :max(6, W // 40)] = (20, 20, 20)
    cx, cy = int(p.pos[0] * W), int(p.pos[1] * H)
    # rotate the tile (cardinal + skew) about its center, expanding
    ang = p.rotation_deg + p.skew_deg
    Rm = cv2.getRotationMatrix2D((side / 2, side / 2), ang, 1.0)
    cos, sin = abs(Rm[0, 0]), abs(Rm[0, 1])
    nw, nh = int(side * cos + side * sin), int(side * sin + side * cos)
    Rm[0, 2] += nw / 2 - side / 2
    Rm[1, 2] += nh / 2 - side / 2
    rot = cv2.warpAffine(tile_bgr, Rm, (nw, nh), borderValue=tuple(map(int, p.substrate_bgr)))
    # adjacent accession text (clutter) before stamping the code, so modules stay clean
    if p.text:
        fs = max(0.4, p.cell / 22.0)
        cv2.putText(canvas, rng.choice(ACCESSION_SAMPLES),
                    (max(0, cx - side), min(H - 4, cy + side)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, tuple(map(int, p.print_bgr)),
                    max(1, int(fs * 1.6)), cv2.LINE_AA)
    # paste rotated tile centered at (cx, cy), clipped to canvas
    y0, x0 = cy - nh // 2, cx - nw // 2
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(H, y0 + nh), min(W, x0 + nw)
    canvas[ys:ye, xs:xe] = rot[ys - y0:ye - y0, xs - x0:xe - x0]
    if p.chip:                           # bright glass-chip blob over the finder corner
        cv2.circle(canvas, (cx - side // 2, cy), max(8, side // 14), (235, 235, 235), -1)
    if p.blur_sigma > 0:
        canvas = cv2.GaussianBlur(canvas, (0, 0), p.blur_sigma)
    if p.noise_sigma > 0:
        canvas = (canvas.astype(np.float32) + np.random.default_rng(
            rng.randint(0, 1 << 30)).normal(0, p.noise_sigma, canvas.shape)
        ).clip(0, 255).astype(np.uint8)
    truth = {"payload": payload, "cx": float(cx), "cy": float(cy),
             "size": float(side), "angle": float(ang), "M": int(M)}
    return canvas, truth
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_synth.py::test_scene_places_code_and_reports_truth -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/datamatrix_reader/synth.py tests/test_synth.py
git commit -m "feat(synth): full-label scene generator (placement, rotation, confounders)"
```

---

## Task 2: `locate.propose` — pyramid texture-blob proposals

**Files:**
- Create: `src/datamatrix_reader/locate.py`
- Test: `tests/test_locate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_locate.py`:

```python
import random
import numpy as np
from datamatrix_reader import synth
from datamatrix_reader.locate import propose


def _payload():
    import zxingcpp
    for t in (b"DMTXSLIDE-LOCATE-TEST-1", b"ABCDEFGHIJKLMNOPQRSTUVWX"):
        a = np.asarray(zxingcpp.create_barcode(t.decode(),
              zxingcpp.BarcodeFormat.DataMatrix).to_image())
        if a.shape[0] == a.shape[1]:
            return t
    raise AssertionError


def _hit(cands, truth, tol_frac=0.4):
    """A proposal counts as a hit if its center is within tol*size of truth and its
    size is within 35% of truth."""
    for cx, cy, size, _ in cands:
        if (abs(cx - truth["cx"]) < tol_frac * truth["size"] and
                abs(cy - truth["cy"]) < tol_frac * truth["size"] and
                abs(size - truth["size"]) < 0.35 * truth["size"]):
            return True
    return False


def test_propose_localizes_offcenter_varied_scale():
    rng = random.Random(1)
    payload = _payload()
    hits = 0
    cases = [(0.25, 0.3, 10.0), (0.75, 0.6, 14.0), (0.5, 0.2, 22.0),
             (0.3, 0.7, 28.0), (0.8, 0.8, 18.0)]
    for fx, fy, cell in cases:
        p = synth.SceneParams(canvas=(900, 1100), cell=cell, pos=(fx, fy),
                              rotation_deg=0.0, text=True, edges=True)
        img, truth = synth.scene(payload, p, rng)
        cands = propose(img[..., 0])
        if _hit(cands, truth):
            hits += 1
    assert hits >= 4   # localizes >=4/5 across position+scale, despite a slide edge
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_locate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'datamatrix_reader.locate'`.

- [ ] **Step 3: Implement `src/datamatrix_reader/locate.py`**

```python
"""Scale/position-robust localization: propose candidate square DataMatrix regions
anywhere on a label, at any bounded scale, via a texture-density image pyramid.

`propose(gray) -> [(cx, cy, size, angle), ...]` ranked best-first (most code-like first),
in NATIVE pixel coordinates. The repair core (register.py) refines + decodes each.
"""
from __future__ import annotations

import cv2
import numpy as np

# pyramid scales: a code's native cell of ~7-35px lands near the canonical ~10px at
# some level. Tune if the scale range widens. (See spec: bounded ~0.3-1.5x of canonical.)
PYRAMID_SCALES = (1.0, 0.7, 0.5, 0.35)
_MIN_SIDE = 40          # reject sub-40px blobs at a level
_MAX_AR = 1.3           # square-ish
_MIN_FILL = 0.4


def _kernel(n):
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def _density(gray):
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (13, 13))
    var = cv2.boxFilter(g * g, -1, (13, 13)) - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def _level_candidates(gray, scale):
    lvl = (gray if scale == 1.0 else
           cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    dens = _density(lvl)
    dn = (dens / (dens.max() + 1e-9) * 255).astype(np.uint8)
    th = cv2.threshold(dn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, _kernel(7))    # drop thin text
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, _kernel(21))  # fill module grid
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        if w < _MIN_SIDE or h < _MIN_SIDE:
            continue
        if max(w, h) / max(1.0, min(w, h)) > _MAX_AR:
            continue
        fill = cv2.contourArea(c) / max(1.0, w * h)
        if fill < _MIN_FILL:
            continue
        m = np.zeros(lvl.shape, np.uint8); cv2.drawContours(m, [c], -1, 1, -1)
        strength = float(fill * dens[m > 0].mean())          # density-weighted squareness
        out.append((cx / scale, cy / scale, (w + h) / 2.0 / scale, ang, strength))
    return out


def _dedup(cands):
    """Merge proposals whose centers are within half the smaller size (same code found at
    adjacent pyramid levels); keep the strongest."""
    cands = sorted(cands, key=lambda c: -c[4])
    kept = []
    for cx, cy, size, ang, strength in cands:
        if any(abs(cx - kx) < 0.5 * min(size, ks) and abs(cy - ky) < 0.5 * min(size, ks)
               for kx, ky, ks, _, _ in kept):
            continue
        kept.append((cx, cy, size, ang, strength))
    return kept


def propose(gray):
    cands = []
    for sc in PYRAMID_SCALES:
        cands.extend(_level_candidates(gray, sc))
    return [(cx, cy, size, ang) for cx, cy, size, ang, _ in _dedup(cands)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_locate.py -v`
Expected: PASS (≥4/5). If 3/5, widen `PYRAMID_SCALES` or relax `_MIN_FILL` to 0.35 and re-run; the test is the acceptance bar.

- [ ] **Step 5: Commit**

```bash
git add src/datamatrix_reader/locate.py tests/test_locate.py
git commit -m "feat(locate): pyramid texture-blob proposals (position/scale-robust)"
```

---

## Task 3: Drop `dark` from the decode path; add a normalize helper

**Files:**
- Modify: `src/datamatrix_reader/register.py:147-174` (`decode_auto`)
- Test: `tests/test_register.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_register.py`:

```python
def test_decode_auto_uses_two_detectors():
    # the decode path must no longer call detect_dark_region
    import inspect
    from datamatrix_reader import register
    src = inspect.getsource(register.decode_auto)
    assert "detect_dark_region" not in src
    assert "detect_area" in src and "detect_data_region" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_register.py::test_decode_auto_uses_two_detectors -v`
Expected: FAIL — `detect_dark_region` still in `decode_auto`.

- [ ] **Step 3: Edit `decode_auto` region list in `src/datamatrix_reader/register.py`**

Change the `regions = [...]` line inside `decode_auto` from three detectors to two:

```python
    regions = [r for r in (detect_area(gray),
                           detect_data_region(gray)) if r]
```

Leave `detect_dark_region` defined (still importable for the harnesses/ablation) but unused by the decode path.

- [ ] **Step 4: Run tests to verify**

Run: `.venv/bin/python -m pytest tests/test_register.py -v`
Expected: all PASS (the existing synthetic broken-border/edge tests still decode with texture+gradient).

- [ ] **Step 5: Commit**

```bash
git add src/datamatrix_reader/register.py tests/test_register.py
git commit -m "perf(register): drop dark-ink detector from decode path (ablation: 0 unique)"
```

---

## Task 4: Rewrite `recover` to propose → normalize → decode (drop the hardcoded ROI)

**Files:**
- Modify: `src/datamatrix_reader/register.py` (`recover`, add `_normalize`)
- Test: `tests/test_register.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_register.py`:

```python
def test_recover_decodes_offcenter_scene():
    import random
    from datamatrix_reader import synth
    from datamatrix_reader.register import recover
    rng = random.Random(3)
    for t in (b"DMTXSLIDE-RECOVER-TEST", b"ABCDEFGHIJKLMNOPQRSTUVWX"):
        import zxingcpp
        a = np.asarray(zxingcpp.create_barcode(t.decode(),
              zxingcpp.BarcodeFormat.DataMatrix).to_image())
        if a.shape[0] == a.shape[1]:
            payload = t; break
    # code in the lower-right (NOT the old upper-left ROI), with border defects
    p = synth.SceneParams(canvas=(900, 1100), cell=18, pos=(0.72, 0.68),
                          rotation_deg=180.0, defects=True, text=True, edges=True)
    img, truth = synth.scene(payload, p, rng)
    assert recover(img[..., 0]) == payload
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_register.py::test_recover_decodes_offcenter_scene -v`
Expected: FAIL — current `recover` crops the upper-left ROI, misses a lower-right code.

- [ ] **Step 3: Rewrite `recover` and add `_normalize` in `src/datamatrix_reader/register.py`**

Replace the current `recover` (and `ROI_FRAC`) with:

```python
from .locate import propose

CANON = 470          # canonical normalized code side (px); detectors are tuned for this
_MARGIN = 0.6        # crop margin around a proposal, as a fraction of its size


def _normalize(gray, cx, cy, size):
    """Crop a window around a proposal and scale so the code is ~CANON px."""
    half = int(size * (0.5 + _MARGIN))
    y0, y1 = max(0, int(cy) - half), min(gray.shape[0], int(cy) + half)
    x0, x1 = max(0, int(cx) - half), min(gray.shape[1], int(cx) + half)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    f = CANON / max(1.0, size)
    return cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def recover(gray):
    """Reader fallback: propose candidate code regions anywhere on the label, normalize
    each to canonical scale, and run the repair decoder. Returns payload bytes or None.
    ECC-validated -> safe."""
    for cx, cy, size, _ in propose(gray):
        up = _normalize(gray, cx, cy, size)
        if up is None:
            continue
        payload, _ = decode_auto(up)
        if payload is not None:
            return payload
    return None
```

Remove the now-unused `ROI_FRAC` constant and its docstring lines.

- [ ] **Step 4: Run tests + WSI regression**

Run: `.venv/bin/python -m pytest tests/test_register.py -v`
Expected: all PASS including `test_recover_decodes_offcenter_scene`.

Run: `.venv/bin/python -m tools.validate_full 2>&1 | tail -8`
Expected: `TOTAL correct : 404/404 = 1.000`, `WRONG : 0`. (If any WSI code regresses, the proposal isn't surfacing the upper-left code — relax `_MIN_FILL` / add a pyramid level — re-run.)

- [ ] **Step 5: Commit**

```bash
git add src/datamatrix_reader/register.py tests/test_register.py
git commit -m "feat(register): recover via propose+normalize (code anywhere, any scale)"
```

---

## Task 5: Generalization + pathology regression harnesses (Phase-1 baseline)

**Files:**
- Create: `tools/validate_synth.py`
- Create: `tools/validate_pathology.py`

- [ ] **Step 1: Implement `tools/validate_synth.py`**

```python
"""Generalization harness: synthetic full-label scenes across position/scale/rotation,
reporting LOCALIZATION recall (propose finds the code) and DECODE rate (recover reads it)
separately, with the worst stratum surfaced.

    .venv/bin/python -m tools.validate_synth
"""
import random
import numpy as np
import zxingcpp
from datamatrix_reader import synth
from datamatrix_reader.locate import propose
from datamatrix_reader.register import recover

_DM = zxingcpp.BarcodeFormat.DataMatrix
PAYLOADS = [p for p in (b"S25-04821 A3-1 HE", b"PCAA00028208 A1-1",
                        b"ABCDEFGHIJKLMNOPQRSTUVWX")
            if np.asarray(zxingcpp.create_barcode(p.decode(), _DM).to_image()).shape[0]
            == np.asarray(zxingcpp.create_barcode(p.decode(), _DM).to_image()).shape[1]]

POS = [(0.25, 0.3), (0.7, 0.65), (0.5, 0.2), (0.8, 0.8)]
CELL = [10.0, 14.0, 22.0, 30.0]
ROT = [0.0, 90.0, 180.0, 270.0]


def _hit(cands, t, tol=0.4):
    return any(abs(cx - t["cx"]) < tol * t["size"] and abs(cy - t["cy"]) < tol * t["size"]
               and abs(s - t["size"]) < 0.35 * t["size"] for cx, cy, s, _ in cands)


def main():
    rng = random.Random(0)
    loc = dec = n = 0
    for pos in POS:
        for cell in CELL:
            for rot in ROT:
                payload = rng.choice(PAYLOADS)
                p = synth.SceneParams(canvas=(900, 1100), cell=cell, pos=pos,
                                      rotation_deg=rot, skew_deg=rng.uniform(-12, 12),
                                      defects=True, text=True, edges=True,
                                      chip=rng.random() < 0.3)
                img, truth = synth.scene(payload, p, rng)
                g = img[..., 0]
                n += 1
                if _hit(propose(g), truth):
                    loc += 1
                if recover(g) == payload:
                    dec += 1
    print(f"synthetic scenes: {n}")
    print(f"localization recall: {loc}/{n} = {loc/n:.3f}")
    print(f"decode rate        : {dec}/{n} = {dec/n:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement `tools/validate_pathology.py`**

```python
"""Regression guard for the 28 decodable corpus/pathology_samples (pseudo-GT = the value
the current Reader decodes). Run before/after register changes; the count must not drop.

    .venv/bin/python -m tools.validate_pathology
"""
import glob
import os
import numpy as np
import cv2
from PIL import Image
from datamatrix_reader.reader import Reader


def load(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return g if g is not None else np.array(Image.open(p).convert("L"))


def main():
    rd = Reader()
    ok = 0
    files = sorted(glob.glob("corpus/pathology_samples/*"))
    for p in files:
        g = load(p)
        if g is None:
            continue
        if rd.read(g).payload is not None:
            ok += 1
    print(f"pathology_samples decoded: {ok}/{len(files)}  (baseline 28; must not drop)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the harnesses to set the Phase-1 baseline**

Run: `.venv/bin/python -m tools.validate_pathology`
Expected: `pathology_samples decoded: 28/36` (must stay ≥28).

Run: `.venv/bin/python -m tools.validate_synth`
Record the printed localization recall and decode rate — these are the Phase-1 baseline numbers and the bar Phase 2 must not drop below. (Diagnose any axis far below the rest; if localization recall is the limiter, tune `locate` thresholds; if decode lags localization, the registration search is the gap — that's Phase 2's job.)

- [ ] **Step 4: Commit**

```bash
git add tools/validate_synth.py tools/validate_pathology.py
git commit -m "test(harness): synthetic generalization + pathology regression gates"
```

---

# PHASE 2 — Efficiency (score-guided registration)

## Task 6: `score_registration` — cheap, no-zxing registration score

**Files:**
- Modify: `src/datamatrix_reader/register.py` (add `score_registration`)
- Test: `tests/test_register.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_register.py`:

```python
def test_score_peaks_at_true_registration():
    from datamatrix_reader.register import score_registration
    payload, dark = _square_symbol()          # existing helper -> (payload, MxM bool)
    M = dark.shape[0]
    img = _canvas(dark, cell=20, quiet=4)      # existing helper
    H, W = img.shape
    cx, cy = W / 2.0, H / 2.0
    true = score_registration(img, cx, cy, 20.0, M, 0.0)
    # mis-registered by half a module / wrong pitch / off-center must score lower
    assert true > score_registration(img, cx + 10, cy, 20.0, M, 0.0)
    assert true > score_registration(img, cx, cy, 26.0, M, 0.0)
    assert true > score_registration(img, cx, cy, 20.0, M, 8.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_register.py::test_score_peaks_at_true_registration -v`
Expected: FAIL — `score_registration` undefined.

- [ ] **Step 3: Implement `score_registration` in `src/datamatrix_reader/register.py`**

```python
# weights tuned in Task 8 against WSI + synthetic; start here.
_W_L, _W_QUIET, _W_BIMODAL = 1.5, 1.0, 0.5


def score_registration(gray, cx, cy, cell, M, deg):
    """Cheap registration quality score in [~0, 3], NO zxing. Peaks at the true grid:
      L_solidity   - best of 4 orientations' (left col + bottom row) dark fraction
      quiet_white  - 1-module ring just OUTSIDE the MxM should be light
      bimodality   - interior data ~50% dark (not a uniform patch)."""
    grid = sample_fast(gray, cx, cy, cell, M, deg)
    l = max((np.rot90(grid, k)[:, 0].mean() + np.rot90(grid, k)[-1, :].mean()) / 2.0
            for k in range(4))
    outer = sample_fast(gray, cx, cy, cell, M + 2, deg)      # ring = the M+2 border
    ring = np.concatenate([outer[0, :], outer[-1, :], outer[:, 0], outer[:, -1]])
    quiet_white = 1.0 - float(ring.mean())
    bimodal = 1.0 - abs(float(grid.mean()) - 0.5) * 2.0
    return _W_L * l + _W_QUIET * quiet_white + _W_BIMODAL * bimodal
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_register.py::test_score_peaks_at_true_registration -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/datamatrix_reader/register.py tests/test_register.py
git commit -m "feat(register): score_registration — cheap no-zxing grid quality score"
```

---

## Task 7: Score-guided `register_candidate` with brute-force backstop

**Files:**
- Modify: `src/datamatrix_reader/register.py` (add `register_candidate`; call it from `decode_auto`)
- Test: `tests/test_register.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_register.py`:

```python
def test_register_candidate_matches_bruteforce():
    import random
    from datamatrix_reader import synth
    from datamatrix_reader.register import recover
    rng = random.Random(7)
    payload = None
    for t in (b"DMTXSLIDE-GUIDED-TEST1", b"ABCDEFGHIJKLMNOPQRSTUVWX"):
        import zxingcpp
        a = np.asarray(zxingcpp.create_barcode(t.decode(),
              zxingcpp.BarcodeFormat.DataMatrix).to_image())
        if a.shape[0] == a.shape[1]:
            payload = t; break
    # several damaged scenes must still decode under the score-guided path
    ok = 0
    for i in range(6):
        p = synth.SceneParams(canvas=(800, 900), cell=16 + i, pos=(0.4, 0.55),
                              rotation_deg=90.0 * (i % 4), defects=True, text=True)
        img, _ = synth.scene(payload, p, rng)
        if recover(img[..., 0]) == payload:
            ok += 1
    assert ok >= 5
```

- [ ] **Step 2: Run test to verify it fails (or is slow)**

Run: `.venv/bin/python -m pytest tests/test_register.py::test_register_candidate_matches_bruteforce -v`
Expected: PASS but SLOW (current brute-force `decode_auto`), or FAIL if a case misses. Either way, proceed — Task 7 makes it fast while keeping it passing.

- [ ] **Step 3: Add `register_candidate` and call it from `decode_auto`**

Add to `src/datamatrix_reader/register.py`:

```python
def _brute_region(gray, cx, cy, te, ang):
    """The exhaustive search (today's decode_auto body) for ONE region — the backstop."""
    for M in SIZES:
        for cell in np.arange(te / (M + 3), te / (M - 1), 0.5):
            for ddeg in np.arange(-3, 3.01, 1.0):
                for dcx in np.arange(-1.5, 1.51, 0.375) * cell:
                    for dcy in np.arange(-1.5, 1.51, 0.375) * cell:
                        grid = sample_fast(gray, cx + dcx, cy + dcy, cell, M, ang + ddeg)
                        for g, lsc, _ in l_orientations(grid):
                            if lsc < 0.6:
                                break
                            try:
                                p = _zxing(render_symbol(g, M))
                            except cv2.error:
                                continue
                            if p is not None:
                                return p
    return None


def register_candidate(gray, cx, cy, te, ang, top_k=4):
    """Score-guided registration for ONE region, with a brute-force backstop.
    Coarse-to-fine maximize score_registration over (center, cell, angle) per M; decode
    the top_k highest-scoring hypotheses; if none decode, fall back to _brute_region."""
    scored = []
    for M in SIZES:
        cell0 = te / M
        best = (cx, cy, cell0, ang)
        # two coarse-to-fine rounds of coordinate refinement
        for step_c, step_cell, step_a in ((0.5 * cell0, 1.0, 1.5), (0.18 * cell0, 0.5, 0.7)):
            bx, by, bc, ba = best
            cand = [(bx + dx, by + dy, bc + dc, ba + da)
                    for dx in (-step_c, 0, step_c) for dy in (-step_c, 0, step_c)
                    for dc in (-step_cell, 0, step_cell) for da in (-step_a, 0, step_a)]
            best = max(cand, key=lambda v: score_registration(gray, v[0], v[1], v[2], M, v[3]))
        s = score_registration(gray, best[0], best[1], best[2], M, best[3])
        scored.append((s, M, best))
    scored.sort(key=lambda v: -v[0])
    for _, M, (rx, ry, rc, ra) in scored[:top_k]:
        grid = sample_fast(gray, rx, ry, rc, M, ra)
        for g, lsc, _ in l_orientations(grid):
            if lsc < 0.6:
                break
            try:
                p = _zxing(render_symbol(g, M))
            except cv2.error:
                continue
            if p is not None:
                return p
    return _brute_region(gray, cx, cy, te, ang)        # backstop — recall can't regress
```

Then change `decode_auto` to delegate per region:

```python
def decode_auto(gray):
    regions = [r for r in (detect_area(gray), detect_data_region(gray)) if r]
    for cx, cy, te, ang in regions:
        p = register_candidate(gray, cx, cy, te, ang)
        if p is not None:
            return p, {"region": (round(cx, 1), round(cy, 1), round(te, 1))}
    return None, None
```

- [ ] **Step 4: Run tests + WSI regression + timing**

Run: `.venv/bin/python -m pytest tests/test_register.py -v`
Expected: all PASS, and `test_register_candidate_matches_bruteforce` is fast now.

Run: `.venv/bin/python -m tools.validate_full 2>&1 | tail -8`
Expected: `404/404`, `WRONG 0`, and `max` per-label time materially lower than the ~4s baseline.

- [ ] **Step 5: Commit**

```bash
git add src/datamatrix_reader/register.py tests/test_register.py
git commit -m "perf(register): score-guided registration + brute-force backstop"
```

---

## Task 8: Tune weights; confirm efficiency + no regression

**Files:**
- Modify: `src/datamatrix_reader/register.py:_W_L,_W_QUIET,_W_BIMODAL` (only if needed)

- [ ] **Step 1: Measure current state**

Run: `.venv/bin/python -m tools.validate_full 2>&1 | tail -3`
Run: `.venv/bin/python -m tools.validate_synth`
Record decode rate + timing. Target: WSI 404/404 held; synthetic decode rate ≥ Phase-1 baseline (Task 5); median fallback time sub-second on WSI.

- [ ] **Step 2: If synthetic decode dropped below Phase-1 baseline, widen the safety net**

Increase `top_k` (e.g., 4 → 6) in `register_candidate`'s signature default, OR adjust `_W_L/_W_QUIET/_W_BIMODAL` so the true grid ranks in the top-K. Re-run both harnesses. The brute-force backstop guarantees correctness; this step only recovers *speed* lost to the backstop firing too often.

- [ ] **Step 3: Commit (only if changed)**

```bash
git add src/datamatrix_reader/register.py
git commit -m "perf(register): tune score weights / top_k for backstop hit rate"
```

---

# PHASE 3 — Justify `gradient` on data

## Task 9: Ablate `gradient` on synthetic edge/chip cases; keep or drop

**Files:**
- Create: `tools/ablate_gradient.py`
- Modify: `src/datamatrix_reader/register.py` (`decode_auto` region list — only if dropping)

- [ ] **Step 1: Implement `tools/ablate_gradient.py`**

```python
"""Decide whether the gradient-anisotropy detector earns its place: run the
edge/chip-confounded synthetic scenes with gradient IN vs OUT and report decode delta.

    .venv/bin/python -m tools.ablate_gradient
"""
import random
import numpy as np
import zxingcpp
from datamatrix_reader import synth, register

_DM = zxingcpp.BarcodeFormat.DataMatrix
PAYLOAD = next(p for p in (b"S25-04821 A3-1 HE", b"ABCDEFGHIJKLMNOPQRSTUVWX")
               if np.asarray(zxingcpp.create_barcode(p.decode(), _DM).to_image()).shape[0]
               == np.asarray(zxingcpp.create_barcode(p.decode(), _DM).to_image()).shape[1])


def _run(use_gradient):
    from datamatrix_reader.locate import propose
    rng = random.Random(0)
    dets = [register.detect_area, register.detect_data_region] if use_gradient \
        else [register.detect_data_region]
    ok = n = 0
    for _ in range(40):
        p = synth.SceneParams(canvas=(900, 1100), cell=rng.choice([12, 18, 26]),
                              pos=(rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)),
                              rotation_deg=rng.choice([0, 90, 180, 270]),
                              defects=True, text=True, edges=True, chip=rng.random() < 0.5)
        img, _ = synth.scene(PAYLOAD, p, rng)
        g = img[..., 0]
        got = None
        for cx, cy, size, _ in propose(g):
            up = register._normalize(g, cx, cy, size)
            if up is None:
                continue
            for d in dets:
                r = d(up)
                if r and (got := register.register_candidate(up, *r)):
                    break
            if got:
                break
        n += 1
        ok += (got == PAYLOAD)
    return ok, n


def main():
    a, n = _run(True)
    b, _ = _run(False)
    print(f"with gradient   : {a}/{n}")
    print(f"texture only    : {b}/{n}")
    print(f"gradient delta  : {a - b}  -> {'KEEP' if a - b > 0 else 'DROP candidate'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and decide**

Run: `.venv/bin/python -m tools.ablate_gradient`
Expected: a delta. If `with gradient` > `texture only`, **keep** gradient (no code change). If equal across a few seeds, **drop** it: change `decode_auto`'s region list to `[r for r in (detect_data_region(gray),) if r]` and re-run `validate_full` (must stay 404/404) + `validate_synth` (must not drop).

- [ ] **Step 3: Commit**

```bash
git add tools/ablate_gradient.py src/datamatrix_reader/register.py
git commit -m "test(ablate): justify gradient detector on synthetic edge/chip scenes"
```

---

## Final verification (after all phases)

- [ ] Run full suite: `.venv/bin/python -m pytest -q` — all green.
- [ ] WSI gate: `.venv/bin/python -m tools.validate_full` — 404/404, WRONG 0.
- [ ] Pathology gate: `.venv/bin/python -m tools.validate_pathology` — ≥28/36.
- [ ] Generalization: `.venv/bin/python -m tools.validate_synth` — localization recall + decode rate at/above Phase-1 baseline; report worst axis.
- [ ] Then use **superpowers:finishing-a-development-branch**.
