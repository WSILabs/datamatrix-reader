# Barcode-Repair Generalization & Efficiency — Design

**Status:** approved (brainstorm), pending implementation plan
**Date:** 2026-06-09
**Scope:** `src/datamatrix_reader/register.py` (the finder-registration fallback) + a new
localization front-end + synthetic validation.

## Goal

Make the DataMatrix barcode-repair fallback (1) **generalize** beyond the Grundium WSI
label format — recover a damaged code placed *anywhere* on a typical slide label, at
*variable (bounded) scale* and at any of the 4 cardinal orientations plus modest skew —
and (2) be **more efficient**: cut the ~0.5–4 s brute-force registration cost ahead of a
likely C port. Both, co-designed.

## Non-goals / fixed assumptions (from the brainstorm)

- **Slide-label domain only**, not arbitrary images.
- **Square symbols only** — rectangular DataMatrix is explicitly out of scope.
- **Bounded scale** — codes are "approximately our size, some smaller/larger" (~0.3–1.5×
  of canonical), not arbitrary scale.
- **Rotation** — the code sits at one of {0,90,180,270}° with skew that is never as
  extreme as 45° from a cardinal (slide-holder tolerance).
- **ECC-validated, WRONG must stay 0** — we reconstruct and delegate ECC200 + Reed-Solomon
  to zxing; a bad reconstruction fails, never mis-reads.
- **Pure-numpy / OpenCV, no scipy** — keep it C-portable.

## Background: current approach and what the ablation showed

Today `recover()` crops the full label to a hardcoded upper-left ROI, upscales 2×, then
`decode_auto()` runs a **union of three detectors** (gradient anisotropy, dark-ink extent,
data-region texture) and, per detected region, brute-forces
`M∈{18,20,22,24} × cell × angle(±3°) × center(±1.5 cell, 9×9)`, sampling each grid in one
`warpAffine`, gating orientation by an L-solidity test, then repainting the canonical
finder/timing and handing a clean symbol to zxing.

**Detector ablation on the DEPLOYED full-label path** (ROI-crop+2×), each detector alone:

| detector | decodes (of 7 WSI residual) | unique |
|---|---|---|
| **texture** (local std) | **7/7** | 329, 330 |
| gradient (`min(|Sobel_x|,|Sobel_y|)`) | 5/7 | — none — |
| dark-ink | 1/7 | — none — |

Findings that shape this design:
- **`dark` is dead weight** — 1/7, nothing unique, on both crops and full labels. **Drop it.**
- **`texture` is the workhorse** — 7/7 alone on the deployed path. The front-end is texture-led.
- **`gradient` adds nothing unique on these 7**, but its job — rejecting straight slide/label
  **edges** and the glass chip — is a robustness property these 7 codes don't stress. **Keep
  it as a conditional add-on, justified by synthetic edge/clutter data, not assumed.**
- The crop ablation was *misleading* (it credited `gradient` with uniquely getting 432);
  the full-label path is the truth. **All validation runs on full-label/synthetic conditions.**

## Architecture

New data flow (replaces the hardcoded ROI-crop+2× and the all-detectors-always brute force):

```
full label
  └─ PROPOSE: pyramid blob-detection → candidate (center, size, angle) regions
              anywhere on the label, any bounded scale, ranked best-first      [NEW]
       └─ for each candidate, best-first:
            NORMALIZE: crop around it + scale to canonical ~470px              [NEW; generalizes the fixed 2×]
            REGISTER:  texture-led detect → score-guided refine (+ backstop)   [search NEW; detectors trimmed]
            REPAINT:   render canonical border → zxing (ECC-validated)         [UNCHANGED]
       └─ first ECC-valid decode wins
```

**Key risk-reducer:** the coarse front-end absorbs all position/scale variability, so
everything downstream stays at the canonical scale the validated repair code already works
at. We add a front-end and swap the search engine; we do not rewrite the repair core.

**Module structure:**
- `src/datamatrix_reader/register.py` — canonical-scale repair core: detectors, `sample_fast`,
  `l_orientations`, `render_symbol`, score-guided + backstop registration, `decode_auto`.
  `dark` detector removed.
- `src/datamatrix_reader/locate.py` — **new** — `propose(gray) -> [(cx, cy, size, angle), ...]`
  (pyramid blob proposals). Kept separate so `register.py` stays focused.
- `recover()` (in `register.py`) — rewritten: `propose → per-candidate normalize →
  register → decode`. No hardcoded ROI/2×.
- `src/datamatrix_reader/synth.py` — extended to emit full-label scenes (below).
- `reader.py` — unchanged interface; still calls `recover()` as the `"autoreg"` fallback.

## Component 1 — Coarse proposal front-end (`locate.py`)

`propose(gray) -> list[(cx, cy, size, angle)]`, ranked best-first.

- Build a small **image pyramid** over a bounded scale range (~0.3–1.5× of canonical,
  ~4 levels) chosen so a code's module pitch lands near the canonical ~10 px at *some* level.
- At each level: **local-std texture-density map**, Otsu threshold, fixed small
  open (drop thin text/specks) + close (fill the module grid into one blob). Fixed kernels
  are valid because, at the matching level, the cell is ~canonical — so scale need not be
  known in advance.
- Connected components → `cv2.minAreaRect` → filter by squareness (`max/min < ~1.3`),
  minimum size, fill ratio.
- Map surviving rects to **native coords** (÷ level scale), collect across levels,
  **dedup** overlapping proposals (same code found at adjacent levels) by center/size
  proximity, keeping the strongest.
- **Rank** by texture-density / squareness so the most code-like region is tried first
  (early exit on first decode).

`minAreaRect`'s angle captures the skew; the gross 90/180/270 is resolved downstream by
the existing 4-way `l_orientations` render. Cost is Sobel/box-filter + connected components
per level — cheap; expensive register/decode runs only on the top proposals.

## Component 2 — Score-guided registration + backstop (`register.py`)

Per normalized candidate, replace the `M×cell×angle×center` zxing sweep with a cheap score
(no zxing), computed from one `sample_fast` + array stats:

```
score(center, cell, angle, M) =
      w1 * L_solidity(best of 4 orientations: left col + bottom row dark fraction)
    + w2 * quiet_zone_whiteness(1-module ring just OUTSIDE the M×M, want light)
    + w3 * interior_bimodality(data modules ~50% dark, bimodal — not a uniform patch)
```

- For each `M∈{18,20,22,24}`: `cell0 = extent/M` from detection.
- **Coarse-to-fine local search** around `(detected center, cell0, detected angle)`
  maximizing `score` (numpy coordinate-ascent / shrinking grid, no scipy).
- Decode the **top-K** scoring hypotheses (across M) via `render_symbol → zxing`.
  First ECC-valid wins.
- **Backstop:** if no top-K hypothesis decodes for this candidate, fall back to today's
  exhaustive brute-force for that candidate. Recall therefore **cannot regress**;
  score-guidance is a best-effort accelerator (~100× fewer zxing calls on the common case),
  not a correctness dependency.

The score peaks at the true registration even for a chip-broken L (still the most-solid L
at the true center; the quiet-zone term anchors placement). Weights `w1..w3` are tuned on
the WSI 7 + synthetic during Phase 2.

## Component 3 — Synthetic generation & validation

**Generator (extend `synth.py`):** render a known-payload square DataMatrix on a
label-like canvas at controlled **position (anywhere), scale (a range), rotation
({0,90,180,270} + skew ≤ ~20°)**, then layer the existing defects (half-printed top timing,
broken finder, faint ink) **plus the new confounders: a glass chip (bright blob on the
finder), straight slide/label edges, adjacent text clutter.** Truth = payload +
`(center, size, angle)`.

**Two metrics, measured separately:**
- **Localization recall** — `propose()` returns a candidate within tolerance of the true
  `(center, size)`. Tests the front-end independent of decode.
- **End-to-end decode rate** — `recover()` across each axis (position/scale/rotation).

This synthetic set is where **Phase 3 justifies `gradient`** (run hard edge/chip cases with
gradient in vs out; keep only if it measurably lifts decode).

**Regression guard (a gate):**
- WSI corpus stays **404/404, WRONG=0** (existing `tools/validate_full.py`).
- The **28 decodable `pathology_samples`** don't regress (decoded values as pseudo-GT) —
  new small harness.
- The synthetic set becomes a permanent regression test.

**Acceptance criteria:**
- *Generality:* synthetic localization recall and decode rate ≥ targets across all three
  axes (numbers set from the Phase-1 baseline run).
- *Efficiency:* median fallback time materially down (target sub-second on the WSI 7, vs
  0.5–4 s now), **with WSI 404/404 preserved** by the backstop.
- *No regression:* WSI 404/404, pathology 28/28, all tests green.

## Phasing (each phase ships working, testable software)

1. **Generality** — `locate.propose` + per-candidate normalize + the *existing*
   detectors/brute-force registration (texture+gradient, `dark` dropped); build the
   synthetic full-label generator + localization/decode harness. Proves "anywhere/any-scale"
   at low risk. Sets the Phase-1 baseline numbers.
2. **Efficiency** — score-guided registration + backstop; tune weights; measure speedup;
   prove no recall regression (WSI 404/404, synthetic ≥ Phase-1).
3. **Justify `gradient`** — synthetic edge/clutter ablation; keep or drop on data.

## Testing

- `propose()` localizes off-center / scaled / rotated synthetic codes within tolerance.
- score-guided register agrees with brute-force on sampled synthetic cases.
- existing PHI-free broken-border + straight-edge tests in `tests/test_register.py` stay green.
- `tools/validate_full.py` (WSI 404/404) and the new pathology pseudo-GT harness run clean.
