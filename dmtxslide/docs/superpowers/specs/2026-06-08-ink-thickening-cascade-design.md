# Progressive ink-thickening fallback cascade — design

**Date:** 2026-06-08
**Status:** approved (brainstorming) → ready for implementation plan
**Builds on:** the zxing re-anchor (`Reader` is a 2-stage zxing cascade).

## Problem

The shipped `Reader` (raw zxing → `up2+CLAHE`) reads 374/404 = 0.926 of the real
WSI labels. The misses are overwhelmingly **faint / under-inked printing** that has
degraded the DataMatrix finder and timing patterns (operator-labelled: faint 40/54,
broken-finder 19/54; capture defects ~none). Two independent exhaustive preprocessing
sweeps converged on one generalizable lever: **progressively thickening the dark ink**
(upscale → strong CLAHE → Otsu → morphological erosion to grow the dark modules)
recovers the faint codes, reaching ~0.975–0.988 — open, no commercial decoder, with
**zero false decodes** (DataMatrix Reed–Solomon ECC makes a successful decode
self-validating).

So: add full-frame, fallback-only stages to `Reader` that escalate ink-thickening
until the code decodes.

## Goals

1. Lift the reader from 0.926 toward ~0.98 on the WSI labels using a **principled,
   generalizable** ink-thickening ladder (not a corpus-fit grab-bag of pipelines).
2. Keep it **fallback-only**: stages run only when prior stages miss, so p50 stays
   ~3 ms and the 374 raw/clahe reads are never touched.
3. Stay open / dependency-free (OpenCV + zxing only; no commercial SDK).

## Non-goals

- Chasing the last ~0.5% with one-off pipelines (sauv_darken etc.) — overfits the
  corpus; explicitly dropped (see "Composition decision").
- A crop/localizer stage — the full-frame ladder reaches the ceiling without one.
- A false-positive guard — unnecessary (ECC); see "No FP guard".
- A `deep`/speed toggle — no real use case in this latency-insensitive deployment;
  the ladder is pure-upside fallback, so it always runs.

## Composition decision (why the principled ladder)

The exhaustive sweeps showed the winners are all the *same idea at escalating
strength*: `upscale → CLAHE(4) → Otsu → erode(N)`, with larger `N`/upscale recovering
fainter codes (`erode×1` +16, `erode×2` +5). We encode **that family** as the ladder.
We deliberately exclude the odd one-off winners (e.g. `unsharp+sauvola+darken`, +4)
that add ~0.5% on *this* corpus but have no clean mechanism and weak generalization.
The ladder reaches ~0.988 here anyway, via the escalation itself.

## Architecture

`Reader.read()` runs raw zxing, then — only on a miss — iterates an ordered list of
full-frame preprocessing stages, decoding after each; first success wins.

```
raw    zxing(grayscale)                                       ~350
clahe  up2 → CLAHE(2.0)                            [existing]  +24
ink1   up2 → CLAHE(4.0) → Otsu → thicken(iter=1)              +16
ink2   up4 → CLAHE(4.0) → Otsu → thicken(iter=2)              +5  (→ ~0.988)
```
`thicken` = `cv2.erode` with a 2×2 kernel, which shrinks bright regions / **grows the
dark modules** (lays down more ink), reconnecting broken finder/timing lines and
filling thin modules.

## Components

### `src/dmtxslide/preprocess.py` (new)

Small, pure, no zxing — just the stage transforms and their ordered list:

```python
CLAHE_TILE = (8, 8)

def _up(g, f):        cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
def _clahe(g, clip):  cv2.createCLAHE(clip, CLAHE_TILE).apply(g)
def _otsu(g):         cv2.threshold(g, 0, 255, THRESH_BINARY|THRESH_OTSU)[1]
def _thicken(b, it):  cv2.erode(b, np.ones((2,2), np.uint8), iterations=it)

def s_clahe(g): return _clahe(_up(g, 2), 2.0)
def s_ink1(g):  return _thicken(_otsu(_clahe(_up(g, 2), 4.0)), 1)
def s_ink2(g):  return _thicken(_otsu(_clahe(_up(g, 4), 4.0)), 2)

# ordered fallback stages: (name, transform: gray -> gray)
STAGES = [("clahe", s_clahe), ("ink1", s_ink1), ("ink2", s_ink2)]
```
All take and return a 2-D uint8 grayscale/binary image. Each is independently
testable. (`s_clahe` reproduces the current shipped stage 2, now sourced here so the
whole ladder lives in one place.)

### `src/dmtxslide/reader.py` (modify)

- Remove the inline `clahe` constants/logic (moved to `preprocess.py`); import
  `from .preprocess import STAGES`.
- `ReadResult` unchanged: `payload: bytes|None`, `stage: str|None`, `elapsed_ms`,
  `.ok`. `stage` values expand to `{"raw","clahe","ink1","ink2",None}`.
- `read(self, image, budget_ms=250.0)`:
  ```
  gray = _gray(image)
  payload = _zxing(gray); stage = "raw" if payload else None
  if payload is None:
      for name, fn in STAGES:
          payload = _zxing(fn(gray))
          if payload is not None:
              stage = name; break
  return ReadResult(payload, stage, elapsed)
  ```
- `budget_ms` still accepted-but-ignored. No constructor args (no `deep` flag).

## No FP guard (trust ECC)

DataMatrix carries Reed–Solomon ECC, so a *successful* decode is self-validating — a
corrupted/over-thickened image yields a decode *failure*, not a wrong payload. Across
all aggressive preprocessing in the sweeps we observed **zero** false decodes. So no
agreement/format guard is added. Accession-format validation (`validate.py`,
`is_valid_payload`) remains a **caller-side** concern, keeping `Reader`
domain-agnostic. (If fresh data ever shows a false decode, revisit with a two-stage
agreement check — noted, not built.)

## Overfitting honesty (first-class)

The stage parameters (clip 4, erode 2×2, iters 1–2, upscale 2/4) are **tuned on the
wsi_labels corpus**. The *principle* (progressive ink-thickening for faint codes)
generalizes; the exact numbers may not. Therefore:
- The reader docstring and spec state the ~0.98 figure is a **wsi_labels baseline**.
- **Acceptance includes re-validation on fresh captures**: when a new batch lands,
  run `python -m tools.compare_backends --corpus <batch>` and record raw-vs-cascade.
  If the ladder under-performs or (unexpectedly) mis-decodes on new data, the params
  are revisited before trusting the number.

## Data flow

`bench` / `compare_backends` / callers → `Reader.read(bgr)` → grayscale → zxing →
(miss) `s_clahe` → (miss) `s_ink1` → (miss) `s_ink2` → `ReadResult(payload, stage)`.
`compare_backends`'s `zxing cascade` fold already calls `Reader`, so it measures the
new ladder with no change.

## Error handling

- zxing returns `[]` → stage yields `None`; the loop continues to the next stage; all
  miss → `payload=None, stage=None, .ok=False`.
- A stage transform on a tiny/degenerate image is wrapped so an OpenCV error is
  treated as a miss (continue to next stage), never a crash.
- Empty/garbage image → all stages miss → not ok.

## Testing strategy

1. **`preprocess.py` units:** each `s_*` returns a 2-D uint8 image; `_thicken`
   increases the dark-pixel count (grows ink); a clean zxing-encoded code, run through
   each stage then decoded, still returns the payload OR (acceptably) fails — the
   assertion is "no crash, correct dtype/shape," since stages target degraded input.
2. **`Reader` cascade:** (a) a clean code → `stage=="raw"`; (b) monkeypatch `_zxing`
   to miss on raw+clahe and hit on the 3rd call → `stage=="ink1"`, and confirm stages
   run in order and stop at first hit; (c) blank image → `.ok is False, stage is
   None`; (d) a synth code degraded enough to fail raw but decode via an ink stage →
   `.ok` and `stage in {"clahe","ink1","ink2"}`.
3. **Regression:** full `pytest tests/` green; `compare_backends --corpus
   corpus/wsi_labels` reports cascade ≈ 0.975–0.988 vs raw ≈ 0.866, p50 ~3 ms.
4. **No false-positive check in CI** (can't synthesize), but the acceptance run
   asserts `WRONG == 0` on wsi_labels.

## Risks / notes

- **Overfitting** — addressed above (principled family + fresh-data acceptance).
- **Latency on the tail** — `ink2` upscales 4× (16 MP intermediate); only the ~5% of
  images that reach it pay ~100 ms. Acceptable for slide scanning; bounded.
- Memory `[[clahe-upscale-fallback-recovers-poor-codes]]` and the resume pointer
  should be updated once this lands (the reader is no longer 2-stage).
