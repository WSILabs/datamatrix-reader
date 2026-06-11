# Design: lazy quiet-zone scrub in `localize.py`

**Date:** 2026-06-01
**Status:** approved (pending implementation plan)
**Area:** `src/dmtxslide/localize.py` (+ its contract with `reader.py`)

## Problem

The synthetic baseline (`runs/baseline.json`, 432 samples) shows decode failure
concentrated at low pixels-per-module, worst at `module_px=3.0`
(found_rate 0.24, correct_rate 0.17). The per-stratum found-vs-correct split
(added to `bench/harness.py`) shows this stratum is **localization-limited**, not
decode-limited.

### Root cause (empirically confirmed)

The failure is **not** small modules / resolution:

- A clean 3px-module code decodes directly via libdmtx.
- Upscaling the failing crops 2x/3x rescues **0/18** cases.

The failure is **quiet-zone crowding**. The harness always sets
`quiet_crowd=True` (text printed against the code, realistic for direct-print
slide labels). The blob finder's `MORPH_CLOSE (9x9, x2)` bridges the code and the
adjacent text into a single blob, so the cropped ROI contains text pressed
against the code edge. `_quiet_pad` pads the *outside* of that crop, but the text
sits *inside* it, touching the code — so libdmtx never sees a clean quiet zone on
that side and the region find fails. At low px/module the symbol is physically
small, so a fixed amount of stuck-on text consumes a larger fraction of the quiet
zone; this is why 3px is the worst stratum and 10px is immune.

Reproduction: same 3px code with crowding off → decodes on rung 0
(crop 72x72, score 1.00); with crowding on → `found=False`
(crop 79x79, score 0.86, text bridged in).

## Constraint

Chosen acceptance bar: **recall first, protect p50.** Maximize found_rate at the
weak strata, but the clean-code / fast path must not regress meaningfully
(matches the README's "without slowing rung 0"). Pareto-gated via
`bench/report.py`: a change is accepted only if it lifts weak strata without
regressing others.

## Approach: B delivered via C

Quiet-zone scrub (B), with the scrubbed candidates generated lazily (C) so the
fast path is untouched. Plus one optional cheap morphology tweak (A), kept only
if the benchmark confirms it.

### 1. Two-tier, lazy candidate generation (C)

`localize()` returns a **generator** instead of a list. It yields, in order:

- **Tier 1 — cheap (today's candidates):** blob crops + the whole-frame
  fallback, exactly as now.
- **Tier 2 — scrubbed isolation crops:** computed **only if the consumer
  iterates past Tier 1.**

`reader.read` already iterates `for cand in candidates: ... if res.ok: return`.
A clean code succeeds on a Tier-1 candidate and returns before Tier 2 is
computed, so **p50 / the clean-code path is untouched**. The cheap blob
detection (Scharr → Otsu → contours) runs once; its `minAreaRect` results are
reused to build Tier 2, so detection is not repeated.

### 2. The scrub (B)

For each blob crop, when (and only when) Tier 2 is reached:

1. **Find the symbol's own square inside the crop, decoder-free.** The barcode is
   a dense, near-square grid of fine features; crowding text is a thinner,
   horizontally-elongated strip near one edge. Build a texture/gradient-density
   map → threshold → connected components → select the component whose bounding
   box is largest **and** most square (squareness x area, or a comparable score).
   That box is the symbol.
2. **Erase everything outside that box to the background level** (robust median
   of the crop's border pixels), then apply the existing `_quiet_pad`.

Result: the code, centered, with clean space on all four sides. The scrub keys to
the *measured* symbol box, so it introduces no printer-specific constant.

### 3. Optional cheap tweak (A), decided empirically

Try lightening `MORPH_CLOSE` so code and text bridge less in the first place
(makes the square-finder's job easier). This is the one tunable constant, so keep
it **only if** the benchmark shows it helps and regresses nothing; otherwise the
scrub carries the fix alone.

### 4. Error handling

- If the square-finder finds nothing sensible (tiny/degenerate crop), that
  candidate is **skipped**, never returned blindly.
- The validator gate already guarantees a bad crop can only cause a *miss*, never
  a wrong read.
- The whole-frame **fallback is not scrubbed** (it may contain several objects).
- Scrub does not depend on libdmtx bbox (failing cases have `found=False`, so no
  bbox); it is texture-based on the crop. (Note: the `binding.py` bbox y-flip was
  separately fixed this session; the scrub does not rely on it.)

## Out of scope (YAGNI)

- No super-resolution / upscaling rung (disproven for this failure mode).
- No new cascade rungs; no changes to `cascade.py`, `adapt.py`, `synth.py`,
  `binding.py` decode logic.
- The decode-side `module_px=6.0` sampling gap (found 0.96, correct 0.77) is a
  separate, later concern.

## Acceptance / testing

- **Anchor test:** reproduce the 3px + `quiet_crowd=True` case (currently
  `found=False`); the scrubbed candidate must flip to decoded. Build against this
  first.
- **Unit tests** for the square-finder on synthetic crops: code centered, and
  code + adjacent text (assert the returned box excludes the text strip).
- **Benchmark gate:** run `bench.harness --synth --per-cell 2 --budget 250 --out
  runs/variant.json`, then `bench.report runs/baseline.json runs/variant.json`.
  Accept **only if**:
  - `module_px=3.0` (and ideally `ink_gain=2`) rise in **both** found_rate and
    correct_rate;
  - **no stratum regresses** in correct_rate;
  - **p50 does not meaningfully increase**.
  If the scrub underperforms, fall back to Approach A; the report makes the call
  explicit.

## Interface change summary

- `localize.localize(gray, ...)` return type: `list[Candidate]` →
  `Iterator[Candidate]` (lazy). `reader.read` consumption is unchanged (already
  iterates). Any other caller that indexes/`len()`s the result would need
  updating — none currently exists besides the reader and ad-hoc diagnostics.
