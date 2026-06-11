# DataMatrix grid-reconstruction decoder — design

> **OUTCOME: BUILT & SHELVED (2026-06-09).** The render→zxing keystone works and the
> decoder recovers finder-erased *synthetic* codes (ECC-safe, WRONG=0). But the
> Task-5 corpus gate failed: **2/404 on the real WSI corpus, recovers 0 residual** —
> `localize` doesn't transfer to real busy labels (quad on ~7/20 trivially-readable
> codes). NOT integrated. Implementation lives on branch `feat/grid-decode` (shelved).
> Revisit needs a robust real-image code-region detector. See
> [[grid-reconstruction-shelved]].

**Date:** 2026-06-09
**Status:** approved (brainstorming) → ready for implementation plan
**Builds on:** the ink-thickening cascade (Reader reaches 0.983; the residual ~8 are
faint/**broken-finder**/damage codes that no whole-image preprocessing recovers).

## Problem

The cascade's residual is dominated by **broken finder/timing patterns** — faint
printing has degraded the very structure (the solid "L" + dashed timing) that
classical decoders use to *locate and grid* the symbol. The feasibility probe found
these finders are **degraded, not absent**: the code's square extent + partial
finder/timing are still visible. So the fix is a fundamentally different decode path
that samples the module grid directly instead of binarizing the whole image.

## Key idea: reconstruct, then delegate

We build **only the novel front-end** — image → a clean N×N module bit-matrix — and
hand the rest to zxing. Once the border is overwritten with the *known-correct*
finder/timing and the interior holds our sampled data bits, we render a pristine
1px/module image and call `zxing.read_barcodes` on it. zxing does the ECC200
codeword layout + Reed–Solomon error correction (it nails clean input), and its ECC
**validates** the result — a wrong reconstruction simply fails to decode, never
mis-reads. We do **not** reimplement any DataMatrix back-end.

Because zxing validates, we can **brute-force** the unknowns (symbol size, rotation)
and accept the first decode — no precise orientation/size detection needed.

## Goals

1. A standalone `griddecode.decode(image) -> bytes | None` that recovers
   broken-finder codes the cascade misses, with **zero false decodes**.
2. Validate on the **full real corpus (n=404)**, not just the ~4 residual: WRONG=0 is
   the hard gate; report how many of the 8 cascade-residual it recovers.
3. Wire it in as the Reader's **final fallback stage** (after the ink cascade) *only
   if* it clears the gate.
4. No new dependencies (OpenCV + zxing only); no Reed–Solomon code of our own.

## Non-goals

- Reimplementing ECC200 layout / de-randomization / RS decode (delegated to zxing).
- Rectangular DataMatrix symbols (the WSI codes are square; square sizes only in v1).
- Decoding codes with no detectable square extent (genuinely destroyed — out of reach).

## Architecture

Standalone module `src/dmtxslide/griddecode.py`:

```
decode(image) -> bytes | None:
    roi  = grayscale(image)                       # caller may pass a crop or full frame
    quad = localize(roi)                          # 4 corners of the code's square
    if quad is None: return None
    warp = perspective_warp(roi, quad, SIDE)      # square buffer, SIDE px (e.g. 480)
    for N in SQUARE_SIZES:                         # 10,12,...,144
        grid = sample_modules(warp, N)            # N x N booleans, per-module threshold
        for r in (0, 90, 180, 270):
            sym = render_symbol(rotate(grid, r), N)   # force L-finder+timing, render 1px/module
            payload = zxing(sym)                      # ECC-validated
            if payload is not None:
                return payload
    return None
```

## Components (each independently testable)

- **`localize(gray) -> quad|None`** — Otsu/adaptive threshold → morph-close → largest
  square-ish contour (`minAreaRect`, aspect ~1, fill ratio high) → 4 corners. Returns
  None if no plausible square blob.
- **`perspective_warp(gray, quad, side) -> ndarray`** — `getPerspectiveTransform` +
  `warpPerspective` to a `side×side` upright square.
- **`sample_modules(warp, N) -> np.ndarray(bool, (N,N))`** — module center at
  `((i+.5)*side/N, (j+.5)*side/N)`; sample a small window mean; per-module dark/light
  by Otsu over the N×N means (dark = data bit set). The grid-aware sampling is the
  whole point — it decides each module from its own center, not a global binarization.
- **`render_symbol(grid, N) -> ndarray`** — takes the N×N sampled `grid`, **overwrites
  the border** (left column + bottom row → solid dark = the L finder; top row + right
  column → alternating = timing) while keeping the sampled interior values, and renders
  the whole N×N as a 1px/module uint8 image. (We never interpret *data* placement —
  zxing does. The 4-rotation brute force is on `grid` *before* this, since the forced
  border only matches the true symbol in one grid orientation; zxing's own image-rotate
  cannot fix a wrong data↔finder mapping.)
- **`zxing(img) -> bytes|None`** — `read_barcodes(ascontiguous(img), formats=DataMatrix)`,
  return `[0].bytes` or None. (Reused idiom from reader.)

`SQUARE_SIZES = [10,12,14,16,18,20,22,24,26,32,36,40,44,48,52,64,72,80,88,96,104,120,132,144]`.

## Data flow / integration

Standalone first (validated on all 404 via a bench script). If it clears the gate,
add a `("grid", lambda g: ...)`-style final fallback so `Reader.read` calls
`griddecode.decode(gray)` after the ink stages miss — `ReadResult.stage = "grid"`.
Integration is a follow-on step in the plan, gated on validation.

## Error handling

- No square blob found → `localize` returns None → `decode` returns None.
- Degenerate warp / tiny N → wrapped; treated as a miss, continue.
- Every `zxing` call is ECC-validated, so a mis-sampled grid yields a decode *failure*,
  not a wrong payload. The brute force can never return an incorrect read.

## Validation strategy (the gate)

1. **Real corpus, n=404 (primary):** run `griddecode.decode` standalone on every WSI
   label. Report: correct decodes, **WRONG (must be 0)**, and how many of the 8
   cascade-residual it recovers. WRONG=0 across 404 is the ship gate.
2. **Synthetic stress (breadth):** encode synth codes, programmatically **erase/degrade
   the finder + timing** (and faint the modules), confirm `griddecode` recovers a broad
   range where the ink cascade fails. This tests the *capability* generally, beyond the
   handful of real broken-finder codes.
3. **Ship only if** WRONG=0 on 404 AND it recovers ≥1 real residual code AND broad
   synthetic recovery. Then integrate as the Reader's final fallback and re-run the
   end-to-end acceptance (cascade+grid ≥ 0.983, WRONG=0).

## Risks / notes

- **Sampling accuracy is the real research risk:** the sampled grid must land within
  RS-correctable error of the true matrix. Mitigated by brute-forcing size/rotation +
  sub-window sampling; if it can't get *any* of the residual, that's the honest finding
  (and we stop — the residual goes to human-flag).
- **Quad detection fails on the damage case** (scan_432's blob over the finder) — likely
  unrecoverable; acceptable.
- **The plan must START with a keystone spike:** prove `localize→sample→render→zxing`
  decodes at least the *easy* codes (sanity) and ≥1 residual, before building out the
  full module + integration. If the spike fails, pivot or stop early.
- This is R&D: a real chance it recovers only 1–2 of the 8. That's still a win
  (capability + generalization to future faint batches), but the gate is WRONG=0, not a
  target count.
