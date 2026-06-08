# dmtxslide — session handoff (2026-06-02)

Portable state doc so work can resume after moving the repo to a new disk.
This lives in the repo, so it travels with the files (the Claude memory dir does
**not** — see "Move gotchas").

## Where we are (the short version)

We bootstrapped the project, made the synthetic benchmark realistic, wired in real
DataMatrix data, and ran a head-to-head of decoders. The big finding **challenges
the project's founding thesis**:

A 3-fold comparison (`tools/compare_backends.py`) on the same images:

| dataset            | preprocess+libdmtx | libdmtx raw | zxing-cpp |
|--------------------|--------------------|-------------|-----------|
| synth (n=432)      | **0.77**           | 0.65        | 0.58      |
| BarBeR real (110)  | 0.31               | 0.55        | **0.80**  |
| pathology hits(36) | 14                 | 20          | **24**    |
| real p50 latency   | 258 ms             | 163 ms      | **17 ms** |

- Our **localize/cascade preprocessing OVERFITS synth**: helps +12pts on synth but
  is **net-negative on every real set** (BarBeR 0.31 vs 0.55 raw).
- **zxing-cpp wins on all real data** and is ~10–15x faster, and it reads QR + linear
  too (a one-stop shop). It is currently an optional dep (`pip install -e ".[compare]"`),
  not the core.
- Caveat: zxing is *last* on synth (0.58); synth is the least trustworthy here.

## The open decision (next session starts here)

**Should we re-anchor the project on zxing-cpp** (decode core), demote preprocessing
to a hard-tail fallback used only when raw zxing fails *and* only if proven on real
data, and refocus our value-add on bounded-latency orchestration + domain
(accession-format) validation + multi-code/multi-symbology aggregation?

**The tiebreaker is still pending:** real **Grundium Ocus** slide captures where the
scanner + `dmtxread` failed. User will produce them later (redaction guidance below).
When they land:
```
python -m tools.compare_backends --corpus <grundium_corpus> --budget 250
python -m tools.compare_backends --pathology <dir>      # if no ground-truth payloads
```
If zxing reads codes that dmtxread/libdmtx can't, that's close to dispositive.

## Parked (specced, not done)

- **Localization quiet-zone scrub** — design at
  `dmtxslide/docs/superpowers/specs/2026-06-01-localize-quiet-zone-scrub-design.md`.
  NOTE: the 3-fold result casts doubt on whether the localize front-end should be
  invested in at all — revisit the spec in light of "preprocessing hurts on real."

## Redacting the Grundium captures (PHI / patient names)

Measured safe rules (a black bar can break decoding only if it touches the code):
- A black bar is fine **as long as it leaves ≥1–2 module clear gap** from the code.
- **Safest = fill the name region with the label's background colour** (no contrast
  edge; decode-neutral even touching the code edge).
- Overlapping the code's modules always fails. Don't blur near the code.
- Self-check: redact a control code that *does* read; if it still reads, you're neutral.

## Environment rebuild (after the move)

Python 3.12 (3.14 has no opencv wheel). No native system dependencies required.

```bash
cd <newpath>/dmtxslide
# .venv is NOT relocatable (absolute paths baked in) — recreate it:
rm -rf .venv
python3.12 -m venv .venv
.venv/bin/pip install -e . pytest
# zxing-cpp ships as a pip wheel; no native build step.
# verify:
.venv/bin/python -m pytest tests/         # expect 28 passed
```

## Reproduce the key results

```bash
.venv/bin/python -m bench.harness --synth --per-cell 1 --budget 250 --out runs/baseline_v2.json
.venv/bin/python -m bench.harness --synth --dump-failures runs/failures   # inspect misses
.venv/bin/python -m tools.compare_backends --synth --per-cell 1 --budget 250
.venv/bin/python -m tools.compare_backends --corpus corpus/barber --budget 250
.venv/bin/python -m tools.dump_synth_samples --out runs/synth_samples       # see synth images
```

## Move gotchas (READ THIS)

1. **The Claude memory dir is keyed to the repo's absolute path** —
   `~/.claude/projects/-Users-cornish-GitHub-datamatrix-reader/memory/`. If the repo
   moves to a new path, a new session derives a *different* dir and won't see those
   memories. Options: keep the same logical path (symlink), or copy that `memory/`
   folder to the new path-derived location. This HANDOFF.md is the portable backup.
2. **`.venv` is broken by any move** (absolute paths) — delete + recreate (above).
   It's gitignored, so don't rely on copying it.
3. **Big data (`corpus/public/` = 9.8 GB)** is the space hog and is gitignored. It
   holds the original dataset zips. `corpus/barber/images/` (139 MB) is regenerable
   from the BarBeR zip via `tools/import_barber.py`. Keep `BarBeR_Dataset.zip` if you
   want to re-derive; the rest (retail/QR sets) we set aside as non-representative.

## Key files

- `src/dmtxslide/` — reader (zxing-cpp 2-stage cascade), synth, validate
- `bench/harness.py` — stratified read-rate bench (+ `--dump-failures`)
- `tools/compare_backends.py` — the 3-fold decoder comparison
- `tools/import_barber.py` — BarBeR zip -> DataMatrix corpus
- `tools/dump_synth_samples.py` — materialize synth images to disk
- `tests/` — 28 tests (reader raw/clahe stages, synth crowding/ink-fairness/non-degeneracy + encoder round-trip, payload spread, failure-dump, label_gt, corpus loader)
