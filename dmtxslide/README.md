# datamatrix-reader

An **adaptive, source-agnostic DataMatrix reader** built on top of libdmtx,
aimed at the slide-label domain but designed to handle arbitrary printers,
symbol sizes, and label colours without per-deployment retuning.

libdmtx's decode core (grid sampling + Reed–Solomon) is sound. Its weaknesses
are upstream — region search robustness and speed — so this project adds an
adaptive front-end and a bounded decode cascade *around* libdmtx and does not
modify its internals. (libdmtx stays a git submodule under `third_party/` so
patching it remains possible, but you almost certainly won't need to.)

## The two design constraints

**1. Robustness comes from coverage, not tuned constants.** A single pipeline
bakes in assumptions that some unseen input will break. Instead we run a
*cascade* of self-calibrating strategies and return on the first **validated**
decode. Generalisation = how well the ladder's rungs span the variation; there
are almost no free constants to overfit, because every rung keys its parameters
to the *measured* module pitch, and the exit is gated on a valid decode.

**2. Latency is bounded by construction.** The reason stock libdmtx can take
seconds is its region search over a full frame, worst on misses. We:
  - **localize once** (cheap, decoder-free, on a downscaled gradient map) and
    hand the cascade a small rectified ROI, so libdmtx's own search is trivial;
  - give every libdmtx call a **per-rung timeout** (tens of ms);
  - enforce a **global deadline** in the orchestrator.
Clean codes exit in rung 0 in a few ms; the hard tail walks the ladder up to
the budget and then stops. Verified: a 1200×1200 noise frame with a 40 ms
timeout returns in 41 ms.

## Pipeline

```
image ──▶ best_contrast_channel ──▶ localize (once) ──▶ for each candidate:
                                                          run_cascade under
                                                          a global deadline
                                                              │
   adapt.py            localize.py                        cascade.py
   (colour → most-     (gradient blob →                   (raw→otsu→adaptive→
    separable channel)  rectified ROI +                    scale→close→open→
                        quiet-zone pad)                     unsharp; per-rung
                                                            timeout; valid-
                                                            decode early exit)
                                                              │
                                                          binding.py
                                                          (staged libdmtx:
                                                           find ≠ decode,
                                                           + region geometry
                                                           + px/module)
```

## Why measure with synthetic data

A real corpus from one lab is too specific to optimise against — tuning to it
just overfits to one printer. So the **primary optimisation surface is
synthetic** (`synth.py`): render known payloads with libdmtx's own encoder,
then degrade them parametrically across the axes you want to generalise over
(module size, blur, ink gain/dropout, label colour, rotation, noise). That
gives unlimited perfectly-labelled samples from sources you don't own. Your
real corpus (`corpus/images` + `labels.csv`) is run the same way, as a
*confirmation* that the degradation model is realistic — never the tuning
target.

Judge the reader by its **worst stratum**, not its mean (`bench/harness.py`
reports per-axis read rate and the single worst cell). A source-agnostic reader
is only as good as its weakest substrate. Adding a cascade rung is justified
only when it lifts a weak stratum without regressing others
(`bench/report.py` diffs two runs and flags regressions).

## Layout

```
src/datamatrix_reader/
  _build_dmtx.py   cffi build + C shim (staged decode, timeout, encode)
  binding.py       typed wrapper → StageResult (found/decoded/bbox/px-module)
  adapt.py         contrast-channel selection, scale normalisation
  localize.py      decoder-free region finding + rectification
  validate.py      payload validators (gate the early exit)
  cascade.py       self-calibrating rung ladder + bounded orchestration
  reader.py        public API: Reader.read(image, budget_ms)
bench/
  harness.py       stratified read/correct rate, latency p50/p95, found-vs-decoded
  report.py        diff two runs, surface regressed strata
third_party/libdmtx/   (git submodule — keeps patching on the table)
corpus/images/, corpus/labels.csv   your real captures + ground truth
experiments/     config files, one per cascade/param variant
```

## Build & run

```bash
# native deps: apt install libdmtx-dev   (or: brew install libdmtx)
pip install -e .
python -m datamatrix_reader._build_dmtx           # compile the shim against libdmtx

# baseline on synthetic strata
python -m bench.harness --synth --per-cell 2 --budget 250 --out runs/baseline.json
# after a change
python -m bench.harness --synth --per-cell 2 --budget 250 --out runs/variant.json
python -m bench.report runs/baseline.json runs/variant.json

# confirmation on your real slides
python -m bench.harness --corpus corpus --budget 250 --out runs/real.json
```

```python
from datamatrix_reader.reader import Reader
from datamatrix_reader.validate import RegexValidator

reader = Reader(validator=RegexValidator(r'^S\d{2}-\d{5}-A\d$'))
res = reader.read(image_bgr, budget_ms=250)   # res.payload, res.rung, res.elapsed_ms
```

## Handoff to Claude Code — good next tasks

The scaffold is the measure→tune frame; the work is filling it in against the
worst strata:

1. **Capture-geometry / scale rungs.** The baseline shows failure concentrated
   at low px/module. Improve `t_scale_adaptive` / add a super-resolution rung,
   and separately confirm how many px/module your rig actually delivers — some
   of this is optical, not software.
2. **Localization recall.** `localize.py` is a first-pass blob finder; measure
   its hit rate (found-rate in the harness) and harden it (multi-scale, better
   squareness/timing-pattern scoring) without slowing rung 0.
3. **Degradation realism.** Calibrate `synth.AXES` against a handful of real
   captures so the synthetic distribution covers your true failure modes; keep
   the real corpus as held-out confirmation only.
4. **Latency tightening.** p95 sits near the budget; profile per rung, prune or
   reorder rungs, and consider a cheap routing feature only if needed.

Each change is accepted/rejected by `bench/report.py` on the worst-stratum and
Pareto criteria — not by the mean read rate on any one corpus.

## Verified in this scaffold
- libdmtx 0.7.7 staged binding (find vs decode split, region geometry, hard
  timeout) — encode→decode round-trip exact.
- Per-call timeout honoured (40 ms request → 41 ms return on noise).
- Full reader decodes a yellow-stock, blurred, ink-gained, rotated, text-
  crowded synthetic capture in ~15 ms.
- Stratified harness over 216 synthetic samples: substrate colour flat across
  white/yellow/pink (channel selection working), failure isolated to low
  px/module.
