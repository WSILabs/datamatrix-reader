# WSI Ground-Truth Collector — design

**Date:** 2026-06-07
**Status:** approved (brainstorming) → ready for implementation plan
**Component:** `tools/label_gt.py` (new) + small change to `tools/compare_backends.py`

## Problem

The real Grundium WSI labels live in `corpus/wsi_labels/` (407 PNGs, 1200×848,
gitignored PHI — filenames contain accession numbers). They have **no ground
truth**, so the decoder comparison can only count decode-*hits*, not *correctness*.
We want a `labels.csv` (`file,payload`) so `compare_backends --corpus` can produce
real correctness scores — the proper, ground-truthed version of the zxing-vs-libdmtx
tiebreaker.

Hand-typing 407 payloads is wasteful when the decoders already agree on most of
them. The tool should **auto-fill what the decoders are confident about and prompt
the user only for the gaps**.

## Goals

1. Auto-populate `labels.csv` from decoder **consensus** without asking the user.
2. Present a local GUI for **only** the un-decided images (no read, or disagreement).
3. Let the user delete the handful of images that contain no barcode at all.
4. Be resumable and idempotent — re-running picks up where it left off.
5. Produce a `labels.csv` that `compare_backends --corpus` reads directly.

## Non-goals (v1)

- Reviewing or editing the auto-filled consensus entries (user: "don't even ask me").
- Accession-number format validation of payloads.
- Multiple codes per label / multi-symbology aggregation.
- Hard-deleting files (we move aside instead — see Delete).

## Data model

`corpus/wsi_labels/` stays **flat** (no `images/` subdir — we don't reshuffle 407
PHI files). Artifacts written alongside the PNGs:

- **`corpus/wsi_labels/labels.csv`** — canonical ground truth, `file,payload`
  (same schema `compare_backends` and `bench/harness` already read). One row per
  image that has a known payload, whether auto-filled or hand-entered. Payload is
  stored as text. Decoder returns are `bytes`; accession numbers are ASCII, so the
  decode-to-text on write and `compare_backends`' `payload.encode()` (utf-8) on read
  round-trip to the identical bytes the decoders produced. (A non-ASCII payload is
  decoded `latin-1` for lossless storage and logged to spot-check, since the utf-8
  re-encode would then not match — not expected for accession labels.)
- **`corpus/wsi_labels_removed/`** — sibling dir (also gitignored) where the
  Delete button **moves** no-barcode images. Recoverable; excluded from the corpus.

An image's state is derived, not stored separately:
- in `labels.csv` → done (auto or manual).
- in `wsi_labels_removed/` → deleted.
- neither → still needs a decision (enters the GUI queue).

## Phase A — auto-fill (silent)

Reuse `compare_backends.FOLDS` (the three decoders). For each PNG in the flat
corpus root, run all three with the given budget. Collect the non-`None` returns.

- **Consensus** = the set of distinct non-`None` payloads has size exactly 1.
  This covers both "all decoders that fired agree" and "exactly one decoder fired"
  (sole reader — accepted by the user as a known, spot-checkable risk).
  → write `file,payload` to `labels.csv`. **Never shown to the user.**
- **No read** (all three returned `None`) or **disagreement** (≥2 distinct
  payloads) → defer to the GUI queue.

Rows already present in `labels.csv` from a prior run are kept; deleted images are
skipped. Auto-fill only adds rows for images that are currently undecided.

## Phase B — GUI (Tkinter)

Stdlib `tkinter` only (zero new deps, fully offline — required for PHI). One window,
one image at a time, iterating the queue (no-read + disagreement images).

```
┌─ wsi_labels GT   (3 / 42)  ─────────────┐
│   [ label image, scaled by --scale ]    │
│   decoders: zxing=ABC   libdmtx=ABD     │   ← hint row; only when they disagree
│   payload: [__________________]  [Save⏎]│
│   [Delete – no barcode]   [Prev] [Next] │
└─────────────────────────────────────────┘
```

- **Display scale:** `--scale` flag, default **0.5** (600×424 from the 1200×848
  source — readable, doesn't fill the screen). Image scaled with PIL/Pillow if
  available, else `tk.PhotoImage.subsample` for integer factors; the printed
  accession text stays legible at 0.5×.
- **Payload field + Save (Enter):** writes/updates the row in `labels.csv`,
  advances to next. Empty payload is rejected (use Delete for no-barcode).
- **Delete – no barcode:** moves the file to `corpus/wsi_labels_removed/`, removes
  any stale `labels.csv` row, advances. Recoverable by moving the file back.
- **Prev/Next:** navigate without saving, so the user can revisit/correct.
- **Disagreement hint:** when decoders disagreed, show each candidate; clicking one
  fills the payload box (still requires Save to commit).
- **Progress counter** in the title: decided-in-this-session / queue-size.

Saves are per-image (write `labels.csv` after each Save/Delete), so a crash or
window close loses nothing.

## Phase C — corpus integration

`compare_backends.py --corpus` currently globs `root / "images"`. Add a **flat-dir
fallback**: if `root / "images"` does not exist, glob image files from `root`
itself (same extension filter `bench/harness` uses: png/jpg/jpeg/tif/tiff/bmp).
`labels.csv` lookup is unchanged. No new flags. After ground truth exists:

```
python -m tools.compare_backends --corpus corpus/wsi_labels --budget 250
```

now reports correctness rates (not just hits) for all three folds on the real
labels. (`bench/harness._iter_corpus` has the same `images/`-only assumption; out
of scope here, noted for a future consistency pass.)

## CLI

```
python -m tools.label_gt --corpus corpus/wsi_labels [--budget 250] [--scale 0.5]
```

- `--corpus` (required): path to the flat image dir.
- `--budget` (default 250): per-decoder timeout ms for the auto-fill pass.
- `--scale` (default 0.5): display scale factor for the GUI.

Run sequence: Phase A runs headless and prints a summary
(`auto-filled N, queue M, already-done K`), then Phase B opens the window only if
the queue is non-empty.

## Error handling

- Missing/unreadable PNG in Phase A → skipped with a logged warning (matches
  existing `cv2.imread is None` handling).
- `labels.csv` with a row whose file no longer exists (e.g. later deleted) → row
  ignored on load; not re-written.
- Window closed mid-queue → already-saved rows persist; re-run resumes the
  remaining queue.
- Pillow not installed → fall back to integer `subsample`; if `--scale` is not a
  clean reciprocal (e.g. 0.5→2, 0.25→4), round to the nearest supported factor and
  warn. (Recommend documenting Pillow as an optional nicety, not required.)

## Testing

- **Consensus logic** (pure, no GUI): given dicts of per-fold returns, assert
  which become auto-fill rows vs queue items — all-agree, sole-reader, two-way
  disagree, all-None. This is the correctness-critical part.
- **labels.csv round-trip:** write rows, reload, assert idempotent re-run produces
  an empty queue for already-decided images.
- **Delete semantics:** moving a file to `wsi_labels_removed/` drops it from both
  the queue and `labels.csv`; a re-run does not re-queue it.
- **Flat-dir fallback in compare_backends:** a temp flat corpus (PNGs + labels.csv,
  no `images/`) loads the expected (truth, img) pairs.
- GUI event wiring is not unit-tested (Tkinter); the logic it calls is tested
  headlessly via the functions above.

## Open risks

- **Sole-reader auto-fill** can bake in a single decoder's misread. Accepted;
  mitigated by being able to diff auto-filled vs hand-entered rows later if a
  spot-check is wanted.
- **PHI:** `labels.csv` contains accession numbers and `wsi_labels_removed/` holds
  label images. Both must be gitignored. `corpus/wsi_labels/` already is; add
  `corpus/wsi_labels_removed/` (and the dual-path variant) to `.gitignore`.
