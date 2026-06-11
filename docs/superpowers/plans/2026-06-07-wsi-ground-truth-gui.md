# WSI Ground-Truth Collector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tools/label_gt.py` — auto-fill `labels.csv` from decoder consensus, then a Tkinter queue to hand-label only the no-read/disagreement images and delete codeless ones — and add a flat-dir fallback to `compare_backends --corpus` so the real Grundium labels can be scored for correctness.

**Architecture:** One module of small pure functions (consensus classification, CSV load/save, queue building, delete-move) plus a thin Tkinter GUI that calls them. The pure functions are unit-tested headlessly; the GUI is assembled and smoke-tested manually. `compare_backends`'s corpus loader is extracted into a testable `load_corpus()` that gains the flat-dir fallback.

**Tech Stack:** Python 3.12, stdlib `csv`/`shutil`/`tkinter` (no Pillow — default 0.5 scale = `PhotoImage.subsample(2)`), reuses `tools/compare_backends.FOLDS` (zxing-cpp + libdmtx) and `cv2` for the decode pass.

Spec: `docs/superpowers/specs/2026-06-07-wsi-ground-truth-gui-design.md`

---

## File Structure

- **Create** `tools/label_gt.py` — pure logic (`decide`, `load_labels`, `save_labels`, `payload_to_text`, `pending_images`, `delete_image`, `autofill`) + GUI (`run_gui`) + `main()`.
- **Create** `tests/test_label_gt.py` — headless tests for the pure logic.
- **Modify** `tools/compare_backends.py` — extract `load_corpus(root)`, add flat-dir fallback + extension filter; `--corpus` path calls it.
- **Modify** `tests/` — add `test_compare_corpus_loader.py` (or fold into existing) for the fallback.
- **Modify** `.gitignore` — ignore `corpus/wsi_labels_removed/` (dual-path).

---

### Task 0: Environment — install Tk

**Files:** none (system setup)

- [ ] **Step 1: Install Tk for homebrew Python 3.12**

Run: `brew install python-tk@3.12`

- [ ] **Step 2: Verify the venv can import tkinter**

Run: `.venv/bin/python -c "import tkinter; print('tk', tkinter.TkVersion)"`
Expected: prints `tk 8.6` (no traceback). If it still fails, recreate the venv (`python3.12 -m venv .venv` per HANDOFF) so it picks up the new `_tkinter`.

---

### Task 1: Consensus classification (`decide`)

**Files:**
- Create: `tools/label_gt.py`
- Test: `tests/test_label_gt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_label_gt.py
from tools.label_gt import decide

def test_all_agree_is_auto():
    assert decide({"a": b"X", "b": b"X", "c": b"X"}) == ("auto", [b"X"])

def test_sole_reader_is_auto():
    assert decide({"a": b"X", "b": None, "c": None}) == ("auto", [b"X"])

def test_no_read_is_queue_empty():
    assert decide({"a": None, "b": None, "c": None}) == ("queue", [])

def test_disagreement_is_queue_with_sorted_candidates():
    assert decide({"a": b"Y", "b": b"X", "c": None}) == ("queue", [b"X", b"Y"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: cannot import name 'decide'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/label_gt.py
from __future__ import annotations
import csv, shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def decide(reads: dict[str, bytes | None]) -> tuple[str, list[bytes]]:
    """Classify per-decoder reads for one image.

    ("auto", [payload]) when the distinct non-None reads number exactly 1
    (all decoders that fired agree, or a single decoder fired). Otherwise
    ("queue", candidates) where candidates is the sorted distinct reads
    (empty when nothing read, >=2 on disagreement)."""
    vals = sorted({v for v in reads.values() if v is not None})
    if len(vals) == 1:
        return ("auto", vals)
    return ("queue", vals)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/label_gt.py tests/test_label_gt.py
git commit -m "feat(label_gt): consensus classification for ground-truth auto-fill"
```

---

### Task 2: CSV load/save + payload decode

**Files:**
- Modify: `tools/label_gt.py`
- Test: `tests/test_label_gt.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_label_gt.py
from tools.label_gt import load_labels, save_labels, payload_to_text

def test_payload_to_text_ascii_and_fallback():
    assert payload_to_text(b"1-S-24-34325 G2-1") == "1-S-24-34325 G2-1"
    assert payload_to_text(b"\xff") == "\xff"  # latin-1 fallback, no crash

def test_labels_roundtrip_and_sorted(tmp_path):
    p = tmp_path / "labels.csv"
    save_labels(p, {"b.png": "2", "a.png": "1"})
    assert p.read_text().splitlines() == ["file,payload", "a.png,1", "b.png,2"]
    assert load_labels(p) == {"a.png": "1", "b.png": "2"}

def test_load_missing_file_is_empty(tmp_path):
    assert load_labels(tmp_path / "nope.csv") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -q`
Expected: FAIL — `cannot import name 'load_labels'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/label_gt.py
def payload_to_text(b: bytes) -> str:
    try:
        return b.decode("ascii")
    except UnicodeDecodeError:
        return b.decode("latin-1")


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if path.exists():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                labels[row["file"]] = row["payload"]
    return labels


def save_labels(path: Path, labels: dict[str, str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "payload"])
        for name in sorted(labels):
            w.writerow([name, labels[name]])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/label_gt.py tests/test_label_gt.py
git commit -m "feat(label_gt): labels.csv load/save + payload decode"
```

---

### Task 3: Queue building + delete-move

**Files:**
- Modify: `tools/label_gt.py`
- Test: `tests/test_label_gt.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_label_gt.py
from tools.label_gt import pending_images, delete_image

def _touch(d, name):
    p = d / name; p.write_bytes(b"x"); return p

def test_pending_excludes_labeled_and_nonimages(tmp_path):
    _touch(tmp_path, "a.png"); _touch(tmp_path, "b.png")
    _touch(tmp_path, "labels.csv")  # non-image, must be ignored
    pend = pending_images(tmp_path, {"a.png": "1"})
    assert [p.name for p in pend] == ["b.png"]

def test_delete_moves_file_and_drops_label(tmp_path):
    img = _touch(tmp_path, "junk.png")
    removed = tmp_path / "removed"
    labels = {"junk.png": "stale"}
    csv_path = tmp_path / "labels.csv"
    delete_image(img, removed, labels, csv_path)
    assert not img.exists()
    assert (removed / "junk.png").exists()
    assert "junk.png" not in labels
    assert load_labels(csv_path) == {}
    # re-running pending must not re-queue it (file is gone)
    assert pending_images(tmp_path, labels) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -q`
Expected: FAIL — `cannot import name 'pending_images'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/label_gt.py
def pending_images(image_dir: Path, labels: dict[str, str]) -> list[Path]:
    return [p for p in sorted(image_dir.iterdir())
            if p.suffix.lower() in IMG_EXTS and p.name not in labels]


def delete_image(path: Path, removed_dir: Path,
                 labels: dict[str, str], labels_csv: Path) -> None:
    removed_dir.mkdir(exist_ok=True)
    shutil.move(str(path), str(removed_dir / path.name))
    if path.name in labels:
        del labels[path.name]
        save_labels(labels_csv, labels)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -q`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/label_gt.py tests/test_label_gt.py
git commit -m "feat(label_gt): pending-queue build + delete-move"
```

---

### Task 4: Refactor `compare_backends` corpus loader + flat-dir fallback

**Files:**
- Modify: `tools/compare_backends.py:116-126` (the `if args.corpus:` block)
- Test: `tests/test_label_gt.py` (new test fn) or `tests/test_compare_corpus.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare_corpus.py
import csv, cv2, numpy as np
from tools.compare_backends import load_corpus

def _write_png(p):
    cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))

def test_load_corpus_flat_dir_with_labels(tmp_path):
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "b.png")
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "payload"])
        w.writerow(["a.png", "111"]); w.writerow(["b.png", "222"])
    items = load_corpus(tmp_path)            # no images/ subdir -> flat fallback
    truths = sorted(t for t, _ in items)
    assert truths == [b"111", b"222"]
    assert all(img is not None for _, img in items)

def test_load_corpus_skips_unlabeled_and_nonimages(tmp_path):
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "c.png")           # no label row -> skipped
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "payload"]); w.writerow(["a.png", "111"])
    items = load_corpus(tmp_path)
    assert [t for t, _ in items] == [b"111"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_compare_corpus.py -q`
Expected: FAIL — `cannot import name 'load_corpus'`.

- [ ] **Step 3: Implement `load_corpus` and call it from main**

Add near the top-level functions in `tools/compare_backends.py`:

```python
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_corpus(root):
    """Load (truth_bytes, bgr_image) pairs from a labelled corpus.

    Accepts both layouts: <root>/images/* + <root>/labels.csv (BarBeR style),
    or a flat <root>/*.png + <root>/labels.csv (wsi_labels style). Only images
    with a labels.csv row are returned."""
    from pathlib import Path
    root = Path(root)
    labels = {}
    lp = root / "labels.csv"
    if lp.exists():
        with lp.open(newline="") as f:
            for row in csv.DictReader(f):
                labels[row["file"]] = row["payload"].encode()
    img_dir = root / "images"
    if not img_dir.is_dir():
        img_dir = root
    items = []
    for p in sorted(img_dir.glob("*")):
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None and p.name in labels:
            items.append((labels[p.name], img))
    return items
```

Then replace the body of the `if args.corpus:` branch in `main()` so it reads:

```python
    if args.corpus:
        items = load_corpus(args.corpus)
        print(f"=== corpus: {args.corpus} (n={len(items)}) ===")
    else:
        items = [(truth, img) for _, truth, img in _iter_synth(args.per_cell)]
        print(f"=== synth per_cell={args.per_cell} (n={len(items)}) ===")
    run_labelled(items, args.budget)
```

(Remove the now-dead inline `labels = {}` / glob loop that the branch used to contain.)

- [ ] **Step 4: Run tests — new + regression**

Run: `.venv/bin/python -m pytest tests/test_compare_corpus.py -q && .venv/bin/python -m pytest tests/ -q`
Expected: new file passes; full suite still green (previously 6 passed, now more).

- [ ] **Step 5: Commit**

```bash
git add tools/compare_backends.py tests/test_compare_corpus.py
git commit -m "feat(compare_backends): extract load_corpus + flat-dir fallback"
```

---

### Task 5: Phase A wiring (`autofill`)

**Files:**
- Modify: `tools/label_gt.py`
- Test: `tests/test_label_gt.py`

- [ ] **Step 1: Write the failing test** (uses a fake folds list — no real decoders)

```python
# append to tests/test_label_gt.py
import cv2, numpy as np
from tools.label_gt import autofill

def _png(d, name):
    cv2.imwrite(str(d / name), np.zeros((8, 8, 3), np.uint8)); return d / name

def test_autofill_writes_consensus_and_queues_rest(tmp_path):
    for n in ("agree.png", "disagree.png", "noread.png"):
        _png(tmp_path, n)
    # fake decoders keyed on filename so the test is deterministic
    def f1(img, b): return b"P"      # fires on all
    def f2(img, b): return None
    folds = [("f1", f1), ("f2", f2)]
    labels = {}
    res = autofill(tmp_path, labels, budget=50, folds=folds)
    # f1 is a sole reader on every image -> all become consensus auto-fills
    assert labels == {"agree.png": "P", "disagree.png": "P", "noread.png": "P"}
    assert res["added"] == 3 and res["queue"] == []
    assert load_labels(tmp_path / "labels.csv")["agree.png"] == "P"

def test_autofill_queues_disagreement(tmp_path):
    _png(tmp_path, "x.png")
    folds = [("f1", lambda i, b: b"A"), ("f2", lambda i, b: b"B")]
    labels = {}
    res = autofill(tmp_path, labels, budget=50, folds=folds)
    assert labels == {}                       # not auto-filled
    assert [(p.name, c) for p, c in res["queue"]] == [("x.png", ["A", "B"])]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -k autofill -q`
Expected: FAIL — `cannot import name 'autofill'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to tools/label_gt.py
import cv2  # add to the imports at top of file


def autofill(image_dir: Path, labels: dict[str, str], budget: int,
             folds=None) -> dict:
    """Run decoders over every still-pending image; auto-fill consensus reads
    into labels (+ labels.csv), return {'added': int, 'queue': [(Path, [str])]}
    for the no-read/disagreement images."""
    if folds is None:
        from tools.compare_backends import FOLDS as folds
    csv_path = image_dir / "labels.csv"
    queue: list[tuple[Path, list[str]]] = []
    added = 0
    for p in pending_images(image_dir, labels):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        reads = {name: fn(img, budget) for name, fn in folds}
        status, vals = decide(reads)
        if status == "auto":
            labels[p.name] = payload_to_text(vals[0])
            added += 1
        else:
            queue.append((p, [payload_to_text(v) for v in vals]))
    save_labels(csv_path, labels)
    return {"added": added, "queue": queue}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_label_gt.py -q`
Expected: all label_gt tests pass (11).

- [ ] **Step 5: Commit**

```bash
git add tools/label_gt.py tests/test_label_gt.py
git commit -m "feat(label_gt): Phase A auto-fill over corpus"
```

---

### Task 6: Tkinter GUI + `main()` (manual smoke)

**Files:**
- Modify: `tools/label_gt.py`

- [ ] **Step 1: Add the GUI and CLI** (no unit test — Tkinter event loop)

```python
# add to tools/label_gt.py
import argparse
import tkinter as tk
from tkinter import messagebox


def run_gui(queue, image_dir: Path, removed_dir: Path,
            labels: dict[str, str], csv_path: Path, scale: float) -> None:
    """Show each queued image at `scale` and collect a payload or a delete."""
    factor = max(1, round(1 / scale))      # 0.5 -> subsample(2); stdlib only
    root = tk.Tk()
    root.title("wsi_labels GT")
    state = {"i": 0}
    img_label = tk.Label(root)
    img_label.pack()
    hint = tk.Label(root, fg="#666"); hint.pack()
    entry = tk.Entry(root, width=40); entry.pack()
    counter = tk.Label(root); counter.pack()
    photo = {"ref": None}                  # keep a ref so Tk doesn't GC it

    def show():
        i = state["i"]
        if i >= len(queue):
            root.destroy(); return
        path, candidates = queue[i]
        photo["ref"] = tk.PhotoImage(file=str(path)).subsample(factor)
        img_label.config(image=photo["ref"])
        hint.config(text=("candidates: " + "   ".join(candidates)) if candidates else "")
        counter.config(text=f"{i + 1} / {len(queue)}   ({path.name})")
        entry.delete(0, tk.END)
        entry.focus_set()

    def save(_=None):
        val = entry.get().strip()
        if not val:
            messagebox.showwarning("Empty", "Enter a payload, or use Delete.")
            return
        path, _c = queue[state["i"]]
        labels[path.name] = val
        save_labels(csv_path, labels)
        state["i"] += 1; show()

    def delete():
        path, _c = queue[state["i"]]
        delete_image(path, removed_dir, labels, csv_path)
        state["i"] += 1; show()

    def nav(d):
        state["i"] = max(0, min(len(queue) - 1, state["i"] + d)); show()

    tk.Button(root, text="Save ⏎", command=save).pack(side="left")
    tk.Button(root, text="Delete - no barcode", command=delete).pack(side="left")
    tk.Button(root, text="Prev", command=lambda: nav(-1)).pack(side="left")
    tk.Button(root, text="Next", command=lambda: nav(1)).pack(side="left")
    root.bind("<Return>", save)
    show()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()

    image_dir = Path(args.corpus)
    removed_dir = image_dir.parent / (image_dir.name + "_removed")
    csv_path = image_dir / "labels.csv"
    labels = load_labels(csv_path)
    before = len(labels)
    res = autofill(image_dir, labels, args.budget)
    print(f"auto-filled {res['added']}  (already had {before})  "
          f"queue {len(res['queue'])}")
    if res["queue"]:
        run_gui(res["queue"], image_dir, removed_dir, labels, csv_path, args.scale)
    print(f"labels.csv now has {len(load_labels(csv_path))} rows -> {csv_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on a 3-image throwaway copy** (don't touch the real corpus yet)

```bash
mkdir -p /tmp/gt_smoke && cp "$(ls corpus/wsi_labels/*.png | head -3)" /tmp/gt_smoke/
.venv/bin/python -m tools.label_gt --corpus /tmp/gt_smoke --budget 250
```
Expected: prints an auto-fill summary; if any of the 3 are no-read/disagree, a window opens at 0.5×. Verify: typing a payload + Enter advances and writes to `/tmp/gt_smoke/labels.csv`; **Delete** moves an image into `/tmp/gt_smoke_removed/` and drops its row. Close the window; the summary line prints the final row count.

- [ ] **Step 3: Run the full unit suite (regression)**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all green (label_gt + compare_corpus + original 6).

- [ ] **Step 4: Commit**

```bash
git add tools/label_gt.py
git commit -m "feat(label_gt): Tkinter labeling GUI + CLI"
```

---

### Task 7: Gitignore the removed dir + real run

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add the ignore rule** (dual-path, like the others)

Append under the existing wsi_labels block in `.gitignore`:

```
# No-barcode labels moved aside by tools/label_gt.py — kept local.
datamatrix_reader/corpus/wsi_labels_removed/
corpus/wsi_labels_removed/
```

- [ ] **Step 2: Verify it's ignored**

Run: `git check-ignore datamatrix_reader/corpus/wsi_labels_removed/ ; git status --short`
Expected: `check-ignore` echoes the path; `git status` shows only the modified `.gitignore`.

- [ ] **Step 3: Commit the ignore rule**

```bash
git add .gitignore
git commit -m "chore: gitignore wsi_labels_removed (kept local)"
```

- [ ] **Step 4: Run for real, then score**

```bash
.venv/bin/python -m tools.label_gt --corpus corpus/wsi_labels --budget 250
# (label the queued images; delete the ~3 codeless ones)
.venv/bin/python -m tools.compare_backends --corpus corpus/wsi_labels --budget 250
```
Expected: `compare_backends` now prints correctness **rates** (not just hits) for all three folds — the ground-truthed tiebreaker. Sanity check: zxing's correct count should be in the neighborhood of last run's 350 hits.

---

## Notes for the implementer

- Run everything with `.venv/bin/python` from `datamatrix_reader/` (project lives in that subdir; the repo root is one level up).
- `tools/` has no `__init__.py` but is imported as `tools.compare_backends` / `tools.label_gt` via the `-m` runner and pytest's rootdir — keep that pattern; don't add packaging.
- The GUI deliberately keeps a reference to the current `PhotoImage` (`photo["ref"]`); dropping it causes Tk to garbage-collect the image and show blank.
- Decoder returns are `bytes`; everything user-facing/CSV is text via `payload_to_text`. Keep that boundary — don't mix.

## Deliberate simplifications vs. the spec (YAGNI)

- **Disagreement candidates are a read-only text hint, not click-to-fill buttons.** Disagreements are rare and the payload is a short accession string; typing it is trivial. Drop the click-to-fill interaction unless it proves annoying in real use.
- **No Pillow / arbitrary-scale.** Only reciprocal-integer scales are exact (`--scale 0.5`→`subsample(2)`, `0.25`→`subsample(4)`). A non-reciprocal value rounds to the nearest integer factor (`0.75`→factor 1 = full size) — acceptable since 0.5 is the default and the source is high-res. Add Pillow only if a non-integer scale is genuinely needed.
