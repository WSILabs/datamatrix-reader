from __future__ import annotations
import argparse
import csv, cv2, re, shutil
import tkinter as tk
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Grundium WSI filenames embed the accession: "<grp>__scan_<n>_<ACCESSION>_label<n>.png".
# We use it only as a pre-fill headstart for manual entry (verify against the image).
_ACCESSION_RE = re.compile(r"scan_\d+_(.*?)_label\d*", re.IGNORECASE)


def parse_accession(filename: str) -> str:
    """Best-effort accession string parsed from a wsi_labels filename, as a
    pre-fill hint for manual entry. Returns '' when the pattern doesn't match
    (e.g. no-code calibration labels). This is a hint to verify, not truth."""
    m = _ACCESSION_RE.search(filename)
    return m.group(1).strip() if m else ""


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


def payload_to_text(b: bytes) -> str:
    # Accession payloads are ASCII, so this round-trips losslessly through
    # labels.csv and compare_backends.load_corpus's utf-8 .encode(). The latin-1
    # branch is only a crash-guard for an unexpected non-ASCII read; such a value
    # would NOT round-trip back to the same bytes (utf-8 re-encode differs), so a
    # non-ASCII row should be treated as suspect rather than trusted ground truth.
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


def pending_images(image_dir: Path, labels: dict[str, str]) -> list[Path]:
    return [p for p in sorted(image_dir.iterdir())
            if p.suffix.lower() in IMG_EXTS and p.name not in labels]


def delete_image(path: Path, removed_dir: Path,
                 labels: dict[str, str], labels_csv: Path) -> None:
    removed_dir.mkdir(exist_ok=True)
    shutil.move(path, removed_dir / path.name)
    if path.name in labels:
        del labels[path.name]
        save_labels(labels_csv, labels)


def autofill(image_dir: Path, labels: dict[str, str], budget: int,
             folds=None, progress: bool = False) -> dict:
    """Run decoders over every still-pending image; auto-fill consensus reads
    into labels (+ labels.csv), return {'added': int, 'queue': [(Path, [str])]}
    for the no-read/disagreement images. The decode pass is the slow part
    (~hundreds of ms/image); pass progress=True to print a live counter
    (counts only, no payloads/filenames -> PHI-safe)."""
    if folds is None:
        from tools.compare_backends import FOLDS as folds
    csv_path = image_dir / "labels.csv"
    pending = pending_images(image_dir, labels)
    total = len(pending)
    queue: list[tuple[Path, list[str]]] = []
    added = 0
    for n, p in enumerate(pending, 1):
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
        if progress:
            print(f"\rPhase A: decoded {n}/{total}  "
                  f"(auto {added}, queue {len(queue)})", end="", flush=True)
    if progress and total:
        print()  # terminate the \r line
    save_labels(csv_path, labels)
    return {"added": added, "queue": queue}


def run_gui(queue, image_dir: Path, removed_dir: Path,
            labels: dict[str, str], csv_path: Path, scale: float) -> None:
    """Show each queued image at `scale` and collect a payload or a delete."""
    factor = max(1, round(1 / scale))      # 0.5 -> subsample(2); stdlib only
    root = tk.Tk()
    root.title("wsi_labels GT")
    state = {"i": 0}
    drafts: dict[int, str] = {}            # unsaved text per index, kept across nav
    last = {"payload": ""}                 # most recent saved payload, for "same as last"
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
        # pre-fill: a value already saved wins, else the unsaved draft, else the
        # accession parsed from the filename as a headstart. select-all so Enter
        # accepts it and typing replaces it.
        prefill = labels.get(path.name) or drafts.get(i) or parse_accession(path.name)
        entry.delete(0, tk.END)
        entry.insert(0, prefill)
        entry.select_range(0, tk.END)
        entry.focus_set()

    def advance(d):
        # step in direction d, skipping queue entries whose file was deleted.
        # forward past the end closes the window; backward past the front stays put.
        i = state["i"] + d
        while 0 <= i < len(queue) and not queue[i][0].exists():
            i += d
        state["i"] = i if i >= 0 else state["i"]
        show()

    def commit_current():
        # persist the current entry if it has content; a blank box means
        # "skip for now" (the image stays in the queue next run). Never loses
        # typed text: non-blank -> labels.csv, blank-but-typed -> in-memory draft.
        val = entry.get().strip()
        if val:
            path, _c = queue[state["i"]]
            labels[path.name] = val
            save_labels(csv_path, labels)
            last["payload"] = val
            drafts.pop(state["i"], None)
        else:
            drafts[state["i"]] = entry.get()

    def nav(d):
        commit_current()                   # Prev/Next/Enter all save first
        advance(d)

    def same_as_last(_=None):
        entry.delete(0, tk.END)
        entry.insert(0, last["payload"])
        entry.select_range(0, tk.END)
        entry.focus_set()

    def delete():
        path, _c = queue[state["i"]]
        delete_image(path, removed_dir, labels, csv_path)
        drafts.pop(state["i"], None)
        advance(1)

    tk.Button(root, text="◀ Prev (saves)", command=lambda: nav(-1)).pack(side="left")
    tk.Button(root, text="Next ⏎ (saves)", command=lambda: nav(1)).pack(side="left")
    tk.Button(root, text="Same as last", command=same_as_last).pack(side="left")
    tk.Button(root, text="Delete - no barcode", command=delete).pack(side="left")
    root.bind("<Return>", lambda e: nav(1))
    root.bind("<Control-l>", same_as_last)  # keyboard shortcut for "same as last"
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
    res = autofill(image_dir, labels, args.budget, progress=True)
    print(f"auto-filled {res['added']}  (already had {before})  "
          f"queue {len(res['queue'])}")
    if res["queue"]:
        run_gui(res["queue"], image_dir, removed_dir, labels, csv_path, args.scale)
    print(f"labels.csv now has {len(load_labels(csv_path))} rows -> {csv_path}")


if __name__ == "__main__":
    main()
