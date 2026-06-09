"""Defect classifier for the codes the zxing Reader misses.

We study WHY ~13% of real WSI DataMatrix codes fail (handheld/commercial readers
get them all, so this is a reader-robustness gap, not hard data). For each
raw-zxing miss we record: a human-drawn bounding box (the only reliable localizer
— both zxing AND libdmtx fail to locate these) and a multiselect set of defects.
The output drives a defect -> which-fix-recovers-it map.

Pipeline:
    python -m tools.defect_classify --corpus corpus/wsi_labels
      -> builds a draft (auto bbox + auto-metric defect guesses) for every
         raw-zxing miss, then opens a GUI to review/correct (image + draggable
         box + pre-ticked checkboxes), writing defects.csv in the corpus dir.

Defect codes are grouped capture/print so we can tell imaging problems (fixable
by preprocessing) from print/damage problems (module-level, may be unrecoverable).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from dmtxslide.reader import Reader, _gray
from dmtxslide import preprocess as pp

_DM = zxingcpp.BarcodeFormat.DataMatrix

# Defect taxonomy — (code, human label). "size" deliberately excluded: the
# capture is uniform across the dataset, so code size is a constant and cannot
# explain differential failure.
DEFECTS: list[tuple[str, str]] = [
    # capture / imaging (vary image-to-image; typically preprocessing-fixable)
    ("glare", "glare / specular highlight"),
    ("uneven_illum", "uneven illumination gradient"),
    ("blur", "out-of-focus blur"),
    ("perspective", "perspective / skew"),
    ("rotated", "rotated (severe only — zxing derotates mild)"),
    ("low_contrast", "low substrate contrast"),
    ("occlusion", "occlusion (redaction/edge on code)"),
    ("out_of_frame", "code partially out of frame"),
    # print / physical (module-level; may be unrecoverable)
    ("faint", "faint / under-inked"),
    ("overinked", "over-inked / dot-gain (merged)"),
    ("broken_finder", "broken finder 'L' pattern"),
    ("broken_timing", "broken timing pattern (dashed edge)"),
    ("damage", "physical damage / scratch"),
    ("quiet_crowd", "quiet-zone crowding"),
    # catch-alls
    ("no_code", "no code present"),
    ("other", "other (see notes)"),
]
DEFECT_CODES = {c for c, _ in DEFECTS}


@dataclass
class Row:
    file: str
    bbox: tuple[int, int, int, int] | None   # (x, y, w, h) in original px
    defects: list[str] = field(default_factory=list)
    recovered: str = ""                       # "clahe" (CLAHE rescued it) | "none"
    notes: str = ""


# ---- decode helpers (mirror the shipped Reader's two stages) ----------------

def _zx(gray: np.ndarray) -> bytes | None:
    res = zxingcpp.read_barcodes(gray, formats=_DM)
    return res[0].bytes if res else None


def stage_of(image: np.ndarray, truth: bytes) -> str:
    """Which Reader stage decodes `image` to `truth`: 'raw', 'clahe', or 'none'."""
    g = _gray(image)
    if _zx(g) == truth:
        return "raw"
    if _zx(pp.s_clahe(g)) == truth:
        return "clahe"
    return "none"


def raw_zxing_misses(corpus: Path) -> list[Row]:
    """Every image whose RAW zxing pass fails — the study set. Each Row is
    tagged recovered='clahe' (the CLAHE stage rescued it) or 'none' (still
    fails the shipped Reader)."""
    labels = _load_truth(corpus)
    rows: list[Row] = []
    for name in sorted(labels):
        img = cv2.imread(str(corpus / name))
        if img is None:
            continue
        st = stage_of(img, labels[name])
        if st == "raw":
            continue                      # raw zxing already reads it; not in study
        rows.append(Row(file=name, bbox=None, recovered=st))
    return rows


def _load_truth(corpus: Path) -> dict[str, bytes]:
    lp = corpus / "labels.csv"
    out: dict[str, bytes] = {}
    if lp.exists():
        with lp.open(newline="") as f:
            for r in csv.DictReader(f):
                out[r["file"]] = r["payload"].encode()
    return out


# ---- auto pre-fill: bbox + metric-driven defect guesses ---------------------

def auto_bbox(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Rough DataMatrix locator for a pre-fill box (human corrects it). The code
    is a dense fine-gradient region; pick the squarest compact high-gradient
    blob. Returns (x, y, w, h) or None."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 24 or h < 24:
            continue
        ar = w / h
        if not (0.5 < ar < 2.0):          # roughly square
            continue
        fill = cv2.contourArea(c) / (w * h)
        if fill < 0.35:                   # compact, not a sprawl
            continue
        score = cv2.contourArea(c)
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))
    return best[1] if best else None


def auto_metrics(gray: np.ndarray, bbox=None) -> dict:
    """Quantitative image stats over the bbox region (or whole image)."""
    g = gray
    if bbox:
        x, y, w, h = bbox
        g = gray[y:y + h, x:x + w]
    if g.size == 0:
        g = gray
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())          # low => blurry
    contrast = float(g.std())                                  # low => low contrast
    # illumination unevenness: spread of a heavily-blurred (background) version
    bg = cv2.GaussianBlur(g, (0, 0), max(g.shape) / 6 + 1)
    illum = float(bg.max() - bg.min())                         # high => gradient/glare
    bright_frac = float((g > 245).mean())                      # high => blown highlight
    return {"blur": round(blur, 1), "contrast": round(contrast, 1),
            "illum": round(illum, 1), "bright_frac": round(bright_frac, 3)}


def guess_defects(metrics: dict) -> list[str]:
    """Cheap metric-driven first guess (human corrects). Thresholds are coarse;
    they only seed the checkboxes."""
    g = []
    if metrics["blur"] < 80:
        g.append("blur")
    if metrics["contrast"] < 35:
        g.append("low_contrast")
    if metrics["illum"] > 90:
        g.append("uneven_illum")
    if metrics["bright_frac"] > 0.04:
        g.append("glare")
    return g


# ---- CSV round-trip ---------------------------------------------------------

_FIELDS = ["file", "x", "y", "w", "h", "defects", "recovered", "notes"]


def save_rows(path: Path, rows: list[Row]) -> None:
    with path.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(_FIELDS)
        for r in sorted(rows, key=lambda r: r.file):
            x, y, w, h = r.bbox if r.bbox else ("", "", "", "")
            wr.writerow([r.file, x, y, w, h, ";".join(r.defects), r.recovered, r.notes])


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            bbox = None
            if r["x"] != "":
                bbox = (int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"]))
            defects = [d for d in r["defects"].split(";") if d]
            rows.append(Row(r["file"], bbox, defects, r.get("recovered", ""),
                            r.get("notes", "")))
    return rows


# ---- GUI --------------------------------------------------------------------

def run_gui(rows: list[Row], corpus: Path, csv_path: Path, scale: float) -> None:
    """Review each miss: drag a box around the code, tick defects, Save/Next."""
    import tkinter as tk

    factor = max(1, round(1 / scale))          # display subsample (0.5 -> 2)
    root = tk.Tk()
    root.title("DataMatrix defect classifier")
    state = {"i": 0, "bbox": None}             # bbox in ORIGINAL px
    photo = {"ref": None}

    main = tk.Frame(root); main.pack(fill="both", expand=True)
    canvas = tk.Canvas(main, cursor="crosshair", highlightthickness=0)
    canvas.grid(row=0, column=0, rowspan=2, padx=4, pady=4)
    side = tk.Frame(main); side.grid(row=0, column=1, sticky="n", padx=6, pady=4)

    counter = tk.Label(side, font=("TkDefaultFont", 11, "bold")); counter.pack(anchor="w")
    tk.Label(side, fg="#666", text="drag on the image to box the code").pack(anchor="w")
    vars_ = {code: tk.IntVar() for code, _ in DEFECTS}
    for code, label in DEFECTS:
        tk.Checkbutton(side, text=label, variable=vars_[code]).pack(anchor="w")
    tk.Label(side, text="notes:").pack(anchor="w", pady=(6, 0))
    notes = tk.Entry(side, width=32); notes.pack(anchor="w")

    rect = {"id": None}

    def draw_bbox():
        if rect["id"]:
            canvas.delete(rect["id"]); rect["id"] = None
        b = state["bbox"]
        if b:
            x, y, w, h = b
            rect["id"] = canvas.create_rectangle(
                x / factor, y / factor, (x + w) / factor, (y + h) / factor,
                outline="#19ff19", width=2)

    def show():
        i = state["i"]
        if i >= len(rows):
            root.destroy(); return
        r = rows[i]
        photo["ref"] = tk.PhotoImage(file=str(corpus / r.file)).subsample(factor)
        canvas.config(width=photo["ref"].width(), height=photo["ref"].height())
        canvas.delete("all"); rect["id"] = None
        canvas.create_image(0, 0, anchor="nw", image=photo["ref"])
        state["bbox"] = r.bbox
        draw_bbox()
        for code, v in vars_.items():
            v.set(1 if code in r.defects else 0)
        notes.delete(0, tk.END); notes.insert(0, r.notes)
        tag = "CLAHE-rescued" if r.recovered == "clahe" else "still fails"
        counter.config(text=f"{i + 1} / {len(rows)}   ({tag})")

    def on_press(e):
        state["_drag"] = (canvas.canvasx(e.x), canvas.canvasy(e.y))

    def on_drag(e):
        x0, y0 = state["_drag"]; x1, y1 = canvas.canvasx(e.x), canvas.canvasy(e.y)
        if rect["id"]:
            canvas.delete(rect["id"])
        rect["id"] = canvas.create_rectangle(x0, y0, x1, y1, outline="#19ff19", width=2)

    def on_release(e):
        x0, y0 = state["_drag"]; x1, y1 = canvas.canvasx(e.x), canvas.canvasy(e.y)
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return                              # ignore a click (keep existing box)
        x, y = int(min(x0, x1) * factor), int(min(y0, y1) * factor)
        w, h = int(abs(x1 - x0) * factor), int(abs(y1 - y0) * factor)
        state["bbox"] = (x, y, w, h)

    def commit_and(advance):
        r = rows[state["i"]]
        r.bbox = state["bbox"]
        r.defects = [code for code, v in vars_.items() if v.get()]
        r.notes = notes.get().strip()
        save_rows(csv_path, rows)
        state["i"] = max(0, state["i"] + advance)
        show()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    bar = tk.Frame(main); bar.grid(row=1, column=1, sticky="sw", padx=6, pady=6)
    tk.Button(bar, text="◀ Prev (saves)", command=lambda: commit_and(-1)).pack(side="left")
    tk.Button(bar, text="Next ⏎ (saves)", command=lambda: commit_and(1)).pack(side="left")
    tk.Button(bar, text="Clear box", command=lambda: (state.update(bbox=None), draw_bbox())).pack(side="left")
    root.bind("<Return>", lambda e: commit_and(1))
    show()
    root.mainloop()


def build_draft(corpus: Path) -> list[Row]:
    """Auto pre-fill: bbox + metric-driven defect guesses for every raw-zxing
    miss. Resumes from an existing defects.csv if present."""
    csv_path = corpus / "defects.csv"
    existing = {r.file: r for r in load_rows(csv_path)}
    rows = raw_zxing_misses(corpus)
    for r in rows:
        if r.file in existing:                  # keep prior human work
            prev = existing[r.file]
            r.bbox, r.defects, r.notes = prev.bbox, prev.defects, prev.notes
            continue
        gray = cv2.cvtColor(cv2.imread(str(corpus / r.file)), cv2.COLOR_BGR2GRAY)
        r.bbox = auto_bbox(gray)
        r.defects = guess_defects(auto_metrics(gray, r.bbox))
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scale", type=float, default=0.5)
    args = ap.parse_args()
    corpus = Path(args.corpus)
    csv_path = corpus / "defects.csv"
    rows = build_draft(corpus)
    n_clahe = sum(1 for r in rows if r.recovered == "clahe")
    print(f"study set: {len(rows)} raw-zxing misses "
          f"({n_clahe} CLAHE-rescued, {len(rows) - n_clahe} still fail). "
          f"draft -> {csv_path}")
    save_rows(csv_path, rows)
    if rows:
        run_gui(rows, corpus, csv_path, args.scale)
    print(f"defects.csv now has {len(load_rows(csv_path))} rows")


if __name__ == "__main__":
    main()
