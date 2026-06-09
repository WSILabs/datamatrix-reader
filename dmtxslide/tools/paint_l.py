"""Minimal B/W pencil editor for the manual_L_fix crops — repair broken DataMatrix L
finders by hand and decode in-tool.

Zoomable/scrollable canvas. Tools: Black / White pencil, brush-size slider, Zoom
in/out (buttons or mouse-wheel), Reset (to the pristine crop), Save (overwrite the
crop), Decode (run the current edit through the full Reader), Prev/Next.

    python -m tools.paint_l
"""
import csv
import re
from pathlib import Path

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw

from dmtxslide.reader import Reader

CORPUS = Path("corpus/wsi_labels")
FOLDER = CORPUS / "manual_L_fix"
VIEW = 820          # canvas viewport size (px); image scrolls/zooms inside it


def _ground_truth() -> dict:
    gt = {}
    for r in csv.DictReader((CORPUS / "labels.csv").open(newline="")):
        m = re.search(r"scan_(\d+)_", r["file"])
        if m:
            gt[m.group(1)] = r["payload"].encode()
    return gt


def main():
    files = sorted(p for p in FOLDER.glob("*.png"))
    if not files:
        print(f"no crops in {FOLDER}")
        return
    edited = FOLDER / "edited"           # edits go HERE; the crops stay pristine
    edited.mkdir(exist_ok=True)

    gt = _ground_truth()
    reader = Reader()
    state = {"i": 0, "color": 0, "brush": 24, "zoom": None}
    work: dict[int, Image.Image] = {}

    def img() -> Image.Image:
        i = state["i"]
        if i not in work:
            e = edited / files[i].name        # resume a prior edit if present, else pristine
            work[i] = Image.open(e if e.exists() else files[i]).convert("L")
        return work[i]

    root = tk.Tk()
    root.title("paint L")
    cframe = tk.Frame(root)
    cframe.grid(row=0, column=0, rowspan=20)
    canvas = tk.Canvas(cframe, width=VIEW, height=VIEW, cursor="crosshair",
                       highlightthickness=0)
    hbar = tk.Scrollbar(cframe, orient="horizontal", command=canvas.xview)
    vbar = tk.Scrollbar(cframe, orient="vertical", command=canvas.yview)
    canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    hbar.grid(row=1, column=0, sticky="ew")

    side = tk.Frame(root)
    side.grid(row=0, column=1, sticky="n", padx=8, pady=8)
    title = tk.Label(side, font=("TkDefaultFont", 12, "bold")); title.pack(anchor="w")
    info = tk.Label(side, fg="#0a7", font=("TkDefaultFont", 11)); info.pack(anchor="w", pady=(0, 6))
    photo = {"ref": None}

    def fit_zoom(im):
        return min(1.0, VIEW / max(im.width, im.height))

    def refresh():
        im = img()
        if state["zoom"] is None:
            state["zoom"] = fit_zoom(im)
        z = state["zoom"]
        shown = im.resize((max(1, int(im.width * z)), max(1, int(im.height * z))),
                          Image.NEAREST)
        photo["ref"] = ImageTk.PhotoImage(shown)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo["ref"])
        canvas.config(scrollregion=(0, 0, shown.width, shown.height))
        scan = re.search(r"scan(\d+)", files[state["i"]].stem).group(1)
        title.config(text=f"{state['i'] + 1}/{len(files)}   scan{scan}   "
                          f"(zoom {int(z * 100)}%)")
        info.config(text="")

    def paint(e):
        z = state["zoom"] or 1.0
        b = state["brush"]
        ix, iy = canvas.canvasx(e.x) / z, canvas.canvasy(e.y) / z   # -> image px
        ImageDraw.Draw(img()).rectangle(
            [ix - b // 2, iy - b // 2, ix + b // 2, iy + b // 2], fill=state["color"])
        refresh()

    canvas.bind("<Button-1>", paint)
    canvas.bind("<B1-Motion>", paint)

    def zoom(factor):
        state["zoom"] = max(0.05, min(8.0, (state["zoom"] or 1.0) * factor))
        refresh()

    canvas.bind("<MouseWheel>", lambda e: zoom(1.1 if e.delta > 0 else 0.9))   # mac/win
    canvas.bind("<Button-4>", lambda e: zoom(1.1))                              # linux
    canvas.bind("<Button-5>", lambda e: zoom(0.9))

    def set_color(c):
        state["color"] = c
        info.config(text=f"pencil: {'BLACK' if c == 0 else 'WHITE'}", fg="#333")

    tk.Button(side, text="Black ●", command=lambda: set_color(0)).pack(fill="x")
    tk.Button(side, text="White ○", command=lambda: set_color(255)).pack(fill="x")
    bsc = tk.Scale(side, from_=2, to=60, orient="horizontal", label="brush px",
                   command=lambda v: state.update(brush=int(v)))
    bsc.set(24); bsc.pack(fill="x", pady=(6, 6))
    zf = tk.Frame(side); zf.pack(fill="x")
    tk.Button(zf, text="Zoom +", command=lambda: zoom(1.25)).pack(side="left", expand=True, fill="x")
    tk.Button(zf, text="Zoom −", command=lambda: zoom(0.8)).pack(side="left", expand=True, fill="x")
    tk.Button(side, text="Fit", command=lambda: (state.update(zoom=fit_zoom(img())), refresh())).pack(fill="x", pady=(2, 6))

    def reset():
        e = edited / files[state["i"]].name
        if e.exists():
            e.unlink()
        work[state["i"]] = Image.open(files[state["i"]]).convert("L")   # pristine crop
        refresh(); info.config(text="reset to original", fg="#333")

    def autosave():
        if state["i"] in work:
            work[state["i"]].save(edited / files[state["i"]].name)   # -> edited/, crop untouched

    def save():
        autosave(); info.config(text="saved", fg="#333")

    def decode():
        autosave()                                     # never lose the edit you just tested
        r = reader.read(np.asarray(img()))
        scan = re.search(r"scan(\d+)", files[state["i"]].stem).group(1)
        if r.payload == gt.get(scan):
            info.config(text=f"DECODED ✓  (stage {r.stage})", fg="#0a7")
        elif r.payload is not None:
            info.config(text="WRONG decode (not the label!)", fg="#c00")
        else:
            info.config(text="no decode", fg="#c00")

    tk.Button(side, text="Reset", command=reset).pack(fill="x")
    tk.Button(side, text="Save", command=save).pack(fill="x")
    tk.Button(side, text="Decode", command=decode).pack(fill="x", pady=(6, 6))

    def nav(d):
        autosave()                  # persist before leaving this image
        state["i"] = max(0, min(len(files) - 1, state["i"] + d))
        state["zoom"] = None        # re-fit the new image
        refresh()

    tk.Button(side, text="◀ Prev", command=lambda: nav(-1)).pack(fill="x")
    tk.Button(side, text="Next ▶", command=lambda: nav(1)).pack(fill="x")

    def on_close():
        autosave()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)

    set_color(0)
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
