"""Parametric finder/timing STAMP aligner for the 7 hard DataMatrix crops.

Instead of free-hand drawing (tools/paint_l.py), overlay a parametric ECC200 border
template — solid L finder (left col + bottom row) + alternating timing (top row +
right col) + a quiet zone — and align it to the real code by tweaking:

    symbol size M (square) · cell size (px/module) · rotation · center x/y · quiet zone

The overlay and the decode-sampling use the SAME rotation transform, so what you align
is exactly what gets sampled. Decode samples the data modules under the aligned stamp,
repaints the canonical border, and hands a clean symbol to zxing (ECC-validated → a bad
fit fails safe). A small local refine (±2px / ±1 cell / ±1°) means you only need to get
close. Every code's parameters persist to manual_L_fix/stamp_params.json so we record
where/how the pattern sits for all 7.

    .venv/bin/python -m tools.stamp_align

Controls: drag = move stamp · arrow keys = nudge 1px · [ ] = rotate ∓0.5° ·
- = / + = cell ∓0.5px · mouse-wheel = zoom canvas · buttons for everything else.
"""
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import zxingcpp

CORPUS = Path("corpus/wsi_labels")
FOLDER = CORPUS / "manual_L_fix"
PARAMS = FOLDER / "stamp_params.json"
VIEW = 860
SIZES = [10, 12, 14, 16, 18, 20, 22, 24, 26]      # valid square ECC200 sizes
_DM = zxingcpp.BarcodeFormat.DataMatrix

# DataMatrix border repaint + zxing call live in the package (single source of truth).
from datamatrix_reader.register import border_mask, render_symbol, _zxing       # noqa: E402,F401


def module_center(lx: float, ly: float, cx: float, cy: float, deg: float):
    """Map a template-local offset (lx,ly) to image coords given center + rotation."""
    t = math.radians(deg)
    cos, sin = math.cos(t), math.sin(t)
    return cx + lx * cos - ly * sin, cy + lx * sin + ly * cos


def sample_grid(gray: np.ndarray, cx, cy, cell, M, deg) -> np.ndarray:
    """Sample an M×M grid of module values under the aligned stamp. Dark via the
    p10/p90 midpoint of module means."""
    h, w = gray.shape
    win = max(1, int(cell * 0.45))
    means = np.empty((M, M), np.float32)
    for i in range(M):
        for j in range(M):
            lx = (j + 0.5 - M / 2) * cell
            ly = (i + 0.5 - M / 2) * cell
            X, Y = module_center(lx, ly, cx, cy, deg)
            xi, yi = int(round(X)), int(round(Y))
            x0, y0 = max(0, xi - win // 2), max(0, yi - win // 2)
            patch = gray[y0:y0 + win, x0:x0 + win]
            means[i, j] = patch.mean() if patch.size else 255.0
    thr = (float(np.percentile(means, 10)) + float(np.percentile(means, 90))) / 2.0
    return means < thr


def _module_quad(i, j, cx, cy, cell, M, deg):
    lx0, ly0 = (j - M / 2) * cell, (i - M / 2) * cell
    return np.array([module_center(lx0, ly0, cx, cy, deg),
                     module_center(lx0 + cell, ly0, cx, cy, deg),
                     module_center(lx0 + cell, ly0 + cell, cx, cy, deg),
                     module_center(lx0, ly0 + cell, cx, cy, deg)], np.int32)


def border_clean_inplace(gray, cx, cy, cell, M, deg) -> np.ndarray:
    """Manual-repair analog: paint a crisp finder/timing border onto a COPY of the
    original crop at the aligned grid, keep interior pixels, let zxing sample. This is
    the path tools/paint_l.py proved (5/7 read at raw after border repair)."""
    out = gray.copy()
    bm = border_mask(M)
    for i in range(M):
        for j in range(M):
            if not (i in (0, M - 1) or j in (0, M - 1)):
                continue
            cv2.fillConvexPoly(out, _module_quad(i, j, cx, cy, cell, M, deg),
                               0 if bm[i, j] else 255)
    return out


def decode_refine(gray, cx, cy, cell, M, deg):
    """Try the aligned params + a small local refine, via BOTH reconstruction (resample
    + render) and in-place border repaint (paint_l analog). Returns (payload,
    refined_params) or (None, None). zxing auto-rotates/inverts."""
    for ddeg in (0, -1, 1):
        for dcell in (0, -1, 1):
            for dcx in (0, -2, 2):
                for dcy in (0, -2, 2):
                    pc, pe, pl, pM, pd = cx + dcx, cy + dcy, cell + dcell, M, deg + ddeg
                    try:
                        p = _zxing(render_symbol(sample_grid(gray, pc, pe, pl, pM, pd), pM))
                        if p is None:
                            p = _zxing(border_clean_inplace(gray, pc, pe, pl, pM, pd))
                    except cv2.error:
                        continue
                    if p is not None:
                        return p, (pc, pe, pl, pM, pd)
    return None, None


# ---- params persistence ----------------------------------------------------------

def load_params() -> dict:
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else {}


def ground_truth() -> dict:
    gt = {}
    for r in csv.DictReader((CORPUS / "labels.csv").open(newline="")):
        m = re.search(r"scan_(\d+)_", r["file"])
        if m:
            gt[m.group(1)] = r["payload"].encode()
    return gt


# ---- GUI -------------------------------------------------------------------------

def main():
    files = sorted(FOLDER.glob("*.png"))
    if not files:
        print(f"no crops in {FOLDER}")
        return
    gt = ground_truth()
    saved = load_params()

    imgs = [np.asarray(Image.open(f).convert("L")) for f in files]

    def scan_of(i):
        return re.search(r"scan(\d+)", files[i].stem).group(1)

    def init_state(i):
        g = imgs[i]
        h, w = g.shape
        s = saved.get(scan_of(i))
        if s:
            return dict(M=s["M"], cell=s["cell"], deg=s["deg"], cx=s["cx"], cy=s["cy"])
        # initial guess: bbox of the largest dark blob -> center + cell
        th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        M0 = 20
        if cnts:
            x, y, bw, bh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
            return dict(M=M0, cell=max(bw, bh) / M0, deg=0.0,
                        cx=x + bw / 2, cy=y + bh / 2)
        return dict(M=M0, cell=min(w, h) / (M0 + 4), deg=0.0, cx=w / 2, cy=h / 2)

    st = {"i": 0, "zoom": 1.0, "opacity": 110, "quiet": 3, "drag": None}
    st.update(init_state(0))

    root = tk.Tk()
    root.title("stamp align")

    cframe = tk.Frame(root)
    cframe.grid(row=0, column=0, rowspan=30)
    canvas = tk.Canvas(cframe, width=VIEW, height=VIEW, cursor="crosshair",
                       highlightthickness=0, bg="#222")
    hbar = tk.Scrollbar(cframe, orient="horizontal", command=canvas.xview)
    vbar = tk.Scrollbar(cframe, orient="vertical", command=canvas.yview)
    canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vbar.grid(row=0, column=1, sticky="ns")
    hbar.grid(row=1, column=0, sticky="ew")

    side = tk.Frame(root)
    side.grid(row=0, column=1, sticky="n", padx=10, pady=8)
    title = tk.Label(side, font=("TkDefaultFont", 12, "bold")); title.pack(anchor="w")
    pinfo = tk.Label(side, font=("TkMonospaceFont", 10), justify="left", fg="#333")
    pinfo.pack(anchor="w", pady=(2, 4))
    status = tk.Label(side, font=("TkDefaultFont", 11, "bold")); status.pack(anchor="w", pady=(0, 6))
    photo = {"ref": None}

    def transform(lx, ly):
        return module_center(lx, ly, st["cx"], st["cy"], st["deg"])

    def build_overlay(base_rgb: Image.Image) -> Image.Image:
        M, cell = st["M"], st["cell"]
        lay = Image.new("RGBA", base_rgb.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        a = st["opacity"]
        bm = border_mask(M)
        isb = np.zeros((M, M), bool)
        isb[0, :] = isb[-1, :] = isb[:, 0] = isb[:, -1] = True
        for i in range(M):
            for j in range(M):
                if not isb[i, j]:
                    continue
                lx0, ly0 = (j - M / 2) * cell, (i - M / 2) * cell
                poly = [transform(lx0, ly0), transform(lx0 + cell, ly0),
                        transform(lx0 + cell, ly0 + cell), transform(lx0, ly0 + cell)]
                if bm[i, j]:
                    d.polygon(poly, fill=(255, 40, 40, a))          # dark border module
                else:
                    d.polygon(poly, outline=(40, 220, 255, 255))    # light timing cell
        # data-region outline + quiet-zone outline
        half = M / 2 * cell
        d.polygon([transform(-half, -half), transform(half, -half),
                   transform(half, half), transform(-half, half)],
                  outline=(255, 255, 0, 255))
        q = (M / 2 + st["quiet"]) * cell
        d.polygon([transform(-q, -q), transform(q, -q),
                   transform(q, q), transform(-q, q)],
                  outline=(0, 255, 0, 200))
        return Image.alpha_composite(base_rgb.convert("RGBA"), lay)

    def refresh():
        i = st["i"]
        base = Image.fromarray(imgs[i]).convert("RGB")
        comp = build_overlay(base)
        z = st["zoom"]
        shown = comp.resize((max(1, int(comp.width * z)), max(1, int(comp.height * z))),
                            Image.NEAREST)
        photo["ref"] = ImageTk.PhotoImage(shown)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo["ref"])
        canvas.config(scrollregion=(0, 0, shown.width, shown.height))
        gtv = gt.get(scan_of(i), b"?").decode("latin-1")
        title.config(text=f"{i + 1}/{len(files)}  scan{scan_of(i)}  → {gtv}")
        pinfo.config(text=f"M={st['M']}  cell={st['cell']:.1f}px  rot={st['deg']:.1f}°\n"
                          f"center=({st['cx']:.0f},{st['cy']:.0f})  quiet={st['quiet']}  "
                          f"zoom={int(z*100)}%")

    # --- canvas interactions ---
    def on_press(e):
        st["drag"] = (canvas.canvasx(e.x), canvas.canvasy(e.y), st["cx"], st["cy"])

    def on_drag(e):
        if not st["drag"]:
            return
        x0, y0, cx0, cy0 = st["drag"]
        z = st["zoom"]
        st["cx"] = cx0 + (canvas.canvasx(e.x) - x0) / z
        st["cy"] = cy0 + (canvas.canvasy(e.y) - y0) / z
        refresh()

    canvas.bind("<Button-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<MouseWheel>", lambda e: zoom(1.1 if e.delta > 0 else 0.9))
    canvas.bind("<Button-4>", lambda e: zoom(1.1))
    canvas.bind("<Button-5>", lambda e: zoom(0.9))

    def zoom(f):
        st["zoom"] = max(0.2, min(8.0, st["zoom"] * f)); refresh()

    def nudge(dx, dy):
        st["cx"] += dx; st["cy"] += dy; refresh()

    def rot(d):
        st["deg"] += d; refresh()

    def celld(d):
        st["cell"] = max(2.0, st["cell"] + d); refresh()

    root.bind("<Left>", lambda e: nudge(-1, 0))
    root.bind("<Right>", lambda e: nudge(1, 0))
    root.bind("<Up>", lambda e: nudge(0, -1))
    root.bind("<Down>", lambda e: nudge(0, 1))
    root.bind("[", lambda e: rot(-0.5))
    root.bind("]", lambda e: rot(0.5))
    root.bind("-", lambda e: celld(-0.5))
    root.bind("=", lambda e: celld(0.5))

    # --- controls ---
    mrow = tk.Frame(side); mrow.pack(fill="x", pady=(4, 0))
    tk.Label(mrow, text="size M").pack(side="left")
    mvar = tk.IntVar(value=st["M"])

    def set_M(*_):
        st["M"] = int(mvar.get()); refresh()

    tk.OptionMenu(mrow, mvar, *SIZES, command=lambda v: set_M()).pack(side="left")

    def slider(label, lo, hi, res, getter, setter, step=None):
        step = res if step is None else step
        f = tk.Frame(side); f.pack(fill="x")
        s = tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal",
                     label=label, length=170,
                     command=lambda v: (setter(float(v)), refresh()))
        s.set(getter())
        tk.Button(f, text="−", width=2,
                  command=lambda: s.set(s.get() - step)).pack(side="left", pady=(14, 0))
        s.pack(side="left", fill="x", expand=True)
        tk.Button(f, text="+", width=2,
                  command=lambda: s.set(s.get() + step)).pack(side="left", pady=(14, 0))
        return s

    cell_s = slider("cell size (px/module)", 4, 60, 0.5,
                    lambda: st["cell"], lambda v: st.update(cell=v))
    rot_s = slider("rotation (°)", -180, 180, 0.5,
                   lambda: st["deg"], lambda v: st.update(deg=v))
    op_s = slider("overlay opacity", 0, 255, 5,
                  lambda: st["opacity"], lambda v: st.update(opacity=int(v)))
    q_s = slider("quiet zone (modules)", 0, 6, 1,
                 lambda: st["quiet"], lambda v: st.update(quiet=int(v)))

    def sync_sliders():
        cell_s.set(st["cell"]); rot_s.set(st["deg"]); mvar.set(st["M"])

    zf = tk.Frame(side); zf.pack(fill="x", pady=(6, 2))
    tk.Button(zf, text="Zoom +", command=lambda: zoom(1.25)).pack(side="left", expand=True, fill="x")
    tk.Button(zf, text="Zoom −", command=lambda: zoom(0.8)).pack(side="left", expand=True, fill="x")

    def decode():
        g = imgs[st["i"]]
        payload, refined = decode_refine(g, st["cx"], st["cy"], st["cell"],
                                         st["M"], st["deg"])
        truth = gt.get(scan_of(st["i"]))
        if payload == truth:
            if refined:
                st["cx"], st["cy"], st["cell"], st["M"], st["deg"] = refined
                sync_sliders(); refresh()
            status.config(text="DECODED ✓ (params snapped to fit)", fg="#0a7")
        elif payload is not None:
            status.config(text="WRONG decode (not the label!)", fg="#c00")
        else:
            status.config(text="no decode — tweak alignment & retry", fg="#c00")

    def save():
        saved[scan_of(st["i"])] = dict(M=st["M"], cell=round(st["cell"], 2),
                                       deg=round(st["deg"], 2),
                                       cx=round(st["cx"], 1), cy=round(st["cy"], 1),
                                       quiet=st["quiet"])
        PARAMS.write_text(json.dumps(saved, indent=2))
        status.config(text=f"saved params → {PARAMS.name}", fg="#333")

    tk.Button(side, text="Decode", command=decode,
              font=("TkDefaultFont", 11, "bold")).pack(fill="x", pady=(8, 2))
    tk.Button(side, text="Save params", command=save).pack(fill="x")

    def nav(d):
        save()                                   # persist before leaving
        st["i"] = max(0, min(len(files) - 1, st["i"] + d))
        st.update(init_state(st["i"]))
        st["zoom"] = 1.0
        sync_sliders(); status.config(text=""); refresh()

    nf = tk.Frame(side); nf.pack(fill="x", pady=(8, 0))
    tk.Button(nf, text="◀ Prev", command=lambda: nav(-1)).pack(side="left", expand=True, fill="x")
    tk.Button(nf, text="Next ▶", command=lambda: nav(1)).pack(side="left", expand=True, fill="x")

    tk.Label(side, text="drag=move · arrows=nudge · [ ]=rotate · -/+ =cell · wheel=zoom",
             font=("TkDefaultFont", 8), fg="#777", wraplength=240,
             justify="left").pack(anchor="w", pady=(10, 0))

    root.protocol("WM_DELETE_WINDOW", lambda: (save(), root.destroy()))
    sync_sliders()
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
