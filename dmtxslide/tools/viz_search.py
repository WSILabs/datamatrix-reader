"""Render the brute-force registration search (register._brute_region) as a playable MP4 so
you can watch, in the EXACT search order, every grid hypothesis the registration tries — the
oriented M x M sampling lattice drawn on the crop, annotated with the symbol size M, cell
pitch, angle/center offsets, the L-solidity score, and whether zxing decoded. This makes the
failure mode visible: e.g. a full wasted sweep of the wrong M before the true one, or the
center walking outward.

Frame colour: GREY = sampled but L too weak to attempt a decode (l < 0.6); YELLOW = decode
attempted but missed; GREEN = the winning registration (held at the end). A side panel tracks
the search dimensions and counts.

    .venv/bin/python -m tools.viz_search scan_96          # pick a corpus label by substring
    .venv/bin/python -m tools.viz_search scan_96 --stride 3 --fps 60 --out /tmp/search.mp4

Per-sample frames can number in the thousands; --stride keeps the video watchable (decode
attempts and the winner are ALWAYS rendered regardless of stride).
"""
import argparse
import glob

import cv2
import numpy as np

import dmtxslide.register as R
from dmtxslide.register import (SIZES, _outward, sample_fast, l_orientations,
                                render_symbol, _zxing, _square_quad, _normalize, _detector,
                                detect_data_region)
from dmtxslide.detect import format_gate
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
DISP = 560          # display height of the crop (px)
PANEL = 320         # side-panel width (px)


def _reconstruct_crop(path):
    """Return the normalized crop `up` decode_auto would receive for this label, or None."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if zxingcpp.read_barcodes(np.ascontiguousarray(g), formats=_DM):
        return None                                   # decodes raw; never reaches autoreg
    det = _detector()
    cands = det.detect(g) if det is not None else []
    for cx, cy, size, *_ in cands:
        up, _tf = _normalize(g, cx, cy, size)
        if up is None:
            continue
        _pl, fmt, _pos = format_gate(up)
        if fmt is None:                               # undecoded region -> this is what autoreg gets
            return up
    return None


def _lattice(cx, cy, cell, M, deg):
    """The M+1 x M+1 grid-line endpoints (module-corner coords) for the sampling lattice."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    Rm = np.array([[c, -s], [s, c]], np.float32)
    k = (np.arange(M + 1) - M / 2.0) * cell
    pts = {}
    for i, u in enumerate(k):
        for j, v in enumerate(k):
            xy = Rm @ np.array([u, v], np.float32) + np.array([cx, cy], np.float32)
            pts[(i, j)] = xy
    return pts


def _draw(base, scale, cx, cy, cell, M, deg, colour, info):
    """One annotated frame: crop + oriented lattice + side panel."""
    H, W = base.shape[:2]
    canvas = np.full((H, W + PANEL, 3), 24, np.uint8)
    canvas[:, :W] = base
    sc = lambda p: (int(p[0] * scale), int(p[1] * scale))
    # module lattice (thin) + bold outer square
    pts = _lattice(cx, cy, cell, M, deg)
    for i in range(M + 1):
        cv2.line(canvas, sc(pts[(i, 0)]), sc(pts[(i, M)]), colour, 1, cv2.LINE_AA)
        cv2.line(canvas, sc(pts[(0, i)]), sc(pts[(M, i)]), colour, 1, cv2.LINE_AA)
    quad = _square_quad(cx, cy, cell * M, deg)
    cv2.polylines(canvas, [np.array([sc(p) for p in quad], np.int32)], True, colour, 2, cv2.LINE_AA)
    cv2.circle(canvas, sc((cx, cy)), 3, colour, -1)
    # side panel text
    x0 = W + 16
    for i, (label, val) in enumerate(info):
        y = 40 + i * 30
        cv2.putText(canvas, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(canvas, val, (x0 + 150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="substring of a corpus label filename (e.g. scan_96)")
    ap.add_argument("--corpus", default="corpus/wsi_labels")
    ap.add_argument("--stride", type=int, default=2, help="render every Nth non-attempt sample")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--out", default="/tmp/search.mp4")
    ap.add_argument("--codec", default="avc1", help="fourcc (avc1=H.264 small; mp4v=fallback)")
    args = ap.parse_args()

    matches = [f for f in sorted(glob.glob(f"{args.corpus}/*.png")) if args.label in f]
    if not matches:
        raise SystemExit(f"no corpus label matches {args.label!r}")
    path = matches[0]
    up = _reconstruct_crop(path)
    if up is None:
        raise SystemExit(f"{path.split('/')[-1]} does not reach the autoreg search (decodes earlier)")
    reg = detect_data_region(up)
    if reg is None:
        raise SystemExit("no texture region detected on this crop")
    cx0, cy0, te, ang0, ocx, ocy, oside = reg          # rect anchors ranges; clip orders

    scale = DISP / up.shape[0]
    base = cv2.cvtColor(cv2.resize(up, None, fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
    fh, fw = base.shape[0], base.shape[1] + PANEL
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*args.codec), args.fps, (fw, fh))
    if not vw.isOpened():                              # codec unavailable -> fall back
        vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (fw, fh))

    GREY, YELLOW, GREEN = (140, 140, 140), (40, 200, 240), (60, 220, 60)
    nsamp = attempts = frames = 0
    won = None

    # Replicate _brute_region's exact iteration order, drawing as we go: ranges anchored to the
    # rect (cx0,cy0,te), order expanding from the coverage-clip estimate (ocx,ocy,oside).
    for M in SIZES:
        cell0 = oside / M
        for cell in _outward(np.arange(te / (M + 3), te / (M - 1), 0.5), cell0):
            for ddeg in _outward(np.arange(-3, 3.01, 1.0), 0.0):
                grid_offs = np.arange(-1.5, 1.51, 0.375) * cell
                offs_x = _outward(grid_offs, ocx - cx0)
                offs_y = _outward(grid_offs, ocy - cy0)
                for dcx in offs_x:
                    for dcy in offs_y:
                        nsamp += 1
                        cx, cy, deg = cx0 + dcx, cy0 + dcy, ang0 + ddeg
                        grid = sample_fast(up, cx, cy, cell, M, deg)
                        best_l = -1.0
                        attempted = hit = False
                        for g, lsc, _ in l_orientations(grid):
                            best_l = max(best_l, lsc)
                            if lsc < 0.6:
                                break
                            attempted = True
                            attempts += 1
                            try:
                                p = _zxing(render_symbol(g, M))
                            except cv2.error:
                                continue
                            if p is not None:
                                hit = True
                                won = (cx, cy, cell, M, deg, p)
                                break
                        render = hit or attempted or (nsamp % args.stride == 0)
                        if render:
                            colour = GREEN if hit else (YELLOW if attempted else GREY)
                            info = [
                                ("symbol M", str(M)),
                                ("cell (px)", f"{cell:.2f}"),
                                ("angle off", f"{ddeg:+.0f} deg"),
                                ("center off", f"({dcx:+.0f},{dcy:+.0f})"),
                                ("L-score", f"{best_l:.2f}"),
                                ("samples", str(nsamp)),
                                ("decode tries", str(attempts)),
                                ("status", "DECODED" if hit else ("decode attempt" if attempted else "L too weak")),
                            ]
                            vw.write(_draw(base, scale, cx, cy, cell, M, deg, colour, info))
                            frames += 1
                        if hit:
                            # hold the winner ~1.5s
                            last = _draw(base, scale, *won[:5], GREEN, [
                                ("RESULT", "DECODED"), ("symbol M", str(M)),
                                ("cell (px)", f"{cell:.2f}"), ("angle off", f"{ddeg:+.0f} deg"),
                                ("center off", f"({dcx:+.0f},{dcy:+.0f})"),
                                ("samples", str(nsamp)), ("decode tries", str(attempts)),
                                ("payload", won[5].decode("latin1")[:18])])
                            for _ in range(int(args.fps * 1.5)):
                                vw.write(last); frames += 1
                            vw.release()
                            print(f"{path.split('/')[-1]}")
                            print(f"  WON at M={M} cell={cell:.2f} ddeg={ddeg:+.0f} off=({dcx:+.0f},{dcy:+.0f})"
                                  f"  after {nsamp} samples / {attempts} decode tries")
                            print(f"  wrote {frames} frames -> {args.out}  ({frames/args.fps:.1f}s @ {args.fps}fps)")
                            return
    vw.release()
    print(f"no decode; wrote {frames} frames -> {args.out}")


if __name__ == "__main__":
    main()
