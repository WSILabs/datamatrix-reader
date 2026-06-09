"""ROI + pitch clustering experiment (exploratory, throwaway).

Hypothesis: the 397 codes zxing DOES read return a 4-corner `.position`. If labels
are laid out consistently, those positions cluster into a tight code-ROI on the label,
and the symbols share a pixel pitch. If so, we can use that as a localization prior for
the 7 hard codes: re-crop each hard label to the consensus ROI, then re-try raw zxing
and the FFT module-pitch recovery on a clutter-free crop.

Run: .venv/bin/python -m tools.roi_pitch_experiment
"""
import glob
import re
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

_DM = zxingcpp.BarcodeFormat.DataMatrix
CORPUS = Path("corpus/wsi_labels")
HARD = ["329", "330", "331", "342", "394", "96", "432"]


def quad(pos):
    return np.array([[pos.top_left.x, pos.top_left.y],
                     [pos.top_right.x, pos.top_right.y],
                     [pos.bottom_right.x, pos.bottom_right.y],
                     [pos.bottom_left.x, pos.bottom_left.y]], float)


def symbol_modules(payload: bytes) -> int | None:
    """Module count (side) of the symbol zxing would mint for this payload."""
    try:
        img = zxingcpp.create_barcode(payload.decode("latin-1"), _DM).to_image()
        a = np.asarray(img)
        # to_image() is M+2 (1px quiet zone each side) at 1px/module
        return min(a.shape[:2]) - 2
    except Exception:
        return None


def pitch_from_fft(gray: np.ndarray) -> float:
    g = gray.astype(float)
    g -= g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g * win)))
    cy, cx = np.array(F.shape) // 2
    F[cy - 2:cy + 3, cx - 2:cx + 3] = 0
    best_r, best_v = 0, -1.0
    rmax = min(F.shape) // 2
    for r in range(max(3, min(F.shape) // 40), rmax):  # pitch 6..40px band
        ys, xs = np.ogrid[:F.shape[0], :F.shape[1]]
        ring = (np.abs(np.hypot(ys - cy, xs - cx) - r) < 1.0)
        v = F[ring].max() if ring.any() else 0
        if v > best_v:
            best_v, best_r = v, r
    return min(F.shape) / best_r if best_r else float("nan")


def main():
    files = sorted(glob.glob(str(CORPUS / "*.png")))
    centers, sides, pitches = [], [], []
    W = H = None
    n_read = 0
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        res = zxingcpp.read_barcodes(img, formats=_DM)
        if not res:
            continue
        n_read += 1
        q = quad(res[0].position)
        c = q.mean(0)
        centers.append([c[0] / W, c[1] / H])
        side = np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])
        sides.append(side)
        mods = symbol_modules(res[0].bytes)
        if mods:
            pitches.append(side / mods)

    centers = np.array(centers)
    sides = np.array(sides)
    pitches = np.array(pitches)
    print(f"read {n_read} codes; label size ~ {W}x{H}\n")

    cx, cy = centers.mean(0)
    sx, sy = centers.std(0)
    print("=== code CENTER (normalized 0..1) ===")
    print(f"  x: mean {cx:.3f}  std {sx:.3f}  [{centers[:,0].min():.3f}, {centers[:,0].max():.3f}]")
    print(f"  y: mean {cy:.3f}  std {sy:.3f}  [{centers[:,1].min():.3f}, {centers[:,1].max():.3f}]")

    print("\n=== code SIDE length (px) ===")
    print(f"  mean {sides.mean():.1f}  std {sides.std():.1f}  [{sides.min():.0f}, {sides.max():.0f}]")

    print("\n=== module PITCH (px/module) ===")
    print(f"  mean {pitches.mean():.2f}  std {pitches.std():.2f}  "
          f"median {np.median(pitches):.2f}  [{pitches.min():.2f}, {pitches.max():.2f}]")

    # consensus ROI in pixels (mean center +/- 1.5*max half-side, clamped)
    half = sides.max() / 2 * 1.5
    roi_cx, roi_cy = cx * W, cy * H
    x0 = max(0, int(roi_cx - half)); x1 = min(W, int(roi_cx + half))
    y0 = max(0, int(roi_cy - half)); y1 = min(H, int(roi_cy + half))
    print(f"\n=== consensus ROI (px) === x[{x0}:{x1}] y[{y0}:{y1}]  ({x1-x0}x{y1-y0})")

    print("\n=== HARD codes: full-label raw / ROI-crop raw / ROI FFT pitch ===")
    pmed = np.median(pitches)
    for h in HARD:
        m = [f for f in files if re.search(rf"scan_{h}_", f)]
        if not m:
            print(f"  scan{h}: file not found"); continue
        img = cv2.imread(m[0], cv2.IMREAD_GRAYSCALE)
        full_raw = bool(zxingcpp.read_barcodes(img, formats=_DM))
        crop = img[y0:y1, x0:x1]
        crop_raw = bool(zxingcpp.read_barcodes(np.ascontiguousarray(crop), formats=_DM))
        p = pitch_from_fft(crop)
        flag = "" if abs(p - pmed) < pmed * 0.5 else "  <-- off"
        print(f"  scan{h}: full_raw={full_raw!s:5}  roi_raw={crop_raw!s:5}  "
              f"fft_pitch={p:5.1f} (median {pmed:.1f}){flag}")


if __name__ == "__main__":
    main()
