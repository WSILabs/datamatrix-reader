"""Generate a YOLO-format DataMatrix detection dataset from the synthetic scene generator
(truth-controlled boxes). Output goes to /tmp/dm_yolo (ephemeral spike data, not in repo).

Each scene places a square DataMatrix anywhere on a label canvas at varied scale/rotation
with the real confounders (border defects, glass chip, slide edges, text, color stock,
blur, noise). The label box is the code's axis-aligned bounding box, clipped to the image.

    .venv/bin/python -m tools.make_yolo_dataset
"""
import random
from pathlib import Path

import cv2
import numpy as np
import zxingcpp

from datamatrix_reader import synth

_DM = zxingcpp.BarcodeFormat.DataMatrix
ROOT = Path("/tmp/dm_yolo")

# pathology-style payloads; keep only the ones that encode to a SQUARE symbol
_POOL = [b"S25-04821 A3-1 HE", b"PCAA00028208 A1-1", b"1-S-25-00828 A8-1",
         b"ABCDEFGHIJKLMNOPQRSTUVWX", b"370956.1/10 PAS", b"GDC-04-123456 B2",
         b"I00725098E", b"1-S-24-14215 A1-9 SATB2"]
def _palette(rng):
    """Realistic (substrate_bgr, ink_bgr): the real labels are NOT pure black on white —
    ink is usually dark GRAY, stock light GRAY (sometimes a pale colored tint). Light stock
    ~200-248, dark-gray ink ~30-115, with contrast clamped >=70 so it stays decodable —
    including the hard low-contrast (laser-etch) end. Used for BOTH positives and
    distractors so color carries no DataMatrix-vs-not signal."""
    base = rng.randint(200, 248)
    tint = rng.choice([(1.0, 1.0, 1.0)] * 4              # neutral light gray (most common)
                      + [(0.85, 1.0, 1.0), (1.0, 0.92, 1.0),   # pale yellow / pink (BGR)
                         (0.90, 1.0, 0.92), (1.0, 0.95, 0.86)])  # pale green / blue
    sub = tuple(int(min(255, base * c)) for c in tint)
    ink_v = rng.randint(30, 115)
    ink_v = min(ink_v, min(sub) - 70)                   # guarantee decodable contrast
    ink = (ink_v, ink_v, ink_v)
    return sub, ink


def _distractor(rng, sub, ink):
    """A BGR tile that is NOT a DataMatrix — teaches the detector to reject look-alikes:
    QR / Aztec codes (other dense square 2D grids) and a perforated cassette mesh.

    Colorized with the SAME (sub, ink) palette the positives use, so color carries no
    information about DataMatrix-vs-not — the model must learn the structural difference,
    not 'colored grid = code'."""
    subf = np.array(sub, np.float32) / 255.0
    inkf = np.array(ink, np.float32) / 255.0
    kind = rng.choice(["qr", "aztec", "mesh", "mesh"])
    if kind in ("qr", "aztec"):
        fmt = (zxingcpp.BarcodeFormat.QRCode if kind == "qr"
               else zxingcpp.BarcodeFormat.Aztec)
        a = np.asarray(zxingcpp.create_barcode(
            rng.choice(["https://x.co/" + str(rng.randint(0, 9999)),
                        "ABC-" + str(rng.randint(0, 9999))]), fmt).to_image())
        s = rng.uniform(6, 16)
        t = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        tn = t.astype(np.float32)[..., None] / 255.0          # 1=light, 0=dark
        return ((tn * subf + (1 - tn) * inkf) * 255).clip(0, 255).astype(np.uint8)
    # perforated mesh: a regular grid of holes (ink) on the substrate stock (cassette well)
    cell = rng.randint(7, 14)
    n = rng.randint(14, 28)
    t = np.full((n * cell, n * cell, 3), sub, np.uint8)
    for i in range(n):
        for j in range(n):
            cv2.circle(t, (int((j + 0.5) * cell), int((i + 0.5) * cell)),
                       max(1, cell // 4), tuple(int(c) for c in ink), -1)
    return t


def _paste_distractor(canvas, box_xyxy, rng):
    """Paste a distractor onto canvas at a random spot NOT overlapping box_xyxy (the DM).
    Distractor color is drawn from the SAME palette as positives (see _distractor)."""
    H, W = canvas.shape[:2]
    sub, ink = _palette(rng)
    d = _distractor(rng, sub, ink)
    dh, dw = d.shape[:2]
    if dh >= H or dw >= W:
        return
    for _ in range(8):
        x, y = rng.randint(0, W - dw), rng.randint(0, H - dh)
        if box_xyxy is not None:
            bx0, by0, bx1, by1 = box_xyxy
            if not (x + dw < bx0 or x > bx1 or y + dh < by0 or y > by1):
                continue                              # overlaps the DM — retry
        canvas[y:y + dh, x:x + dw] = d
        return


def _square_payloads():
    # filter using the BYTES encoding (what synth.render/scene actually use); the str and
    # bytes encodings choose different symbol shapes, so checking the str form is wrong.
    out = []
    for p in _POOL:
        a = np.asarray(zxingcpp.create_barcode(p, _DM).to_image())
        if a.shape[0] == a.shape[1]:
            out.append(p)
    if not out:
        raise RuntimeError("no square-encoding payloads in pool")
    return out


def _bbox(truth, W, H):
    """Axis-aligned, image-clipped YOLO box (cx,cy,w,h normalized) or None if the code is
    mostly off-canvas. A square side `size` rotated by `angle` spans size*(|cos|+|sin|)."""
    ang = np.radians(truth["angle"])
    span = truth["size"] * (abs(np.cos(ang)) + abs(np.sin(ang)))
    x0, y0 = truth["cx"] - span / 2, truth["cy"] - span / 2
    x1, y1 = truth["cx"] + span / 2, truth["cy"] + span / 2
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if (cx1 - cx0) < 0.6 * span or (cy1 - cy0) < 0.6 * span:
        return None                                   # too clipped — drop
    cx, cy = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    return cx / W, cy / H, (cx1 - cx0) / W, (cy1 - cy0) / H


def _gen(split, n, seed, neg_frac=0.18):
    rng = random.Random(seed)
    (ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    pays = _square_payloads()
    made = 0
    for i in range(n):
        H = rng.choice([700, 800, 900, 1000])
        W = rng.choice([900, 1000, 1100, 1200])
        sub, ink = _palette(rng)
        name = f"{split}_{i:05d}"
        if rng.random() < neg_frac:
            # PURE NEGATIVE: QR/Aztec/mesh distractors, NO DataMatrix -> empty label.
            canvas = np.full((H, W, 3), sub, np.uint8)
            for _ in range(rng.randint(1, 3)):
                _paste_distractor(canvas, None, rng)
            cv2.imwrite(str(ROOT / "images" / split / f"{name}.png"), canvas)
            (ROOT / "labels" / split / f"{name}.txt").write_text("")
            made += 1
            continue
        p = synth.SceneParams(
            canvas=(H, W), cell=rng.uniform(8, 22),
            pos=(rng.uniform(0.22, 0.78), rng.uniform(0.22, 0.78)),
            rotation_deg=rng.choice([0, 90, 180, 270]), skew_deg=rng.uniform(-18, 18),
            substrate_bgr=sub, print_bgr=ink, text=True,
            edges=rng.random() < 0.5, chip=rng.random() < 0.3,
            defects=rng.random() < 0.7, blur_sigma=rng.uniform(0, 1.0),
            noise_sigma=rng.uniform(0, 4))
        img, truth = synth.scene(rng.choice(pays), p, rng)
        box = _bbox(truth, W, H)
        if box is None:
            continue
        # composite 0-2 distractors that do NOT overlap the real DM box
        cx, cy, bw, bh = box
        px = (int((cx - bw / 2) * W), int((cy - bh / 2) * H),
              int((cx + bw / 2) * W), int((cy + bh / 2) * H))
        for _ in range(rng.randint(0, 2)):
            _paste_distractor(img, px, rng)
        cv2.imwrite(str(ROOT / "images" / split / f"{name}.png"), img)
        (ROOT / "labels" / split / f"{name}.txt").write_text(
            "0 %.6f %.6f %.6f %.6f\n" % box)
        made += 1
    return made


def main():
    tr = _gen("train", 2500, 0)
    va = _gen("val", 350, 1)
    (ROOT / "data.yaml").write_text(
        f"path: {ROOT}\ntrain: images/train\nval: images/val\n"
        "nc: 1\nnames:\n  0: datamatrix\n")
    print(f"dataset at {ROOT}: {tr} train, {va} val images")


if __name__ == "__main__":
    main()
