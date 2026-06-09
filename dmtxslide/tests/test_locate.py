import random
import numpy as np
from dmtxslide import synth
from dmtxslide.locate import propose


def _payload():
    import zxingcpp
    for t in (b"DMTXSLIDE-LOCATE-TEST-1", b"ABCDEFGHIJKLMNOPQRSTUVWX"):
        a = np.asarray(zxingcpp.create_barcode(t.decode(),
              zxingcpp.BarcodeFormat.DataMatrix).to_image())
        if a.shape[0] == a.shape[1]:
            return t
    raise AssertionError


def _hit(cands, truth, tol_frac=0.4):
    """A proposal counts as a hit if its center is within tol*size of truth and its
    size is within 35% of truth."""
    for cx, cy, size, _ in cands:
        if (abs(cx - truth["cx"]) < tol_frac * truth["size"] and
                abs(cy - truth["cy"]) < tol_frac * truth["size"] and
                abs(size - truth["size"]) < 0.35 * truth["size"]):
            return True
    return False


def test_propose_localizes_offcenter_varied_scale():
    rng = random.Random(1)
    payload = _payload()
    hits = 0
    cases = [(0.25, 0.3, 10.0), (0.75, 0.6, 14.0), (0.5, 0.2, 22.0),
             (0.3, 0.7, 28.0), (0.8, 0.8, 18.0)]
    for fx, fy, cell in cases:
        p = synth.SceneParams(canvas=(900, 1100), cell=cell, pos=(fx, fy),
                              rotation_deg=0.0, text=True, edges=True)
        img, truth = synth.scene(payload, p, rng)
        cands = propose(img[..., 0])
        if _hit(cands, truth):
            hits += 1
    assert hits >= 4   # localizes >=4/5 across position+scale, despite a slide edge
