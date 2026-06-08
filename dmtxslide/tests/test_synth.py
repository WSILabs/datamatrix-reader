"""Tests for the synthetic degradation model (synth.py)."""
import random

import cv2
import numpy as np

from dmtxslide import synth
from dmtxslide.reader import Reader


def _contrast(substrate_ink):
    sub, ink = substrate_ink
    return sum(abs(int(s) - int(i)) for s, i in zip(sub, ink))


def _easy_params(key, val):
    """DegradeParams with every axis at its easiest value, except `key`=`val`.
    Crowding off, so this isolates whether the *degradation* keeps the code
    readable (degeneracy) from whether the reader can *localize* it."""
    sub, ink = max(synth.AXES["substrate"], key=_contrast)
    f = dict(module_px=max(synth.AXES["module_px"]),
             blur_sigma=min(synth.AXES["blur_sigma"]),
             ink_gain=min(synth.AXES["ink_gain"]),
             rotation_deg=min(synth.AXES["rotation_deg"]),
             substrate_bgr=sub, print_bgr=ink, quiet_crowd=False)
    if key == "substrate":
        f["substrate_bgr"], f["print_bgr"] = val
    else:
        f[key] = val
    return synth.DegradeParams(**f)


def test_ink_gain_does_not_annihilate_low_px_codes():
    """Dot gain must scale with module pitch. A fixed-pixel morphology turns a
    low-px code into a solid blob (physically unreadable by ANY reader, so an
    unfair benchmark sample). Max ink_gain at the smallest module size,
    otherwise clean, must still decode."""
    reader = Reader()
    payload = b"S25-04821-A3"
    p = synth.DegradeParams(
        module_px=min(synth.AXES["module_px"]),
        ink_gain=max(synth.AXES["ink_gain"]),
        blur_sigma=0.0, rotation_deg=0.0,
        substrate_bgr=(255, 255, 255), print_bgr=(10, 10, 10),
        quiet_crowd=False,
    )
    img = synth.degrade(synth.render(payload), p, random.Random(0))
    assert reader.read(img, budget_ms=500).payload == payload


def test_every_axis_value_is_decodable_when_otherwise_easy():
    """No degenerate strata: each AXES value must yield a code the reader can
    decode when all other axes are easy. A 0% stratum should mean a reader
    weakness, never a benchmark that asks the impossible."""
    reader = Reader()
    payload = b"S25-04821-A3"
    for key, vals in synth.AXES.items():
        for val in vals:
            p = _easy_params(key, val)
            img = synth.degrade(synth.render(payload), p, random.Random(0))
            res = reader.read(img, budget_ms=500)
            assert res.payload == payload, f"degenerate stratum: {key}={val!r}"


def test_crowd_quiet_zone_preserves_code_and_adds_margin_text():
    """Crowding must invade the quiet-zone MARGIN, never overwrite the code.

    Real pathology labels crowd accession text against the code in the quiet
    zone; the code modules themselves stay intact. The failure that creates is
    localization (finding the code next to text), not decode corruption.
    """
    rng = random.Random(0)
    # stand-in "code": a distinctive solid block, clearly != white substrate
    code = np.full((40, 40, 3), (0, 0, 200), np.uint8)
    substrate = (255, 255, 255)
    ink = (0, 0, 0)

    out = synth.crowd_quiet_zone(code, substrate, ink, rng)

    # canvas grew (a margin was added on which to crowd text)
    assert out.shape[0] > code.shape[0] and out.shape[1] > code.shape[1]

    # the code appears, pixel-identical, centered in the canvas
    H, W = out.shape[:2]
    h, w = code.shape[:2]
    y0, x0 = (H - h) // 2, (W - w) // 2
    assert np.array_equal(out[y0:y0 + h, x0:x0 + w], code), "code modules were altered"

    # the margin carries ink text (non-substrate pixels) adjacent to the code
    margin_ink = ((out[:y0] != 255).any() or (out[y0 + h:] != 255).any()
                  or (out[:, :x0] != 255).any() or (out[:, x0 + w:] != 255).any())
    assert margin_ink, "no crowding text found in the quiet-zone margin"


def test_degrade_routes_crowding_through_margin():
    """degrade(quiet_crowd=True) must add a quiet-zone margin (route through
    crowd_quiet_zone), so the output is larger than the uncrowded render."""
    grid = synth.render(b"S25-04821-A3")
    p_no = synth.DegradeParams(module_px=6.0, quiet_crowd=False)
    p_yes = synth.DegradeParams(module_px=6.0, quiet_crowd=True)
    a = synth.degrade(grid, p_no, random.Random(0))
    b = synth.degrade(grid, p_yes, random.Random(0))
    assert b.shape[0] > a.shape[0] and b.shape[1] > a.shape[1], \
        "crowding should add a quiet-zone margin, not draw over the code"


def test_render_is_1px_module_binary_and_round_trips():
    grid = synth.render(b"S25-04821-A3")
    assert grid.ndim == 2 and grid.dtype == np.uint8
    assert set(np.unique(grid)).issubset({0, 255})
    big = cv2.resize(grid, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    assert Reader().read(big).payload == b"S25-04821-A3"


def test_render_payload_pool_spans_multiple_symbol_sizes():
    from bench.harness import _payload_pool
    sizes = {synth.render(p).shape for p in _payload_pool()}
    assert len(sizes) >= 3, f"only {len(sizes)} symbol size(s): {sizes}"
