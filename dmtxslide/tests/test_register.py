"""Synthetic (PHI-free) tests for the finder-registration recovery.

Builds a known square DataMatrix, then verifies decode_auto recovers it even when the
finder/timing border is erased (the core capability) and when a fake straight edge (a
"slide rim") is added (the gradient-anisotropy detector must reject it).
"""
import cv2
import numpy as np
import pytest
import zxingcpp

from dmtxslide.register import decode_auto, detect_area, border_mask, render_symbol

_DM = zxingcpp.BarcodeFormat.DataMatrix


def _square_symbol():
    """Return (payload_bytes, MxM bool dark grid) for a payload that encodes SQUARE."""
    for text in ("DMTXSLIDE-REGISTER-TEST", "1-S-24-14215 A1-9 SATB2",
                 "ABCDEFGHIJKLMNOPQRSTUVWX", "REGISTRATION-SELFTEST-0001"):
        arr = np.asarray(zxingcpp.create_barcode(text, _DM).to_image())
        h, w = arr.shape[:2]
        if h == w:                                   # square (incl 1px quiet each side)
            core = arr[1:-1, 1:-1]                    # strip quiet zone -> M×M
            return text.encode(), core < 128         # True = dark module
    pytest.skip("no square symbol found among test payloads")


def _canvas(dark, cell=20, quiet=4):
    """Render an M×M dark grid to a grayscale image at `cell` px/module with a white
    quiet zone of `quiet` modules."""
    sym = np.where(dark, 0, 255).astype(np.uint8)
    sym = cv2.copyMakeBorder(sym, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(sym, None, fx=cell, fy=cell, interpolation=cv2.INTER_NEAREST)


def test_render_symbol_roundtrips_through_zxing():
    payload, dark = _square_symbol()
    assert decode_auto is not None
    assert _zxing_ok(render_symbol(dark, dark.shape[0]), payload)


def _zxing_ok(img, payload):
    res = zxingcpp.read_barcodes(np.ascontiguousarray(img), formats=_DM)
    return bool(res) and res[0].bytes == payload


def test_decode_auto_recovers_clean_code():
    payload, dark = _square_symbol()
    img = _canvas(dark)
    got, params = decode_auto(img)
    assert got == payload
    assert params["M"] == dark.shape[0]


def test_decode_auto_recovers_broken_border():
    """Core capability: corrupt the timing arms entirely (the real WSI defect — the top
    timing row prints half-height) and nick the finder L (a chip). The intact data plus
    the canonical border repaint must still decode."""
    payload, dark = _square_symbol()
    M = dark.shape[0]
    bad = dark.copy()
    bad[0, :] = False                                # erase top timing row
    bad[:, -1] = False                               # erase right timing col
    bad[M // 3:M // 3 + 3, 0] = False                # chip: nick the left finder arm
    img = _canvas(bad)
    got, _ = decode_auto(img)
    assert got == payload


def test_detect_area_rejects_straight_edge():
    """A fake slide rim (long black bar) has gradient in one direction only; the
    anisotropy detector must still center on the code, and decode must still work."""
    payload, dark = _square_symbol()
    img = _canvas(dark)
    img[:, :12] = 0                                  # black vertical bar down the left edge
    area = detect_area(img)
    assert area is not None
    cx, cy, ext, _ = area
    H, W = img.shape
    assert cx > 30                                   # centered on the code, not the bar
    got, _ = decode_auto(img)
    assert got == payload


def test_decode_auto_returns_none_on_blank():
    blank = np.full((400, 400), 255, np.uint8)
    got, params = decode_auto(blank)
    assert got is None and params is None


def test_decode_auto_uses_two_detectors():
    # the decode path must no longer call detect_dark_region
    import inspect
    from dmtxslide import register
    src = inspect.getsource(register.decode_auto)
    assert "detect_dark_region" not in src
    assert "detect_area" in src and "detect_data_region" in src


def test_recover_decodes_offcenter_scene():
    import random
    from dmtxslide import synth
    from dmtxslide.register import recover
    rng = random.Random(3)
    for t in (b"DMTXSLIDE-RECOVER-TEST", b"ABCDEFGHIJKLMNOPQRSTUVWX"):
        import zxingcpp
        a = np.asarray(zxingcpp.create_barcode(t.decode(),
              zxingcpp.BarcodeFormat.DataMatrix).to_image())
        if a.shape[0] == a.shape[1]:
            payload = t; break
    # code in the lower-right (NOT the old upper-left ROI), with border defects
    p = synth.SceneParams(canvas=(900, 1100), cell=18, pos=(0.72, 0.68),
                          rotation_deg=180.0, defects=True, text=True, edges=True)
    img, truth = synth.scene(payload, p, rng)
    assert recover(img[..., 0]) == payload
