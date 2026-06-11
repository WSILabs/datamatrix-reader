"""Synthetic tests for the finder-registration recovery.

Builds a known square DataMatrix, then verifies decode_auto recovers it even when the
finder/timing border is erased (the core capability) and when a fake straight edge (a
"slide rim") is added (the gradient-anisotropy detector must reject it).
"""
import cv2
import numpy as np
import pytest
import zxingcpp

from datamatrix_reader.register import (decode_auto, detect_area, border_mask, render_symbol,
                                _square_from_coverage, _fft_pitch)

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
    got, reg = decode_auto(img)
    assert got == payload
    assert reg is not None and len(reg) == 4  # (cx, cy, side, deg)


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


def test_square_from_coverage_clips_narrow_protrusion():
    """The coverage-clip square fit ignores a narrow protrusion (the ink-drip case) that a
    naive minAreaRect would let drag the center/extent. On a clean square it's a no-op."""
    sq = np.zeros((320, 320), np.uint8)
    sq[80:200, 90:210] = 1                           # 120x120 square, center (150,140)
    c = cv2.findContours(sq, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    cx, cy, side, _ = _square_from_coverage(sq.shape, c, 0.0)
    assert abs(cx - 150) < 4 and abs(cy - 140) < 4 and abs(side - 120) < 8   # clean: no-op

    drip = sq.copy()
    drip[200:280, 140:162] = 1                        # thin tail below (a drip)
    cd = cv2.findContours(drip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    dcx, dcy, dside, _ = _square_from_coverage(drip.shape, cd, 0.0)
    (_, rcy), _, _ = cv2.minAreaRect(cd)             # rect center dragged DOWN by the tail
    assert abs(dcy - 140) < 12 and abs(dside - 120) < 16   # clip stays on the square
    assert rcy > dcy + 8                             # ...whereas the rect drifted down


def test_fft_pitch_recovers_known_module_pitch():
    """_fft_pitch must recover a known module pitch (it orders the symbol-size search;
    autocorrelation was biased ~1.5px low, the FFT peak is not)."""
    rng = np.random.default_rng(0)
    M, cell = 22, 20
    grid = rng.random((M, M)) < 0.5
    img = np.where(np.kron(grid, np.ones((cell, cell))), 0, 255).astype(np.uint8)
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    p = _fft_pitch(img, cell * 0.6, cell * 1.5)
    assert p is not None and abs(p - cell) < 2.0   # within 2px of the true 20px pitch


def test_decode_auto_returns_none_on_blank():
    blank = np.full((400, 400), 255, np.uint8)
    got, params = decode_auto(blank)
    assert got is None and params is None


def test_decode_auto_uses_two_detectors():
    # the decode path must no longer call detect_dark_region
    import inspect
    from datamatrix_reader import register
    src = inspect.getsource(register.decode_auto)
    assert "detect_dark_region" not in src
    assert "detect_area" in src and "detect_data_region" in src


def test_recover_decodes_offcenter_scene():
    import random
    from datamatrix_reader import synth
    from datamatrix_reader.register import recover
    rng = random.Random(3)
    payload = _square_symbol()[0]
    # code in the lower-right (NOT the old upper-left ROI), with border defects
    p = synth.SceneParams(canvas=(900, 1100), cell=18, pos=(0.72, 0.68),
                          rotation_deg=180.0, defects=True, text=True, edges=True)
    img, truth = synth.scene(payload, p, rng)
    assert recover(img[..., 0])[0] == payload


def test_recover_decodes_damaged_scenes():
    import random
    from datamatrix_reader import synth
    from datamatrix_reader.register import recover
    rng = random.Random(7)
    payload = None
    for t in (b"DMTXSLIDE-GUIDED-TEST1", b"ABCDEFGHIJKLMNOPQRSTUVWX"):
        import zxingcpp
        a = np.asarray(zxingcpp.create_barcode(t.decode(),
              zxingcpp.BarcodeFormat.DataMatrix).to_image())
        if a.shape[0] == a.shape[1]:
            payload = t; break
    # several damaged scenes must still decode
    ok = 0
    for i in range(6):
        p = synth.SceneParams(canvas=(800, 900), cell=16 + i, pos=(0.4, 0.55),
                              rotation_deg=90.0 * (i % 4), defects=True, text=True)
        img, _ = synth.scene(payload, p, rng)
        if recover(img[..., 0])[0] == payload:
            ok += 1
    assert ok >= 5


def test_brute_region_order_preserves_recovery():
    # center-out ordering must not change WHICH codes decode — a damaged off-center
    # scene that decoded before must still decode.
    import random
    from datamatrix_reader import synth
    from datamatrix_reader.register import recover
    rng = random.Random(11)
    payload = _square_symbol()[0]
    p = synth.SceneParams(canvas=(850, 1000), cell=20, pos=(0.6, 0.4),
                          rotation_deg=270.0, skew_deg=6.0,
                          defects=True, text=True, edges=True)
    img, _ = synth.scene(payload, p, rng)
    assert recover(img[..., 0])[0] == payload


def test_recover_returns_quad_enclosing_code():
    import random
    from datamatrix_reader import synth
    from datamatrix_reader.register import recover
    rng = random.Random(5)
    payload = _square_symbol()[0]
    p = synth.SceneParams(canvas=(900, 1100), cell=18, pos=(0.6, 0.45),
                          rotation_deg=90.0, defects=True, text=True)
    img, truth = synth.scene(payload, p, rng)
    pl, quad = recover(img[..., 0])
    assert pl == payload
    assert quad is not None and quad.shape == (4, 2)
    # quad center should sit near the true code center (within ~1 code-size)
    cx, cy = quad[:, 0].mean(), quad[:, 1].mean()
    assert abs(cx - truth["cx"]) < truth["size"] and abs(cy - truth["cy"]) < truth["size"]
    # quad side should be roughly the code size (within 40%)
    import numpy as np
    side = np.linalg.norm(quad[0] - quad[1])
    assert 0.6 * truth["size"] < side < 1.6 * truth["size"]
