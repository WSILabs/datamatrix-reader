import cv2, numpy as np, zxingcpp
from dmtxslide import reader as R
from dmtxslide.reader import Reader

_DM = zxingcpp.BarcodeFormat.DataMatrix

def _encoded(payload, scale=8):
    grid = np.asarray(zxingcpp.create_barcode(payload, _DM).to_image())
    return cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def test_reads_clean_code_stage_raw():
    r = Reader().read(_encoded(b"S25-04821-A3"))
    assert r.payload == b"S25-04821-A3"
    assert r.stage == "raw"
    assert r.ok and r.elapsed_ms >= 0

def test_blank_image_is_not_ok():
    r = Reader().read(np.full((120, 120), 255, np.uint8))
    assert r.payload is None and r.stage is None and not r.ok

def test_accepts_bgr_and_gray():
    gray = _encoded(b"S25-04821-A3")
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert Reader().read(bgr).payload == b"S25-04821-A3"

def test_falls_back_to_clahe_stage(monkeypatch):
    calls = {"n": 0}
    def fake_zxing(gray):
        calls["n"] += 1
        return None if calls["n"] == 1 else b"RECOVERED"
    monkeypatch.setattr(R, "_zxing", fake_zxing)
    r = Reader().read(np.full((50, 50), 255, np.uint8))
    assert r.payload == b"RECOVERED" and r.stage == "clahe"
    assert calls["n"] == 2

def test_falls_back_through_thicken_stages(monkeypatch):
    # raw + clahe + first thicken miss; the 4th _zxing call hits -> 3rd stage name.
    seq = iter([None, None, None, b"P"])
    monkeypatch.setattr(R, "_zxing", lambda g: next(seq))
    r = Reader().read(np.full((60, 60), 255, np.uint8))
    assert r.payload == b"P" and r.stage == "thick_u2_i2"   # raw, clahe, u2_i1, u2_i2

def test_stage_transform_error_is_treated_as_miss(monkeypatch):
    # a stage transform that raises cv2.error must be skipped like a miss, not crash
    boom = [("clahe", lambda g: (_ for _ in ()).throw(cv2.error("x")))] + list(R.STAGES[1:])
    monkeypatch.setattr(R, "STAGES", boom)
    monkeypatch.setattr(R, "_zxing", lambda g: None)        # everything misses
    r = Reader().read(np.full((60, 60), 255, np.uint8))
    assert r.payload is None and r.stage is None
