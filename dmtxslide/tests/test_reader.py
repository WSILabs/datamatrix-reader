import cv2, numpy as np, zxingcpp
from dmtxslide import reader as R
from dmtxslide import register as REG
from dmtxslide.reader import Reader

_DM = zxingcpp.BarcodeFormat.DataMatrix

def _encoded(payload, scale=8):
    grid = np.asarray(zxingcpp.create_barcode(payload, _DM).to_image())
    return cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

def _fake_quad():
    return np.zeros((4, 2), np.float32)

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
    # raw zxing misses; the cascade 'clahe' stage hits -> stage reported as "clahe".
    monkeypatch.setattr(REG, "_collect",
                        lambda gray, first_only=False, fallback=True:
                            ([(b"RECOVERED", _fake_quad(), "DataMatrix", "clahe")], []))
    r = Reader().read(np.full((50, 50), 255, np.uint8))
    assert r.payload == b"RECOVERED" and r.stage == "clahe"

def test_falls_back_through_thicken_stages(monkeypatch):
    # raw + clahe + thick_u2_i1 miss; thick_u2_i2 hits -> stage is "thick_u2_i2".
    monkeypatch.setattr(REG, "_collect",
                        lambda gray, first_only=False, fallback=True:
                            ([(b"P", _fake_quad(), "DataMatrix", "thick_u2_i2")], []))
    r = Reader().read(np.full((60, 60), 255, np.uint8))
    assert r.payload == b"P" and r.stage == "thick_u2_i2"

def test_u4_gate_skips_oversampled_runs_small_and_blind():
    # The cascade's costly 4x stages run only for under-sampled (small/dense) regions or
    # when nothing was localized. Threshold is px/module = size/_EST_MODULES < _PXMOD_GATE.
    big = (REG._PXMOD_GATE + 2) * REG._EST_MODULES   # comfortably oversampled
    small = (REG._PXMOD_GATE - 2) * REG._EST_MODULES  # under-sampled
    assert REG._needs_u4([]) is True                  # blind safety net
    assert REG._needs_u4([big]) is False              # oversampled -> skip u4
    assert REG._needs_u4([small]) is True             # small -> run u4
    assert REG._needs_u4([big, small]) is True        # any small region triggers u4


def test_stage_transform_error_is_treated_as_miss(monkeypatch):
    # a stage transform that raises cv2.error must be skipped like a miss, not crash.
    # Verify via _collect: a STAGES list where clahe raises cv2.error, everything else
    # misses -> _collect returns nothing -> Reader.read returns no decode.
    from dmtxslide import preprocess as PP
    boom = [("clahe", lambda g: (_ for _ in ()).throw(cv2.error("x")))] + list(PP.STAGES[1:])
    # Inject the boom STAGES into _collect via a thin wrapper that shadows the import.
    import dmtxslide.preprocess as _PP_MOD
    monkeypatch.setattr(_PP_MOD, "STAGES", boom)
    # Also stub zxingcpp.read_barcodes in register to always miss (blank image -> no code).
    # The blank 60x60 white image has no code, so raw zxing misses naturally;
    # the cascade runs but the image has no code, so all stages produce no decode.
    r = Reader().read(np.full((60, 60), 255, np.uint8))
    assert r.payload is None and r.stage is None


def test_read_all_finds_multiple_datamatrix_and_qr_hint():
    import numpy as np, cv2, zxingcpp
    from dmtxslide.reader import Reader
    _DM = zxingcpp.BarcodeFormat.DataMatrix

    def tile(payload, fmt, scale=12):
        a = np.asarray(zxingcpp.create_barcode(payload, fmt).to_image())
        return cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    t1 = tile(b"1-S-25-00001 A1", _DM)
    t2 = tile(b"1-S-25-00002 B2", _DM)
    qr = tile("https://lis.example/9", zxingcpp.BarcodeFormat.QRCode, 10)
    # canvas tall enough to fit the QR at row 420 (420 + 330 = 750 < 800)
    canvas = np.full((800, 1100), 245, np.uint8)
    canvas[120:120 + t1.shape[0], 120:120 + t1.shape[1]] = t1
    canvas[120:120 + t2.shape[0], 700:700 + t2.shape[1]] = t2
    canvas[420:420 + qr.shape[0], 420:420 + qr.shape[1]] = qr

    res = Reader().read_all(canvas)
    payloads = set(res.payloads)
    assert b"1-S-25-00001 A1" in payloads and b"1-S-25-00002 B2" in payloads
    assert all(c.format == "DataMatrix" and c.quad.shape == (4, 2) for c in res.datamatrix)
    # the QR is reported as a non-DM hint, NOT among the DataMatrix
    assert any(c.format == "QRCode" for c in res.other_2d)
    assert all(c.format != "QRCode" for c in res.datamatrix)
