import csv

import cv2
import numpy as np
import zxingcpp

from tools.defect_classify import (
    DEFECTS, DEFECT_CODES, Row, stage_of, raw_zxing_misses,
    auto_bbox, auto_metrics, guess_defects, save_rows, load_rows,
)

_DM = zxingcpp.BarcodeFormat.DataMatrix


def _code_img(payload, scale=8):
    grid = np.asarray(zxingcpp.create_barcode(payload, _DM).to_image())
    return cv2.cvtColor(cv2.resize(grid, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)


def test_defect_taxonomy_unique_and_excludes_size():
    codes = [c for c, _ in DEFECTS]
    assert len(codes) == len(set(codes))            # no dup codes
    assert "size" not in DEFECT_CODES               # size deliberately dropped
    assert {"glare", "broken_finder", "no_code", "rotated"} <= DEFECT_CODES


def test_csv_round_trip_with_bbox_and_defects(tmp_path):
    rows = [
        Row("b.png", (10, 20, 30, 40), ["glare", "blur"], "clahe", "note,with,comma"),
        Row("a.png", None, [], "none", ""),
    ]
    p = tmp_path / "defects.csv"
    save_rows(p, rows)
    back = {r.file: r for r in load_rows(p)}
    assert back["b.png"].bbox == (10, 20, 30, 40)
    assert back["b.png"].defects == ["glare", "blur"]
    assert back["b.png"].notes == "note,with,comma"     # csv quoting survives
    assert back["a.png"].bbox is None and back["a.png"].defects == []
    # sorted by filename on save
    assert [r["file"] for r in csv.DictReader(p.open())] == ["a.png", "b.png"]


def test_stage_of_classifies_raw_and_none():
    assert stage_of(_code_img(b"S25-04821-A3"), b"S25-04821-A3") == "raw"
    blank = np.full((120, 120, 3), 255, np.uint8)
    assert stage_of(blank, b"WHATEVER") == "none"


def test_raw_zxing_misses_excludes_readable_includes_unreadable(tmp_path):
    cv2.imwrite(str(tmp_path / "easy.png"), _code_img(b"S25-04821-A3"))
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((120, 120, 3), 255, np.uint8))
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "payload"])
        w.writerow(["easy.png", "S25-04821-A3"]); w.writerow(["blank.png", "NOPE"])
    rows = raw_zxing_misses(tmp_path)
    assert [r.file for r in rows] == ["blank.png"]       # easy.png read by raw -> excluded
    assert rows[0].recovered == "none"                   # blank never decodes


def test_auto_metrics_keys_and_guess_defects():
    m = auto_metrics(np.full((50, 50), 128, np.uint8))
    assert set(m) == {"blur", "contrast", "illum", "bright_frac"}
    # a flat low-contrast patch -> low_contrast guessed; a glary one -> glare
    assert "low_contrast" in guess_defects({"blur": 200, "contrast": 5,
                                            "illum": 0, "bright_frac": 0})
    assert "glare" in guess_defects({"blur": 200, "contrast": 80,
                                     "illum": 0, "bright_frac": 0.2})


def test_auto_bbox_none_on_blank_and_box_on_pasted_code():
    blank = np.full((400, 400), 255, np.uint8)
    assert auto_bbox(blank) is None
    canvas = np.full((400, 400), 255, np.uint8)
    code = np.asarray(zxingcpp.create_barcode(b"S25-04821-A3", _DM).to_image())
    code = cv2.resize(code, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    ch, cw = code.shape
    canvas[60:60 + ch, 90:90 + cw] = code              # paste at a known spot
    box = auto_bbox(canvas)
    assert box is not None
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2                       # detected center near the code center
    assert 90 < cx < 90 + cw + 30 and 60 < cy < 60 + ch + 30
