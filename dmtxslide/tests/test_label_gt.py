from tools.label_gt import decide

def test_all_agree_is_auto():
    assert decide({"a": b"X", "b": b"X", "c": b"X"}) == ("auto", [b"X"])

def test_sole_reader_is_auto():
    assert decide({"a": b"X", "b": None, "c": None}) == ("auto", [b"X"])

def test_no_read_is_queue_empty():
    assert decide({"a": None, "b": None, "c": None}) == ("queue", [])

def test_disagreement_is_queue_with_sorted_candidates():
    assert decide({"a": b"Y", "b": b"X", "c": None}) == ("queue", [b"X", b"Y"])


from tools.label_gt import load_labels, save_labels, payload_to_text

def test_payload_to_text_ascii_and_fallback():
    assert payload_to_text(b"1-S-24-34325 G2-1") == "1-S-24-34325 G2-1"
    assert payload_to_text(b"\xff") == "\xff"  # latin-1 fallback, no crash

def test_labels_roundtrip_and_sorted(tmp_path):
    p = tmp_path / "labels.csv"
    save_labels(p, {"b.png": "2", "a.png": "1", "c.png": "RACK,A1"})
    assert p.read_text().splitlines() == ["file,payload", "a.png,1", "b.png,2", 'c.png,"RACK,A1"']
    assert load_labels(p) == {"a.png": "1", "b.png": "2", "c.png": "RACK,A1"}
    assert load_labels(p)["c.png"] == "RACK,A1"

def test_load_missing_file_is_empty(tmp_path):
    assert load_labels(tmp_path / "nope.csv") == {}


from tools.label_gt import pending_images, delete_image

def _touch(d, name):
    p = d / name; p.write_bytes(b"x"); return p

def test_pending_excludes_labeled_and_nonimages(tmp_path):
    _touch(tmp_path, "a.png"); _touch(tmp_path, "b.png")
    _touch(tmp_path, "labels.csv")  # non-image, must be ignored
    pend = pending_images(tmp_path, {"a.png": "1"})
    assert [p.name for p in pend] == ["b.png"]

def test_delete_moves_file_and_drops_label(tmp_path):
    img = _touch(tmp_path, "junk.png")
    removed = tmp_path / "removed"
    labels = {"junk.png": "stale"}
    csv_path = tmp_path / "labels.csv"
    delete_image(img, removed, labels, csv_path)
    assert not img.exists()
    assert (removed / "junk.png").exists()
    assert "junk.png" not in labels
    assert load_labels(csv_path) == {}
    assert pending_images(tmp_path, labels) == []


import cv2, numpy as np
from tools.label_gt import autofill

def _png(d, name):
    cv2.imwrite(str(d / name), np.zeros((8, 8, 3), np.uint8)); return d / name

def test_autofill_writes_consensus_and_queues_rest(tmp_path):
    for n in ("agree.png", "disagree.png", "noread.png"):
        _png(tmp_path, n)
    def f1(img, b): return b"P"      # fires on all
    def f2(img, b): return None
    folds = [("f1", f1), ("f2", f2)]
    labels = {}
    res = autofill(tmp_path, labels, budget=50, folds=folds)
    # f1 is a sole reader on every image -> all become consensus auto-fills
    assert labels == {"agree.png": "P", "disagree.png": "P", "noread.png": "P"}
    assert res["added"] == 3 and res["queue"] == []
    assert load_labels(tmp_path / "labels.csv")["agree.png"] == "P"

def test_autofill_queues_disagreement(tmp_path):
    _png(tmp_path, "x.png")
    folds = [("f1", lambda i, b: b"A"), ("f2", lambda i, b: b"B")]
    labels = {}
    res = autofill(tmp_path, labels, budget=50, folds=folds)
    assert labels == {}                       # not auto-filled
    assert [(p.name, c) for p, c in res["queue"]] == [("x.png", ["A", "B"])]
