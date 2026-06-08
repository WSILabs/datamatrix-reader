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
    save_labels(p, {"b.png": "2", "a.png": "1"})
    assert p.read_text().splitlines() == ["file,payload", "a.png,1", "b.png,2"]
    assert load_labels(p) == {"a.png": "1", "b.png": "2"}

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
