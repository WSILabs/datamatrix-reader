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
