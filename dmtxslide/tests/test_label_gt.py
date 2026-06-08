from tools.label_gt import decide

def test_all_agree_is_auto():
    assert decide({"a": b"X", "b": b"X", "c": b"X"}) == ("auto", [b"X"])

def test_sole_reader_is_auto():
    assert decide({"a": b"X", "b": None, "c": None}) == ("auto", [b"X"])

def test_no_read_is_queue_empty():
    assert decide({"a": None, "b": None, "c": None}) == ("queue", [])

def test_disagreement_is_queue_with_sorted_candidates():
    assert decide({"a": b"Y", "b": b"X", "c": None}) == ("queue", [b"X", b"Y"])
