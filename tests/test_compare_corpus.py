import csv, cv2, numpy as np
from tools.compare_backends import load_corpus

def _write_png(p):
    cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))

def test_load_corpus_flat_dir_with_labels(tmp_path):
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "b.png")
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "payload"])
        w.writerow(["a.png", "111"]); w.writerow(["b.png", "222"])
    items = load_corpus(tmp_path)            # no images/ subdir -> flat fallback
    truths = sorted(t for t, _ in items)
    assert truths == [b"111", b"222"]
    assert all(img is not None for _, img in items)

def test_load_corpus_skips_unlabeled_and_nonimages(tmp_path):
    _write_png(tmp_path / "a.png")
    _write_png(tmp_path / "c.png")           # no label row -> skipped
    with (tmp_path / "labels.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "payload"]); w.writerow(["a.png", "111"])
    items = load_corpus(tmp_path)
    assert [t for t, _ in items] == [b"111"]
