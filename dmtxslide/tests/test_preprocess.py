import numpy as np

from dmtxslide import preprocess as pp


def test_stages_named_and_ordered():
    names = [name for name, _ in pp.STAGES]
    assert names == ["clahe",
                     "thick_u2_i1", "thick_u2_i2", "thick_u2_i3",
                     "thick_u4_i1", "thick_u4_i2", "thick_u4_i3",
                     "sauv"]


def test_each_stage_returns_2d_uint8():
    g = np.full((40, 40), 128, np.uint8)
    for name, fn in pp.STAGES:
        out = fn(g)
        assert out.ndim == 2 and out.dtype == np.uint8, name


def test_thicken_grows_dark_region():
    b = np.full((20, 20), 255, np.uint8)
    b[8:12, 8:12] = 0                       # small ink square
    out = pp._thicken(b, 1)
    assert (out == 0).sum() > (b == 0).sum()
    # more iterations -> more ink
    assert (pp._thicken(b, 2) == 0).sum() > (out == 0).sum()


def test_thicken_stages_binarize_to_two_levels():
    g = np.random.default_rng(0).integers(0, 255, (40, 40)).astype(np.uint8)
    for name, fn in pp.STAGES:
        if name == "clahe":
            continue                        # clahe is grayscale, not binarized
        assert set(np.unique(fn(g))).issubset({0, 255}), name
