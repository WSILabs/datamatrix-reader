"""Tests for benchmark harness realism."""
import random

import numpy as np

from bench.harness import _payload_pool, run
from datamatrix_reader.synth import DegradeParams, degrade, render


def test_payload_pool_spans_multiple_symbol_sizes():
    """Real pathology codes range 12x12..24x24 modules (payload length drives
    symbol size). The benchmark must exercise that spread, not a single size."""
    sizes = {render(p).shape for p in _payload_pool()}
    assert len(sizes) >= 3, f"payload pool yields only {len(sizes)} symbol size(s): {sizes}"


def test_run_dumps_only_failed_images(tmp_path):
    """A failing synthetic cell must be saved for inspection (so a reader miss
    can be told apart from an impossible/buggy sample); passing cells must not
    clutter the dump."""
    payload = b"S25-04821-A3"
    good = degrade(render(payload), DegradeParams(module_px=8.0), random.Random(0))
    blank = np.full((120, 120, 3), 255, np.uint8)  # nothing to find -> guaranteed miss
    items = [({"case": "good"}, payload, good),
             ({"case": "blank"}, b"NOPE", blank)]

    run(items, budget_ms=200, dump_failures=str(tmp_path))

    dumped = sorted(p.name for p in tmp_path.glob("*.png"))
    assert len(dumped) == 1, f"expected only the failure dumped, got {dumped}"
    assert "blank" in dumped[0], f"dumped file should name the failed stratum: {dumped[0]}"
