"""Tests for benchmark harness realism."""
from bench.harness import _payload_pool
from dmtxslide import binding


def test_payload_pool_spans_multiple_symbol_sizes():
    """Real pathology codes range 12x12..24x24 modules (payload length drives
    symbol size). The benchmark must exercise that spread, not a single size."""
    sizes = {binding.encode(p, module_size=1, margin=2).shape for p in _payload_pool()}
    assert len(sizes) >= 3, f"payload pool yields only {len(sizes)} symbol size(s): {sizes}"
