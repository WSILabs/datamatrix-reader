"""Optional payload validators for callers that want format-level gating.

A decode is only "success" if it's *valid*: a misformatted decode counts as a
miss, not just "some bytes came back". The default accepts anything; supply a
real one for your accession format to measure correctness. These are an
application-layer concern — the zxing Reader does not consult them.
"""
from __future__ import annotations

import re
from typing import Protocol


class Validator(Protocol):
    def __call__(self, payload: bytes) -> bool: ...


class AcceptAny:
    def __call__(self, payload: bytes) -> bool:
        return len(payload) > 0


class RegexValidator:
    """e.g. RegexValidator(r'^[A-Z]\\d{2}-\\d{5}-[A-Z]\\d$') for an accession id."""

    def __init__(self, pattern: str, encoding: str = "ascii"):
        self._rx = re.compile(pattern)
        self._enc = encoding

    def __call__(self, payload: bytes) -> bool:
        try:
            return self._rx.match(payload.decode(self._enc)) is not None
        except UnicodeDecodeError:
            return False
