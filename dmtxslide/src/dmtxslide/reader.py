"""Public API: Reader.read(image, budget_ms).

Pipeline: best contrast channel -> localize once -> walk candidates, running
the cascade on each, all under one global time budget. First validated decode
across all candidates wins. Worst-case latency is bounded by budget_ms.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import adapt, localize
from .cascade import DEFAULT_LADDER, CascadeResult, Rung, run_cascade
from .validate import AcceptAny, Validator


@dataclass
class ReadResult:
    payload: bytes | None
    rung: str | None
    candidate_idx: int | None
    elapsed_ms: float
    candidate_traces: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.payload is not None


@dataclass
class Reader:
    validator: Validator = AcceptAny()
    ladder: list[Rung] = field(default_factory=lambda: list(DEFAULT_LADDER))
    rung_timeout_ms: int = 35

    def read(self, image: np.ndarray, budget_ms: float = 250.0) -> ReadResult:
        t0 = time.perf_counter()
        deadline = t0 + budget_ms / 1000.0

        gray = adapt.best_contrast_channel(image)
        candidates = localize.localize(gray)

        traces = []
        for idx, cand in enumerate(candidates):
            if time.perf_counter() >= deadline:
                break
            res: CascadeResult = run_cascade(
                cand.crop, validator=self.validator, ladder=self.ladder,
                rung_timeout_ms=self.rung_timeout_ms, deadline=deadline,
            )
            traces.append((cand.source, res))
            if res.ok:
                return ReadResult(res.payload, res.rung, idx,
                                  (time.perf_counter() - t0) * 1000, traces)

        return ReadResult(None, None, None,
                          (time.perf_counter() - t0) * 1000, traces)
