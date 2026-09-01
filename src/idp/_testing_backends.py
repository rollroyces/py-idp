"""Testing-only LLM backends.

This module is intentionally NOT part of the public ``idp`` namespace.
Its backends are gated by env vars (``IDP_ENABLE_SLOWMOCK=1``) so
they cannot be invoked in production by accident.
"""
from __future__ import annotations

import random
import time
from typing import Any

from idp.llm.backend import CompletionRequest, MockBackend


class SlowMockBackend(MockBackend):
    """A MockBackend that sleeps to simulate real LLM latency.

    Configured by ``LOAD_LATENCY_MS`` (default 1500) +/- ``LOAD_JITTER_MS``
    (default 500). Only enabled when ``IDP_ENABLE_SLOWMOCK=1``.
    """

    name = "slowmock"

    def __init__(self, latency_ms: int = 1500, jitter_ms: int = 500, **kwargs: Any):
        super().__init__(**kwargs)
        self.latency_ms = latency_ms
        self.jitter_ms = jitter_ms

    def complete(self, req: CompletionRequest) -> str:
        sleep_s = (self.latency_ms + random.uniform(-self.jitter_ms, self.jitter_ms)) / 1000.0
        if sleep_s > 0:
            time.sleep(sleep_s)
        return super().complete(req)


__all__ = ["SlowMockBackend"]