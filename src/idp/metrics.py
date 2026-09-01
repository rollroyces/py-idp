"""Lightweight in-process metrics + observability hooks.

Production deployments typically want:
  - request count, error count, latency histograms per stage
  - LLM-token tracking (for cost monitoring)
  - in-flight request gauge

This module provides a thread-safe ``Metrics`` singleton with simple
counters and gauges. It's deliberately small — no Prometheus client
dependency unless the user opts in. For Prometheus integration, see
``Metrics.export_prometheus()`` which returns a plain-text format string.

Usage:
    from idp.metrics import metrics

    @metrics.timed("extract")
    def extract(doc):
        ...
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any


class Metrics:
    """Thread-safe counters, gauges, and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._hist_max_samples = int(os.environ.get("IDP_METRICS_MAX_SAMPLES", "1000"))

    # ---- Counters ----
    def inc(self, name: str, value: int = 1, **labels: str) -> None:
        """Increment a counter. Labels are part of the metric name."""
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    # ---- Gauges ----
    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge value (current state, e.g. in-flight requests)."""
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value

    # ---- Histograms ----
    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a value in a histogram."""
        key = self._key(name, labels)
        with self._lock:
            samples = self._histograms[key]
            samples.append(value)
            # Cap samples to bound memory; keep most recent N.
            if len(samples) > self._hist_max_samples:
                # Cheap ring-buffer trim
                self._histograms[key] = samples[-self._hist_max_samples :]

    # ---- Timing decorator ----
    def timed(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that records call duration in ``{name}_duration_seconds``."""
        hist_name = f"{name}_duration_seconds"

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return fn(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    self.observe(hist_name, elapsed)
                    self.inc(f"{name}_calls")
            return wrapper

        return decorator

    # ---- Snapshot ----
    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all metrics."""
        with self._lock:
            histograms: dict[str, Any] = {
                name: self._hist_summary(samples)
                for name, samples in self._histograms.items()
            }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": histograms,
            }

    def reset(self) -> None:
        """Reset all metrics. For tests only."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    # ---- Export ----
    def export_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format.

        Histograms are summarised as ``{count, sum, min, max, mean}`` —
        not full bucket distributions (would need a Prometheus client lib).
        """
        with self._lock:
            lines: list[str] = []
            for name, count in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {count}")
            for name, gvalue in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {gvalue:.6f}")
            for name, samples in sorted(self._histograms.items()):
                summary = self._hist_summary(samples)
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {int(summary['count'])}")
                lines.append(f"{name}_sum {summary['sum']:.6f}")
            return "\n".join(lines)

    # ---- Internals ----
    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        # Prometheus-ish label syntax
        parts = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{parts}}}"

    @staticmethod
    def _hist_summary(samples: list[float]) -> dict[str, float]:
        if not samples:
            return {"count": 0, "sum": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "count": len(samples),
            "sum": sum(samples),
            "min": min(samples),
            "max": max(samples),
            "mean": sum(samples) / len(samples),
        }


# Module-level singleton. Importers can replace it for tests.
metrics = Metrics()


__all__ = ["Metrics", "metrics"]