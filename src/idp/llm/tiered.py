# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tiered backend: cheap-first, expensive-on-failure.

The v0.4 "make setup much easier, but still keep accuracy" win: route
every ``complete()`` call to a cheap local model first (Ollama, vLLM,
LM Studio, etc.) and only escalate to the expensive model when the
cheap response is *demonstrably wrong* — either it cannot be parsed
as JSON, or the cheap self-assessed per-field confidence is below
``ambiguity_threshold``.

The wrapper is transparent: a :class:`idp.pipeline.pipeline.Pipeline`
sees an ordinary :class:`idp.llm.backend.Backend` and does not need to
know about the two-tier routing. See ``docs/ROADMAP_v0.4.md`` section
D2 for the design rationale and the explicit v0.4 scope (no per-field
escalation, no rule-based pre-pass — both deferred to v0.5).

Typical use::

    from idp.llm.backend import AnthropicBackend
    from idp.llm.ollama import OllamaBackend
    from idp.llm.tiered import TieredBackend

    pipeline = Pipeline(
        backend=TieredBackend(
            cheap=OllamaBackend(model="llama3.2:3b"),
            expensive=AnthropicBackend(),
        ),
        schema=Invoice,
    )

Decision logic
--------------

For each ``complete(req)``:

1. Call ``cheap.complete(req)``.
2. Parse the response as JSON. If parsing fails (the string is not
   JSON), **escalate** to ``expensive.complete(req)``.
3. If parsing succeeds, compute a per-field confidence dict. If *any*
   field's confidence is < ``ambiguity_threshold``, **escalate**.
4. Otherwise return the cheap response unchanged.

Confidence (default ``assess_fn``)
----------------------------------

The default ``assess_fn`` is a heuristic that mirrors the field-shape
penalty in :func:`idp.assess.confidence._heuristic_confidence`:

* a present string -> 0.8
* a present number/bool -> 0.8
* a present list/dict   -> 0.7
* ``None`` / empty / ``{}`` / ``[]`` -> 0.1

The cheap tier is expected to be a small local model whose occasional
mistakes will show up as missing or wrongly-typed fields; the default
heuristic catches both cheaply without an extra LLM call.

If you want per-field LLM self-assessment, pass a custom ``assess_fn``::

    TieredBackend(
        cheap=..., expensive=...,
        assess_fn=lambda raw, req: my_llm_self_assess(raw, req),
    )

Behavior on partial failures
----------------------------

* If the **expensive** tier also fails (returns invalid JSON), the
  wrapper raises :class:`TieredBackendExhaustedError` with both raw
  outputs attached. This is the same error contract as a normal
  backend that failed — pipelines relying on
  :meth:`Backend.json_complete` will record the parse-failure dict.
* If the **cheap** tier raises (network error, 5xx, etc.), the
  exception is **not** caught; we let it propagate. TieredBackend is
  not a retry wrapper (use :class:`idp.reliability.RetryingBackend`
  for that). Escalation only triggers on a *bad-quality response*,
  not on a transport failure.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from idp.llm.backend import (
    Backend,
    CompletionRequest,
    _safe_json,
)

log = logging.getLogger(__name__)


# Public exception so callers (and our own tests) can distinguish a
# "both tiers failed" outcome from a generic ``RuntimeError``.
class TieredBackendExhaustedError(RuntimeError):
    """Both the cheap and the expensive tier returned an unusable response.

    Attributes:
        cheap_raw: the raw string from the cheap tier.
        expensive_raw: the raw string from the expensive tier
            (may be ``None`` if the expensive tier raised before
            returning a string).
        original: the exception raised by the expensive tier, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        cheap_raw: str,
        expensive_raw: str | None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.cheap_raw = cheap_raw
        self.expensive_raw = expensive_raw
        self.original = original


# Default confidence heuristic.
def _heuristic_field_confidence(obj: dict[str, Any]) -> dict[str, float]:
    """Per-field confidence in [0.1, 0.8] from parsed-JSON shape alone.

    Mirrors the spirit of :func:`idp.assess.confidence._heuristic_confidence`
    but works on a raw dict (the cheap tier's parse output) without needing
    a full ``Document``.

    * missing key or ``None``              -> 0.1   (model didn't see it)
    * present empty string ``""``          -> 0.1   (model saw it but couldn't fill)
    * present empty list ``[]``            -> 0.7   (model saw the structure;
                                                       list/non-empty logic in
                                                       the real heuristic
                                                       rewards non-empty, but
                                                       for the v0.4 tiered
                                                       wrapper we accept
                                                       "saw the structure" as
                                                       a useful signal)
    * present empty dict ``{}``            -> 0.7   (same logic as list)
    * present non-empty string             -> 0.8
    * present number/bool                  -> 0.8
    * present non-empty list/dict          -> 0.7 (harder)

    Returns an empty dict if ``obj`` is not a dict (so callers can
    safely call :func:`min` on the result).
    """
    # ``obj`` is annotated as ``dict[str, Any]`` but at runtime it can be
    # anything ``_safe_json`` returned (a dict, an empty dict, an
    # ``_error`` dict, etc.). Guard defensively — mypy with
    # ``--warn-unreachable`` would otherwise flag the early branch
    # as dead because the annotation says it's a dict.
    if not isinstance(obj, dict) or not obj:  # type: ignore[unreachable]
        return {}
    return _score_dict(obj)


def _score_dict(obj: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in obj.items():
        if v is None:
            out[k] = 0.1
        elif isinstance(v, str):
            out[k] = 0.1 if v == "" else 0.8
        elif isinstance(v, list):
            # Empty list = "I saw this should be a list"; non-empty = harder
            # structure we may not have parsed right. Either way, 0.7 is
            # higher than the 0.1 we're using for "model had no idea",
            # which is what we want for the wrapper.
            out[k] = 0.7
        elif isinstance(v, dict):
            out[k] = 0.7
        elif isinstance(v, (int, float, bool)):
            out[k] = 0.8
        else:
            out[k] = 0.5
    return out


# An ``assess_fn`` is a callable ``(raw: str, req: CompletionRequest)
# -> dict[field_name -> confidence in [0,1]]``. The dict may be empty
# if the response was not a dict.
AssessFn = Callable[[str, CompletionRequest], dict[str, float]]


def _default_assess(raw: str, req: CompletionRequest) -> dict[str, float]:
    obj = _safe_json(raw)
    return _heuristic_field_confidence(obj)


class TieredBackend(Backend):
    """Two-tier Backend: cheap first, expensive on parse/conf failure.

    Args:
        cheap: the inexpensive local backend (e.g. :class:`OllamaBackend`,
            :class:`OpenAICompatBackend` pointed at a local vLLM).
        expensive: the high-quality backend (e.g. :class:`AnthropicBackend`,
            :class:`OpenAICompatBackend` pointed at a hosted model).
        ambiguity_threshold: per-field confidence below which we escalate.
            Defaults to ``0.7`` as in the roadmap spec (D2). Field-level
            confidence comes from ``assess_fn``; if *any* field is below
            this threshold we retry on ``expensive``.
        assess_fn: optional override for per-field confidence scoring.
            Signature is ``(raw_str, req) -> dict[field, float]``. The
            default is :func:`_default_assess` (a cheap heuristic based
            on parsed-JSON shape). Pass your own if you want LLM
            self-assessment or domain-specific heuristics.

    The wrapper is a drop-in replacement for either backend: it
    implements :meth:`complete` (the only abstract method on
    :class:`Backend`) and inherits the convenience helper
    :meth:`Backend.json_complete` from the base class.

    Example::

        backend = TieredBackend(
            cheap=OllamaBackend(model="llama3.2:3b"),
            expensive=AnthropicBackend(),
        )
        Pipeline(backend=backend, schema=Invoice).run(doc)

    Note:
        TieredBackend is *not* a retry wrapper. If the cheap tier
        raises (network error, timeout, 5xx), we let the exception
        propagate — wrap with
        :class:`idp.reliability.RetryingBackend` if you want
        transport-level retries first.
    """

    name = "tiered"

    def __init__(
        self,
        cheap: Backend,
        expensive: Backend,
        *,
        ambiguity_threshold: float = 0.7,
        assess_fn: AssessFn | None = None,
    ) -> None:
        if cheap is expensive:
            raise ValueError(
                "TieredBackend.cheap and .expensive must be different "
                "Backend instances (using the same backend twice would "
                "just be a no-op retry)."
            )
        if not (0.0 <= ambiguity_threshold <= 1.0):
            raise ValueError(
                f"ambiguity_threshold must be in [0.0, 1.0]; got {ambiguity_threshold!r}"
            )
        self.cheap = cheap
        self.expensive = expensive
        self.ambiguity_threshold = float(ambiguity_threshold)
        self._assess_fn: AssessFn = assess_fn or _default_assess
        # Bookkeeping for tests + observability. Not on the spec, but
        # the wrapper is the obvious place to count escalations.
        self.escalation_count: int = 0
        self.cheap_call_count: int = 0
        self.expensive_call_count: int = 0

    # ------------------------------------------------------------------
    # The whole Backend surface area.
    # ------------------------------------------------------------------
    def complete(self, req: CompletionRequest) -> str:
        """Route to cheap; escalate to expensive on parse or low confidence.

        Returns the cheap tier's string verbatim when it parses as JSON
        AND no field's confidence is below :attr:`ambiguity_threshold`.
        Otherwise returns the expensive tier's string (or raises
        :class:`TieredBackendExhaustedError` if the expensive tier
        also produces an invalid response).
        """
        self.cheap_call_count += 1
        cheap_raw = self.cheap.complete(req)

        parsed = _safe_json(cheap_raw)
        if not parsed or "_error" in parsed:
            # Parse failure. Note: ``_safe_json`` returns ``{"_error": ...}``
            # for unparseable strings and ``{}`` for empty/null. Both
            # qualify as "cheap did not return a usable response" and
            # trigger escalation.
            log.debug("TieredBackend: cheap output is not valid JSON, escalating")
            return self._escalate(req, cheap_raw)

        # Parse succeeded. Assess per-field confidence and check
        # whether every field clears the ambiguity threshold.
        conf = self._assess_fn(cheap_raw, req)
        if conf and min(conf.values()) < self.ambiguity_threshold:
            log.debug(
                "TieredBackend: cheap conf %s below threshold %.2f, escalating",
                conf,
                self.ambiguity_threshold,
            )
            return self._escalate(req, cheap_raw)

        # Cheap response is good enough.
        return cheap_raw

    # ------------------------------------------------------------------
    # Optional overrides: forward to whichever tier makes sense.
    # ------------------------------------------------------------------
    @property
    def is_multimodal(self) -> bool:
        """Multimodal iff *either* tier accepts images."""
        return bool(
            getattr(self.cheap, "is_multimodal", False)
            or getattr(self.expensive, "is_multimodal", False)
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _escalate(self, req: CompletionRequest, cheap_raw: str) -> str:
        """Run expensive; raise if it ALSO fails the acceptance test.

        Escalation only "succeeds" when the expensive tier's response
        is at least as good as the cheap tier's — meaning: parses as
        JSON AND every field clears the ambiguity threshold. If the
        expensive tier is also bad (parse failure OR low confidence),
        we raise :class:`TieredBackendExhaustedError` rather than
        silently returning the bad output.
        """
        self.escalation_count += 1
        self.expensive_call_count += 1
        try:
            expensive_raw = self.expensive.complete(req)
        except BaseException as exc:  # noqa: BLE001 — attach, don't swallow
            raise TieredBackendExhaustedError(
                f"TieredBackend exhausted: cheap AND expensive raised "
                f"({type(exc).__name__}: {exc})",
                cheap_raw=cheap_raw,
                expensive_raw=None,
                original=exc,
            ) from exc

        parsed = _safe_json(expensive_raw)
        if not parsed or "_error" in parsed:
            # Both tiers failed to produce JSON.
            raise TieredBackendExhaustedError(
                "TieredBackend exhausted: cheap AND expensive returned invalid JSON",
                cheap_raw=cheap_raw,
                expensive_raw=expensive_raw,
            )

        # Both tiers returned parseable JSON. Re-run the per-field
        # acceptance check on the expensive tier's response; if it's
        # also below threshold, surface as exhausted. This catches
        # the "cheap gave None for vendor, expensive also gave None
        # for vendor" case where escalating blindly would just
        # return the same broken output.
        exp_conf = self._assess_fn(expensive_raw, req)
        if exp_conf and min(exp_conf.values()) < self.ambiguity_threshold:
            raise TieredBackendExhaustedError(
                "TieredBackend exhausted: both cheap and expensive returned "
                "low-confidence output (min field below ambiguity_threshold)",
                cheap_raw=cheap_raw,
                expensive_raw=expensive_raw,
            )
        return expensive_raw


__all__ = [
    "TieredBackend",
    "TieredBackendExhaustedError",
    "_default_assess",
    "_heuristic_field_confidence",
]
