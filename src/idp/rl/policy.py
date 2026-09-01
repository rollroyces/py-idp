"""Policy update from aggregated rewards.

Given a `PolicyStats` (per-field failure rates) we update two things:

  1. `ClassifierConfidenceFloor` — fields with high failure rates get
     a higher confidence floor for HITL escalation (i.e., even if the
     model is "confident", we mark them for review when humans have
     corrected them often).

  2. `FieldPenalty` — fields with high failure rates get their
     `assess_confidence` heuristic base lowered, so they end up
     flagged for review more often.

We do NOT update the LLM itself. We're updating the *post-hoc* confidence
adjustment layer that decides what to flag for HITL. This is the highest-ROI
thing we can do with reward signals: turn "the model said it was sure" into
"we know the model is wrong here, send it to review".

Honest about what this is and isn't:
  IS: a deterministic, inspectable, version-controllable rule update
       from human feedback.
  ISN'T: a learned reward model, a fine-tuned LLM, or any neural update.
        Those would be orders-of-magnitude more engineering for ~2-3% F1 gain.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from idp.rl.reward import PolicyStats

log = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Inspectable policy parameters updated from reward signals."""

    # per-field: if observed failure rate (revisions / total) >= threshold,
    # the field is "high-failure" and gets escalation treatment.
    high_failure_threshold: float = 0.30
    # when a field is high-failure, its confidence floor for HITL rises.
    # e.g. 0.6 means "even if the model says 0.65, mark for review".
    high_failure_confidence_floor: float = 0.95
    # normal confidence floor for HITL (fields below this are flagged).
    base_confidence_floor: float = 0.6
    # when a field is high-failure, deduct this from the heuristic base.
    high_failure_penalty: float = 0.20
    # minimum total reviews per field before any floor/penalty kicks in.
    # below this threshold, fail-rate estimates have too much variance for
    # the binary floor ON/OFF rule. Set higher = more conservative.
    min_reviews: int = 10

    # per-field overrides (set by update_policy).
    field_floors: dict[str, float] = field(default_factory=dict)
    field_penalties: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PolicyConfig:
        d = json.loads(Path(path).read_text())
        return cls(**d)


def update_policy(stats: PolicyStats, current: PolicyConfig | None = None) -> PolicyConfig:
    """Compute a new PolicyConfig from reward statistics.

    Honest about what gets set:
      - `field_floors[field] = high_failure_confidence_floor` if the field's
        observed fail_rate >= high_failure_threshold AND the field has at
        least `min_reviews` total observations.
      - `field_penalties[field] = high_failure_penalty` for the same fields.

    Fields NOT in the high-failure set use the base config values
    (which means "no override").

    Conservative defaults: high_failure_threshold=0.30, min_reviews=10,
    so we only override behaviour for fields the user has corrected
    >=30% of the time across at least 10 observations.
    """
    current = current or PolicyConfig()
    new_floors = dict(current.field_floors)
    new_penalties = dict(current.field_penalties)

    for field_name, counts in stats.per_field.items():
        n_corrected = counts.get("+1", 0)
        n_total = n_corrected + counts.get("0", 0)
        if n_total == 0:
            continue
        # GUARD: small-sample fields get no override. With n=1..9,
        # a single correction puts fail_rate at 100% or 0%; below 10
        # observations the variance is too high to commit a floor.
        if n_total < current.min_reviews:
            continue
        fail_rate = n_corrected / n_total
        if fail_rate >= current.high_failure_threshold:
            new_floors[field_name] = current.high_failure_confidence_floor
            new_penalties[field_name] = current.high_failure_penalty
            log.info(
                "policy: %s high-failure (rate=%.2f, n=%d) -> floor=%.2f, penalty=%.2f",
                field_name, fail_rate, n_total,
                current.high_failure_confidence_floor, current.high_failure_penalty,
            )

    return PolicyConfig(
        high_failure_threshold=current.high_failure_threshold,
        high_failure_confidence_floor=current.high_failure_confidence_floor,
        base_confidence_floor=current.base_confidence_floor,
        high_failure_penalty=current.high_failure_penalty,
        min_reviews=current.min_reviews,
        field_floors=new_floors,
        field_penalties=new_penalties,
    )


def policy_to_review_flags(
    confidence: dict[str, float], policy: PolicyConfig
) -> dict[str, bool]:
    """Given a confidence dict + the current policy, return per-field HITL flags."""
    flags: dict[str, bool] = {}
    for field_name, conf in confidence.items():
        floor = policy.field_floors.get(field_name, policy.base_confidence_floor)
        flags[field_name] = conf < floor
    return flags


def policy_to_penalised_confidence(
    confidence: dict[str, float], policy: PolicyConfig
) -> dict[str, float]:
    """Apply per-field penalty from policy to a confidence dict.

    A high-failure field's confidence is reduced by its penalty before
    downstream stages (HITL UI) read it. Capped at [0, 1].
    """
    out: dict[str, float] = {}
    for field_name, conf in confidence.items():
        penalty = policy.field_penalties.get(field_name, 0.0)
        out[field_name] = max(0.0, min(1.0, conf - penalty))
    return out


# ---------------------------------------------------------------------------
# Backwards-compatible re-exports
# ---------------------------------------------------------------------------
# The "from-X-input" policy updaters live in ``idp.rl.update`` for code-
# organisation reasons, but historically users have imported them from
# ``idp.rl.policy`` too. Re-exporting here preserves that symmetry.
from idp.rl.update import (  # noqa: E402, F401
    update_policy_from_reviews_file,
    update_policy_from_sql,
    update_policy_from_storage,
)
