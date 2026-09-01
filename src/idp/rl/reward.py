"""Reward signals derived from human reviews.

A review is the diff between the model's extraction and the human-edited
extraction. We turn that diff into a per-field reward in {-1, 0, +1}:

  +1  human changed a field away from the model's value
       -> model was wrong, increase weight on this field
   0  human accepted the model's value
       -> no signal
  -1  human changed the model's value to something else
       -> we don't know if model was wrong or human was wrong;
          conservatively treat as "model was right, don't decrease"

For now we collapse all three states into the simple +1/0/-1 contract
above. A future iteration could add a second signal for "human spent
>5 minutes on this doc" -> weight the disagreement more.

This is intentionally NOT trying to learn a reward model from text.
We're learning a *policy* (per-field confidence thresholds, classifier
fallback rules) from the simplest possible signal: did the human agree.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from idp.storage.store import StoredResult

log = logging.getLogger(__name__)


@dataclass
class FieldReward:
    """One field's reward signal from one review."""

    field: str
    reward: int               # -1, 0, or +1
    model_value: Any
    human_value: Any


@dataclass
class ReviewRewards:
    """All field rewards from one human review."""

    doc_id: str
    schema_name: str
    field_rewards: list[FieldReward] = field(default_factory=list)

    @property
    def total_reward(self) -> int:
        return sum(r.reward for r in self.field_rewards)

    @property
    def n_corrected(self) -> int:
        return sum(1 for r in self.field_rewards if r.reward == +1)

    @property
    def n_accepted(self) -> int:
        return sum(1 for r in self.field_rewards if r.reward == 0)


def _norm_for_compare(v: Any) -> Any:
    """Normalize for equality comparison (mirrors eval/metrics._norm)."""
    if v is None or v == "" or v == [] or v == {}:
        return None
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, dict):
        return {k: _norm_for_compare(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_norm_for_compare(x) for x in v]
    return v


def derive_field_rewards(stored: StoredResult) -> ReviewRewards | None:
    """Compute per-field rewards for one stored result that has been reviewed.

    Returns None if the stored result has not been reviewed.
    """
    if not stored.reviewed or stored.reviewed_extraction is None:
        return None

    rewards = ReviewRewards(doc_id=stored.doc_id, schema_name=stored.schema_name)
    model = stored.extraction or {}
    human = stored.reviewed_extraction or {}

    # union of all keys touched
    keys = set(model.keys()) | set(human.keys())
    for k in keys:
        m_val = _norm_for_compare(model.get(k))
        h_val = _norm_for_compare(human.get(k))
        if m_val == h_val:
            rewards.field_rewards.append(FieldReward(k, 0, model.get(k), human.get(k)))
        else:
            # human changed model output -> reward +1 (model was wrong)
            rewards.field_rewards.append(FieldReward(k, +1, model.get(k), human.get(k)))
    return rewards


@dataclass
class PolicyStats:
    """Per-field aggregated rewards across many reviews."""

    n_reviews: int = 0
    n_fields: int = 0
    per_field: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "n_reviews": self.n_reviews,
            "n_fields": self.n_fields,
            "per_field": self.per_field,
        }
        # also surface the highest-failure fields
        fail_rates = []
        for f, counts in self.per_field.items():
            total = counts.get("+1", 0) + counts.get("0", 0)
            if total > 0:
                fail_rates.append((f, counts.get("+1", 0) / total))
        fail_rates.sort(key=lambda kv: -kv[1])
        out["top_failure_fields"] = [
            {"field": f, "fail_rate": round(rate, 3), "n": self.per_field[f].get("+1", 0)}
            for f, rate in fail_rates[:5]
        ]
        return out


def aggregate_rewards(reviews: list[ReviewRewards]) -> PolicyStats:
    """Aggregate a list of per-review rewards into per-field counts."""
    stats = PolicyStats(n_reviews=len(reviews))
    per_field: dict[str, Counter] = {}
    for r in reviews:
        for fr in r.field_rewards:
            # use signed for non-zero (so they sort cleanly), plain "0" for zero
            key = f"{fr.reward:+d}" if fr.reward != 0 else "0"
            per_field.setdefault(fr.field, Counter())[key] += 1
    stats.per_field = {f: dict(c) for f, c in per_field.items()}
    stats.n_fields = len(per_field)
    return stats
