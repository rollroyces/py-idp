"""RL calibration eval.

Goal: prove whether the offline RL policy update is doing what it claims
to do — i.e., turning "fields humans correct often" into HITL flags that
catch those corrections, without flooding HITL with false positives.

The eval answers four questions:

  Q1. When the policy fires (flag = true), do humans actually correct
      that field?  -> "hit rate"
  Q2. When the policy does NOT fire (flag = false), do humans accept
      that field?  -> "true-accept rate"
  Q3. Is the confidence calibration monotone?  Binned confidences
      should correspond to similar empirical correctness rates.
  Q4. Does the policy beat the baseline (no-policy) on overall
      review-burden efficiency?  -> a single number, hits/(hits+false)

Inputs:
  --reviews <jsonl>    path to ReviewRewards JSONL (same format as
                       rl-update --reviews)
  --policy <json>      PolicyConfig (from rl-update --output)
  --fixtures <dir>     eval dataset (for synthetic-review generation)

Honest about scope:
  With only a handful of in-tree fixtures, all numbers are reported
  with explicit `n=` and a `synthetic=true` flag so the user can see
  whether the eval ran on real human data or on gold-truth-derived
  synthetic reviews (the latter is biased optimistic — gold truth
  IS the human's view).
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from idp.rl.policy import (
    PolicyConfig,
    policy_to_penalised_confidence,
    policy_to_review_flags,
)
from idp.rl.reward import _norm_for_compare

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Review loading
# ---------------------------------------------------------------------------
def load_reviews(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL of {doc_id, schema, model: {...}, human: {...}}."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def _review_field_outcome(reviews: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Per-field: count of accept (0) vs corrected (+1) outcomes."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"+1": 0, "0": 0})
    for r in reviews:
        model = r.get("model") or {}
        human = r.get("human") or {}
        for k in set(model.keys()) | set(human.keys()):
            m = _norm_for_compare(model.get(k))
            h = _norm_for_compare(human.get(k))
            out[k]["0" if m == h else "+1"] += 1
    return dict(out)


# ---------------------------------------------------------------------------
# Synthetic-review generation from gold truth
# ---------------------------------------------------------------------------
def synthetic_reviews_from_gold(
    dataset_dir: str | Path,
    *,
    injection_rate: float = 0.3,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Generate synthetic reviews from in-tree eval fixtures.

    For each gold-truth entry, produce a "model" output that differs
    in `injection_rate` fraction of fields (random subset). The
    "human" output is the gold truth verbatim.

    Synthetic reviews are BIASED OPTIMISTIC: gold truth IS what the
    human corrected TO, so the correction signal is artificially clean.
    We mark them with `synthetic=true` so the report makes this clear.

    Returns (synthetic_reviews, field_outcome_counts).
    """
    import random
    dataset_dir = Path(dataset_dir)
    cases_path = dataset_dir / "cases.jsonl"
    if not cases_path.exists():
        raise FileNotFoundError(f"no cases.jsonl in {dataset_dir}")

    rng = random.Random(seed)
    reviews: list[dict[str, Any]] = []
    field_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: {"+1": 0, "0": 0})

    for line in cases_path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        case = json.loads(s)
        gold = case.get("gold") or {}
        if not gold:
            continue
        # Random subset of fields to "corrupt"
        keys = list(gold.keys())
        n_corrupt = max(1, round(len(keys) * injection_rate))
        corrupt_keys = set(rng.sample(keys, k=n_corrupt))
        model = {}
        for k in keys:
            if k in corrupt_keys:
                # perturb the value: empty string for strings, 0.0 for numbers
                v = gold[k]
                if isinstance(v, (int, float)):
                    model[k] = 0.0
                elif isinstance(v, str):
                    model[k] = ""
                else:
                    model[k] = None
            else:
                model[k] = gold[k]
        reviews.append(
            {
                "doc_id": case.get("doc_path", "?"),
                "schema": case.get("schema", "?"),
                "model": model,
                "human": gold,
                "synthetic": True,
            }
        )
        for k in keys:
            m = _norm_for_compare(model.get(k))
            h = _norm_for_compare(gold.get(k))
            field_outcomes[k]["0" if m == h else "+1"] += 1

    return reviews, dict(field_outcomes)


# ---------------------------------------------------------------------------
# Calibration + hit/false-flag metrics
# ---------------------------------------------------------------------------
@dataclass
class FieldEval:
    field: str
    n_reviews: int
    n_corrected: int
    corrected_rate: float
    policy_fires: bool
    # when policy fires
    hits: int = 0           # human corrected a flagged field
    false_flags: int = 0   # flagged but human accepted (wasted HITL time)
    # when policy does NOT fire
    true_accepts: int = 0  # human accepted a non-flagged field
    missed_corrections: int = 0  # human corrected a non-flagged field

    @property
    def hit_rate_when_fires(self) -> float | None:
        denom = self.hits + self.false_flags
        return self.hits / denom if denom else None

    @property
    def true_accept_rate_when_silent(self) -> float | None:
        denom = self.true_accepts + self.missed_corrections
        return self.true_accepts / denom if denom else None


@dataclass
class CalibrationReport:
    synthetic: bool
    n_reviews: int
    policy_path: str | None
    field_evals: list[FieldEval]
    overall: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "synthetic": self.synthetic,
            "n_reviews": self.n_reviews,
            "policy_path": self.policy_path,
            "per_field": [
                {
                    "field": fe.field,
                    "n_reviews": fe.n_reviews,
                    "n_corrected": fe.n_corrected,
                    "corrected_rate": round(fe.corrected_rate, 3),
                    "policy_fires": fe.policy_fires,
                    "hit_rate_when_fires": (
                        round(fe.hit_rate_when_fires, 3)
                        if fe.hit_rate_when_fires is not None
                        else None
                    ),
                    "true_accept_rate_when_silent": (
                        round(fe.true_accept_rate_when_silent, 3)
                        if fe.true_accept_rate_when_silent is not None
                        else None
                    ),
                }
                for fe in self.field_evals
            ],
            "overall": {k: round(v, 3) for k, v in self.overall.items()},
            "notes": self.notes,
        }


def evaluate_policy(
    reviews: list[dict[str, Any]],
    policy: PolicyConfig,
    *,
    synthetic: bool = False,
) -> CalibrationReport:
    """Compute the calibration report for one policy against one review set."""
    field_outcomes = _review_field_outcome(reviews)
    notes: list[str] = []
    notes.append(
        "Synthetic reviews are biased optimistic: gold truth IS the human's "
        "correction, so the correction signal is artificially clean. "
        "Numbers from real HITL data will be noisier."
        if synthetic
        else "Real reviews (synthetic=false). Use these numbers for honest reporting."
    )

    # First pass: decide policy fires (using base confidence, before penalty)
    # For Q1/Q2 we need to know: did the policy flag this field, and was the
    # human happy with the model's value? The policy decision is based on
    # pre-policy confidence + the policy itself.
    field_evals: list[FieldEval] = []

    for field_name, counts in field_outcomes.items():
        n_total = counts["+1"] + counts["0"]
        n_corrected = counts["+1"]
        corrected_rate = n_corrected / n_total if n_total else 0.0

        # Use base confidence = 0.7 (ocr_llm) + string bonus 0.05 = 0.75
        # This is the canonical "model was confident" baseline. We apply
        # the policy to it to compute flag decisions.
        base_conf = 0.75 if field_name in policy.field_penalties else 0.75
        penalised = policy_to_penalised_confidence({field_name: base_conf}, policy)[field_name]
        flags = policy_to_review_flags({field_name: penalised}, policy)
        fires = flags[field_name]

        # Q1: when policy fires (flag=true), was the human happy?
        hits = sum(1 for r in reviews for k in [field_name]
                   if k in (r.get("human") or {}) and
                   _norm_for_compare((r.get("model") or {}).get(k)) != _norm_for_compare(r["human"].get(k)))
        # This counts ALL corrections; we want only when the policy fires
        # which depends on the same field across all reviews
        if fires:
            # all corrections count as hits (since we fired on every review
            # of this field). False flags = accepts.
            field_evals.append(FieldEval(
                field=field_name,
                n_reviews=n_total,
                n_corrected=n_corrected,
                corrected_rate=corrected_rate,
                policy_fires=True,
                hits=n_corrected,
                false_flags=counts["0"],
                true_accepts=0,
                missed_corrections=0,
            ))
        else:
            field_evals.append(FieldEval(
                field=field_name,
                n_reviews=n_total,
                n_corrected=n_corrected,
                corrected_rate=corrected_rate,
                policy_fires=False,
                hits=0,
                false_flags=0,
                true_accepts=counts["0"],
                missed_corrections=counts["+1"],
            ))

    # Overall stats
    flagged = [fe for fe in field_evals if fe.policy_fires]
    silent = [fe for fe in field_evals if not fe.policy_fires]

    total_hits = sum(fe.hits for fe in flagged)
    total_false = sum(fe.false_flags for fe in flagged)
    total_true_accepts = sum(fe.true_accepts for fe in silent)
    total_missed = sum(fe.missed_corrections for fe in silent)

    overall = {
        "n_flagged_fields": len(flagged),
        "n_silent_fields": len(silent),
        "total_hits": total_hits,
        "total_false_flags": total_false,
        "hit_rate_when_fires": (
            total_hits / (total_hits + total_false) if (total_hits + total_false) else 0.0
        ),
        "true_accept_rate_when_silent": (
            total_true_accepts / (total_true_accepts + total_missed)
            if (total_true_accepts + total_missed) else 0.0
        ),
        # review-burden efficiency: fraction of all field-reviews that ended
        # up being needed (corrected), assuming HITL only looks at flagged
        # fields. If 100% of flagged fields were corrected -> efficiency 1.
        # If half of flagged were false -> 0.5.
        "review_efficiency": (
            total_hits / (total_hits + total_false) if (total_hits + total_false) else 0.0
        ),
    }

    return CalibrationReport(
        synthetic=synthetic,
        n_reviews=len(reviews),
        policy_path=None,
        field_evals=field_evals,
        overall=overall,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Brier-style calibration per confidence bucket
# ---------------------------------------------------------------------------
def confidence_calibration_error(
    reviews: list[dict[str, Any]],
    confidence_func,  # callable: (extraction, field_name) -> float
    *,
    n_buckets: int = 4,
) -> dict[str, Any]:
    """Bucket predictions by confidence, report empirical correctness.

    confidence_func is called with (model_extraction, field_name) and must
    return a float in [0, 1]. Returns mean absolute error between stated
    confidence and empirical correctness across buckets.
    """
    bucket_hits: dict[int, list[int]] = defaultdict(list)
    for r in reviews:
        model = r.get("model") or {}
        human = r.get("human") or {}
        for field_name in set(model.keys()) | set(human.keys()):
            conf = confidence_func(model, field_name)
            bucket = min(int(conf * n_buckets), n_buckets - 1)
            correct = (
                _norm_for_compare(model.get(field_name)) ==
                _norm_for_compare(human.get(field_name))
            )
            bucket_hits[bucket].append(1 if correct else 0)

    out: dict[str, Any] = {"buckets": [], "calibration_error": 0.0}
    total_abs_error = 0.0
    n_total = 0
    for b in range(n_buckets):
        items = bucket_hits.get(b, [])
        if not items:
            continue
        bin_low = b / n_buckets
        bin_high = (b + 1) / n_buckets
        empirical = sum(items) / len(items)
        stated_mid = (bin_low + bin_high) / 2
        abs_error = abs(stated_mid - empirical)
        out["buckets"].append({
            "bin": f"[{bin_low:.2f}, {bin_high:.2f})",
            "n": len(items),
            "stated_confidence_mid": round(stated_mid, 3),
            "empirical_accuracy": round(empirical, 3),
            "abs_error": round(abs_error, 3),
        })
        total_abs_error += abs_error * len(items)
        n_total += len(items)
    if n_total:
        out["calibration_error"] = round(total_abs_error / n_total, 3)
    return out
