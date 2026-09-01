"""Offline RL batch: derive rewards from a stored-results file, update policy.

This is the v0.1 version — runs offline against any storage backend.
Online (per-review) update ships as a v0.2 hook.

Usage:
    idp rl-update --storage idp_data/results.jsonl --output policy.json
    idp rl-update --reviews reviews.jsonl --output policy.json

Inputs:
  --storage <jsonl>    path to JsonFileStorage dump
  --reviews <jsonl>    OR a hand-crafted JSONL of {model, human, doc_id, schema}

Output:
  --output <json>      policy.json with field_floors + field_penalties

The output is plain JSON, version-controllable, inspectable, no hidden state.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from idp.policy_config import PolicyConfig  # re-exported here
from idp.rl.policy import update_policy
from idp.rl.reward import ReviewRewards, aggregate_rewards, derive_field_rewards
from idp.storage.store import JsonFileStorage, StoredResult

log = logging.getLogger(__name__)


def update_policy_from_storage(
    storage_path: str | Path,
    output_path: str | Path,
    current: PolicyConfig | None = None,
) -> PolicyConfig:
    """Read all reviewed results from storage, aggregate, update policy."""
    storage = JsonFileStorage(str(storage_path))
    # List reviewed entries by reading all
    all_results = storage.list(limit=100_000)
    reviews: list[ReviewRewards] = []
    for r in all_results:
        rewards = derive_field_rewards(r)
        if rewards is None:
            continue
        reviews.append(rewards)
    stats = aggregate_rewards(reviews)
    new_policy = update_policy(stats, current=current)
    new_policy.save(output_path)
    return new_policy


def update_policy_from_reviews_file(
    reviews_path: str | Path,
    output_path: str | Path,
    current: PolicyConfig | None = None,
) -> PolicyConfig:
    """Read hand-crafted reviews from JSONL and update policy.

    Each line: {"doc_id": "...", "schema": "...", "model": {...}, "human": {...}}
    """
    reviews: list[ReviewRewards] = []
    for line in Path(reviews_path).read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        d = json.loads(s)
        rewards = ReviewRewards(doc_id=d["doc_id"], schema_name=d["schema"])
        keys = set((d.get("model") or {}).keys()) | set((d.get("human") or {}).keys())
        for k in keys:
            mv = (d.get("model") or {}).get(k)
            hv = (d.get("human") or {}).get(k)
            from idp.rl.reward import _norm_for_compare
            if _norm_for_compare(mv) == _norm_for_compare(hv):
                from idp.rl.reward import FieldReward
                rewards.field_rewards.append(FieldReward(k, 0, mv, hv))
            else:
                from idp.rl.reward import FieldReward
                rewards.field_rewards.append(FieldReward(k, +1, mv, hv))
        reviews.append(rewards)
    stats = aggregate_rewards(reviews)
    new_policy = update_policy(stats, current=current)
    new_policy.save(output_path)
    return new_policy


def update_policy_from_sql(
    db_url: str,
    output_path: str | Path,
    current: PolicyConfig | None = None,
) -> PolicyConfig:
    """Read reviews from a SQL storage backend (sqlite or postgres) and update policy.

    Requires SqlStorage (idp.storage.sql). Import is lazy to avoid forcing
    psycopg as a dependency.
    """
    from idp.storage.sql import SqlStorage

    storage = SqlStorage(db_url)
    review_dicts = storage.reviews_as_dicts()
    reviews: list[ReviewRewards] = []
    for d in review_dicts:
        rewards = ReviewRewards(
            doc_id=d.get("doc_id", "?"),
            schema_name=d.get("schema", "?"),
        )
        model = d.get("model") or {}
        human = d.get("human") or {}
        from idp.rl.reward import _norm_for_compare, FieldReward

        keys = set(model.keys()) | set(human.keys())
        for k in keys:
            mv = model.get(k)
            hv = human.get(k)
            if _norm_for_compare(mv) == _norm_for_compare(hv):
                rewards.field_rewards.append(FieldReward(k, 0, mv, hv))
            else:
                rewards.field_rewards.append(FieldReward(k, +1, mv, hv))
        reviews.append(rewards)
    stats = aggregate_rewards(reviews)
    new_policy = update_policy(stats, current=current)
    new_policy.save(output_path)
    return new_policy
