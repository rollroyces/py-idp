# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optional API key authentication.

For self-hosted single-user deployment this is overkill. For SaaS or
multi-tenant installs you wire this in front of the FastAPI / Streamlit
mount points (see examples/api.py).
"""
from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from hashlib import sha256


@dataclass
class AuthContext:
    user_id: str
    api_key: str


def hash_key(raw: str, salt: str = "py-idp") -> str:
    """Hash an API key for at-rest storage."""
    return sha256(f"{salt}:{raw}".encode()).hexdigest()


def verify(raw: str, stored_hash: str, salt: str = "py-idp") -> bool:
    """Constant-time compare of raw key against stored hash."""
    return hmac.compare_digest(hash_key(raw, salt), stored_hash)


def require_api_key(env_var: str = "IDP_API_KEY") -> str | None:
    """Read an API key from environment. Returns None if not set."""
    return os.environ.get(env_var)


def make_default_key() -> str:
    """Generate a dev API key for bootstrap (printed to console, never logged)."""
    import secrets
    return secrets.token_urlsafe(32)
