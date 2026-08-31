"""Module-level re-export so users can `from idp.policy_config import PolicyConfig`.

Most users will use `idp.rl.PolicyConfig`; this is a shortcut.
"""
from idp.rl.policy import PolicyConfig  # noqa: F401
