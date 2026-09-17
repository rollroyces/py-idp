"""Backward-compat shim. ``pretty_print_result`` moved to ``idp.console``.

The function was previously here as an implementation-detail module;
it's now part of the public surface. The old import
``from idp._util import pretty_print_result`` keeps working.

Prefer the new path: ``from idp.console import pretty_print_result``.
"""
from idp.console import pretty_print_result

__all__ = ["pretty_print_result"]