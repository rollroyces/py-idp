"""Backward-compat shim. process_batch moved to idp.batch.

This module is a thin re-export so existing user code
(``from idp.llm.nanonets_batch import process_batch``) keeps
working. The new public path is ``idp.batch``.

This file used to contain the full implementation; it was
moved to ``idp.batch`` and this file is now a re-export shim.
"""
from idp.batch import BatchItemResult, process_batch

__all__ = ["BatchItemResult", "process_batch"]
