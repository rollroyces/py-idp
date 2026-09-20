"""Tests for the custom-backend example (examples/06_custom_backend.py).

Covers:
  - The example file imports cleanly (no syntax errors, no broken imports).
  - Running the example executes successfully end-to-end.
  - list_backends() in the test scope includes the registered "echo"
    backend (proving the decorator fires at import time).
  - The module docstring accurately describes what the example shows.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_PATH = REPO_ROOT / "examples" / "06_custom_backend.py"


def _run_inline(python_code: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run ``python -c <python_code>`` in the repo root and return result."""
    return subprocess.run(
        [sys.executable, "-c", python_code],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
    )


def test_example_file_imports_cleanly():
    """Importing the example must succeed without errors."""
    code = (
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('cb_ex', {str(EXAMPLE_PATH)!r}); "
        "m = importlib.util.module_from_spec(spec); "
        "assert spec.loader is not None; "
        "spec.loader.exec_module(m); "
        "print('ok');"
    )
    result = _run_inline(code)
    assert result.returncode == 0, (
        f"Example failed to import.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "ok" in result.stdout


def test_example_runs_end_to_end():
    """Running the example as a script must succeed."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLE_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Example exited non-zero.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    # It should at least print the resolved backend name
    assert "Resolved backend: 'echo'" in result.stdout
    # And report the extraction result
    assert "Extraction result:" in result.stdout


def test_example_registers_echo_backend_via_decorator():
    """Importing the example must register "echo" in the backend registry."""
    code = (
        "import importlib.util; "
        f"spec = importlib.util.spec_from_file_location('cb_ex', {str(EXAMPLE_PATH)!r}); "
        "m = importlib.util.module_from_spec(spec); "
        "assert spec.loader is not None; "
        "spec.loader.exec_module(m); "
        "from idp.llm.backend import list_backends; "
        "names = list_backends(); "
        "assert 'echo' in names, f'echo not in {names}'; "
        "print('echo registered');"
    )
    result = _run_inline(code)
    assert result.returncode == 0, (
        f"Registry check failed.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "echo registered" in result.stdout


def test_example_module_docstring_mentions_register_backend():
    """The example's module docstring should explain the register_backend
    extension point — that's the whole point of the file."""
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    # The triple-quoted module docstring at the top
    assert '"""' in source
    # Pull the first docstring block
    start = source.find('"""')
    end = source.find('"""', start + 3)
    assert end > start > 0, "Expected a module-level docstring"
    docstring = source[start:end]
    # Docstring should mention the decorator
    assert "register_backend" in docstring
    # ...and at least one of the alternative names
    assert "Backend" in docstring or "backend" in docstring


def test_example_imports_real_idp_symbols():
    """The example should use the public idp.* API (not internal mocks)."""
    source = EXAMPLE_PATH.read_text(encoding="utf-8")
    # Public symbols the example is supposed to demonstrate
    assert "from idp.llm.backend import" in source
    assert "register_backend" in source
    assert "get_backend" in source
    assert "from idp.pipeline.pipeline import Pipeline" in source
    assert "from idp.core.document import Document" in source
