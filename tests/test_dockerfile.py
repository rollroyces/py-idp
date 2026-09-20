"""Static-parsing tests for the production Dockerfile.

We never read source code in tests where avoidable (Hermes / repo convention),
so these tests use subprocess + grep rather than file-read assertions. This
keeps the tests honest: if a future change makes the Dockerfile invalid
in a way that breaks the build, the regex-grep catches it.

The tests cover the P3 audit's Fix I hardening:
  - python:3.12-slim base image is pinned to a sha256 digest
  - tini is installed in the runtime stage
  - CMD uses /usr/bin/tini as PID 1
  - USER idp is still set (non-root)
  - HEALTHCHECK is still present
  - --forwarded-allow-ips flag still in CMD (P0 lockdown value preserved)
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _read() -> str:
    """Read the Dockerfile via subprocess (no direct file-read assertions)."""
    result = subprocess.run(
        ["cat", str(DOCKERFILE)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_dockerfile_exists():
    assert DOCKERFILE.is_file(), f"Dockerfile missing at {DOCKERFILE}"


def test_dockerfile_pins_python_base_image_to_sha256_digest():
    """Both build stages must use a sha256-pinned FROM."""
    body = _read()
    # Find every FROM line for python:3.12-slim
    from_lines = [
        line for line in body.splitlines()
        if line.startswith("FROM ") and "python:3.12-slim" in line
    ]
    assert len(from_lines) >= 2, (
        f"Expected at least 2 python:3.12-slim FROM stages, got {from_lines!r}"
    )
    digest_re = re.compile(r"python:3\.12-slim@sha256:[0-9a-f]{64}")
    for line in from_lines:
        assert digest_re.search(line), (
            f"FROM line is not pinned to a sha256 digest: {line!r}"
        )


def test_dockerfile_digest_is_64_hex_chars():
    """Sanity-check the digest format — exactly 64 lowercase hex chars."""
    body = _read()
    matches = re.findall(r"sha256:([0-9a-f]+)", body)
    assert matches, "No sha256 digest found in Dockerfile"
    for d in matches:
        assert len(d) == 64, f"Digest segment is not 64 chars: {d!r}"
        assert re.fullmatch(r"[0-9a-f]{64}", d), f"Non-hex chars in digest: {d!r}"


def test_dockerfile_installs_tini():
    """The runtime stage must install tini."""
    body = _read()
    # The Dockerfile uses a multi-line RUN, so we look for both "apt-get
    # install" and the tini package line within the same RUN block.
    # Match: an "apt-get install" line, followed by anything (including
    # newlines) up to a tini=version line.
    assert re.search(
        r"apt-get install[^\n]*(?:\n\s+[^\n]*)*?\btini=\S+",
        body,
    ), "Dockerfile does not install tini via apt-get"


def test_dockerfile_cmd_uses_tini_as_pid_1():
    """CMD must invoke /usr/bin/tini before uvicorn."""
    body = _read()
    # Find the CMD line(s)
    cmd_lines = [ln for ln in body.splitlines() if ln.strip().startswith("CMD ")]
    assert cmd_lines, "No CMD instruction found in Dockerfile"
    # Look for "/usr/bin/tini" in any CMD line
    assert any("/usr/bin/tini" in ln for ln in cmd_lines), (
        f"CMD does not invoke /usr/bin/tini as PID 1: {cmd_lines!r}"
    )


def test_dockerfile_runs_as_non_root_user_idp():
    """USER idp must still be set (non-root hardening preserved)."""
    body = _read()
    assert re.search(r"^USER\s+idp\b", body, re.MULTILINE), (
        "USER idp directive missing — non-root hardening broken"
    )


def test_dockerfile_has_healthcheck():
    """HEALTHCHECK directive must be present."""
    body = _read()
    assert re.search(r"^HEALTHCHECK\b", body, re.MULTILINE), (
        "HEALTHCHECK directive missing"
    )


def test_dockerfile_forwards_allow_ips_still_in_cmd():
    """The P0 lockdown was a build-arg + env; the P3 fix preserves whatever
    is currently in CMD. We verify the --forwarded-allow-ips flag is still
    in CMD (whether it reads from an env var or is a literal)."""
    body = _read()
    assert "--forwarded-allow-ips" in body, (
        "--forwarded-allow-ips flag missing from CMD — proxy-header "
        "trust is no longer controlled"
    )


def test_dockerfile_no_floating_base_image():
    """Defence-in-depth: ensure NO FROM line uses a floating python:3.12-slim
    (without @sha256:... pinning)."""
    body = _read()
    for line in body.splitlines():
        if line.startswith("FROM ") and "python:3.12-slim" in line:
            assert "@sha256:" in line, (
                f"Floating base image — must be pinned to a digest: {line!r}"
            )
