"""Tests for the ``idp batch`` CLI's path-source resolution.

These tests cover ``idp.cli_sources.collect_paths`` directly, plus
end-to-end tests of the ``idp batch`` command via Typer's CliRunner.

The CLI is exercised in-process (no subprocess) so the tests run fast
and can introspect output paths written to a tempdir.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from idp.cli_sources import collect_paths
from idp.pipeline.cli import app

# --- collect_paths unit tests --------------------------------------------


def test_collect_paths_single_file(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("hi")
    out = collect_paths([str(f)])
    assert out == [f.resolve()]


def test_collect_paths_directory_recurses(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.pdf").write_text("")
    (sub / "b.png").write_bytes(b"")
    (sub / "ignored.txt.bak").write_text("not a doc")  # unknown extension
    out = collect_paths([str(tmp_path)])
    names = sorted(p.name for p in out)
    # both a.pdf and b.png should be picked up; .bak is ignored
    assert names == ["a.pdf", "b.png"]


def test_collect_paths_deduplicates(tmp_path: Path) -> None:
    f = tmp_path / "doc.pdf"
    f.write_text("")
    # Same file via different absolute-path representations
    out = collect_paths([str(f), str(f.resolve())])
    assert len(out) == 1


def test_collect_paths_at_file(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_text("")
    (docs / "b.pdf").write_text("")
    list_file = tmp_path / "list.txt"
    list_file.write_text("\n".join([str(docs / "a.pdf"), str(docs / "b.pdf"), "", "# comment"]))

    out = collect_paths([f"@{list_file}"])
    assert len(out) == 2


def test_collect_paths_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_paths([str(tmp_path / "nope.pdf")])


def test_collect_paths_at_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_paths([f"@{tmp_path / 'missing.txt'}"])


# --- CLI integration tests -----------------------------------------------


runner = CliRunner()


def test_batch_help() -> None:
    r = runner.invoke(app, ["batch", "--help"])
    assert r.exit_code == 0
    assert "Run a pipeline over many documents" in r.stdout


def test_batch_dry_run_lists_paths(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.pdf").write_text("")
    (docs / "b.pdf").write_text("")
    r = runner.invoke(
        app,
        ["batch", str(docs), "--backend", "mock", "--dry-run"],
    )
    assert r.exit_code == 0
    assert "dry-run: would process 2 documents" in r.stdout


def test_batch_writes_output_and_report(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(3):
        (docs / f"inv_{i}.txt").write_text(f"Invoice {i}")
    out_path = tmp_path / "out.jsonl"
    report_path = tmp_path / "report.json"
    dlq_path = tmp_path / "dlq.jsonl"

    r = runner.invoke(
        app,
        [
            "batch", str(docs),
            "--backend", "mock",
            "--output", str(out_path),
            "--report", str(report_path),
            "--dlq", str(dlq_path),
        ],
    )

    assert r.exit_code == 0, r.stdout

    # output JSONL: one record per doc, all valid JSON
    out_lines = out_path.read_text().splitlines()
    assert len(out_lines) == 3
    for line in out_lines:
        rec = json.loads(line)
        assert "path" in rec
        assert "ok" in rec
        assert "extraction" in rec

    # report JSON: summary with expected keys
    report = json.loads(report_path.read_text())
    assert report["total"] == 3
    assert report["succeeded"] == 3
    assert report["failed"] == 0
    assert "elapsed_seconds" in report
    assert "throughput_docs_per_sec" in report
    assert "latency_p50_sec" in report
    assert "latency_p95_sec" in report
    assert report["backend"] == "mock"

    # DLQ file was created even though empty
    assert dlq_path.exists()


def test_batch_missing_source_exits_1(tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        ["batch", str(tmp_path / "missing"), "--backend", "mock"],
    )
    assert r.exit_code == 1
    assert "error" in r.stdout.lower()


def test_batch_no_sources_exits_1(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    r = runner.invoke(
        app,
        ["batch", str(empty), "--backend", "mock"],
    )
    assert r.exit_code == 1
    assert "no documents found" in r.stdout

# ---------------------------------------------------------------------------
# cli_sources: edge cases (lines 63, 73, 90)
# ---------------------------------------------------------------------------
def test_at_file_is_directory_raises_not_a_directory(tmp_path):
    """@file pointing at a directory raises NotADirectoryError."""
    dir_as_file = tmp_path / "actually_a_dir"
    dir_as_file.mkdir()
    with pytest.raises(NotADirectoryError):
        collect_paths([f"@{dir_as_file}"])


def test_at_file_relative_paths_resolve_against_list_file(tmp_path):
    """Inside @file, relative paths resolve against the list file's dir."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.pdf").write_text("")
    list_file = tmp_path / "list.txt"
    list_file.write_text("sub/a.pdf\n")  # relative path

    out = collect_paths([f"@{list_file}"])
    assert len(out) == 1
    assert out[0].resolve() == (sub / "a.pdf").resolve()


def test_collect_paths_handles_paths_with_comments_and_blanks(tmp_path):
    """@file with comments and blank lines: only valid paths are returned."""
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "a.pdf").write_text("")
    (sub / "b.pdf").write_text("")
    list_file = tmp_path / "list.txt"
    list_file.write_text(
        f"# This is a comment\n"
        f"\n"
        f"{sub}/a.pdf\n"
        f"# Another comment\n"
        f"{sub}/b.pdf\n"
        f"   \n"  # whitespace-only line
    )
    out = collect_paths([f"@{list_file}"])
    assert len(out) == 2
