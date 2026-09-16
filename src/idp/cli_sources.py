"""Source resolution for the ``idp batch`` CLI.

A "source" in ``idp batch`` is one of:
  - a path to a file: included as-is
  - a path to a directory: recursively scanned for document-like files
  - ``@<file>``: a text file with one path per line, treated as a path list

Document-like extensions (matched case-insensitively):
  - .pdf, .png, .jpg, .jpeg, .tiff, .tif, .txt

This module is intentionally separate from the CLI itself: it can
be imported by Python scripts that want the same path-collection
behavior without going through the CLI.

The functions here are sync on purpose. If you need async (e.g.
streaming paths from an S3 prefix without listing first), wrap
the result and stream yourself.
"""
from __future__ import annotations

from pathlib import Path

# Document extensions we recognize. Case-insensitive match.
DOCUMENT_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".txt",
)


def _is_document(path: Path) -> bool:
    return path.suffix.lower() in DOCUMENT_EXTENSIONS


def _expand_one(source: str, base: Path | None = None) -> list[Path]:
    """Expand a single source specifier to a list of file paths.

    Args:
        source: one of:
          - a literal filesystem path (file or directory)
          - "@<file>" pointing to a text file of paths, one per line
        base: when reading @-files, relative paths are resolved
               against this base. Defaults to the file's parent dir.

    Returns:
        List of paths. Order is filesystem order (not sorted).
        Duplicates are removed.

    Raises:
        FileNotFoundError: if the source doesn't exist.
        NotADirectoryError: if @<file> points at a directory.
    """
    # @file -> read paths from file
    if source.startswith("@"):
        list_path = Path(source[1:])
        if not list_path.exists():
            raise FileNotFoundError(f"path list file not found: {list_path}")
        if not list_path.is_file():
            raise NotADirectoryError(f"path list is a directory, expected a file: {list_path}")
        # Relative paths inside the list resolve against the list file's dir
        rel_base = base if base is not None else list_path.parent
        out: list[Path] = []
        for raw in list_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            p = Path(stripped)
            if not p.is_absolute():
                p = rel_base / p
            out.append(p)
        return out

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"source not found: {p}")
    if p.is_file():
        return [p]
    if p.is_dir():
        # Recursive scan, sorted for deterministic output
        return sorted(
            f
            for f in p.rglob("*")
            if f.is_file() and _is_document(f)
        )
    # Special files (symlinks, devices) — not supported
    raise FileNotFoundError(f"unsupported source type: {p}")


def collect_paths(sources: list[str]) -> list[Path]:
    """Expand a list of source specifiers to a flat, deduplicated path list.

    Args:
        sources: list of source strings (see ``_expand_one``).

    Returns:
        Flat list of paths that are files (not directories). Order is
        the order in which they were discovered, with duplicates
        removed (first occurrence wins).

    Raises:
        FileNotFoundError: if any source doesn't exist.
    """
    seen: set[Path] = set()
    out: list[Path] = []
    for s in sources:
        for p in _expand_one(s):
            # Resolve to absolute so dedup works across relative + absolute
            abs_p = p.resolve() if p.exists() else p
            if abs_p in seen:
                continue
            seen.add(abs_p)
            out.append(abs_p)
    return out
