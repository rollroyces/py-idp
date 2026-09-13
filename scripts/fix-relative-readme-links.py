#!/usr/bin/env python3
"""Rewrite relative links in the three py-idp READMEs to absolute GitHub URLs.

PyPI doesn't resolve relative markdown links against the GitHub repo, so links
like (README.md) on the PyPI page 404. This script rewrites every relative URL
appearing inside `(...)` in README.md / README.zh-CN.md / README.zh-TW.md to
https://github.com/rollroyces/py-idp/blob/main/<path>, leaving already-absolute
URLs (http://, https://, mailto:, #anchor) untouched.

Approach: scan `(...)` groupings with balanced-paren tracking. For each opening
`(` that follows `[...]` (a markdown link/image syntax), look at the URL inside
and rewrite it if relative. This handles nested badge syntax
`[![alt](badge.svg)](LICENSE-AGPL)` correctly because we always look at the URL
inside the parens, not the contents of the surrounding brackets.

Run:
    .venv/bin/python scripts/fix-relative-readme-links.py
"""
from __future__ import annotations

from pathlib import Path

BASE = "https://github.com/rollroyces/py-idp/blob/main/"
TARGETS = [
    Path("README.md"),
    Path("README.zh-CN.md"),
    Path("README.zh-TW.md"),
]


def is_absolute(url: str) -> bool:
    return url.startswith(("http://", "https://", "mailto:", "#"))


def find_link_urls(text: str) -> list[tuple[int, int, str]]:
    """Yield (start_of_paren, end_of_paren, url) for every markdown link/image URL.

    Only records URLs where the immediately preceding non-whitespace character
    is `]` (closing a markdown link/image). This avoids accidentally rewriting
    parens in code blocks, URLs in plain text, etc.
    """
    results: list[tuple[int, int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        # Find the next `(...)` block with balanced parens.
        if text[i] != "(":
            i += 1
            continue
        # Walk to find the matching close paren.
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            ch = text[j]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            # Unbalanced; skip and advance.
            i += 1
            continue
        # Check the character immediately before the `(`. If it's `]`, this is
        # a markdown link/image target.
        k = i - 1
        while k >= 0 and text[k] in " \t":
            k -= 1
        if k >= 0 and text[k] == "]":
            url = text[i + 1 : j]
            results.append((i, j, url))
        i = j + 1
    return results


def rewrite(text: str) -> tuple[str, int]:
    matches = find_link_urls(text)
    if not matches:
        return text, 0
    out: list[str] = []
    cursor = 0
    changes = 0
    for start, end, url in matches:
        out.append(text[cursor:start + 1])  # include the opening "("
        if is_absolute(url):
            out.append(url)
        elif "#" in url:
            path, anchor = url.split("#", 1)
            out.append(f"{BASE}{path}#{anchor}")
            changes += 1
        else:
            out.append(f"{BASE}{url}")
            changes += 1
        out.append(text[end])  # include the closing ")"
        cursor = end + 1
    out.append(text[cursor:])
    return "".join(out), changes


def rewrite_file(path: Path) -> int:
    text = path.read_text()
    new, n = rewrite(text)
    if n:
        path.write_text(new)
    return n


def main() -> None:
    total = 0
    for p in TARGETS:
        if not p.exists():
            print(f"skip (missing): {p}")
            continue
        n = rewrite_file(p)
        print(f"{p}: rewrote {n} links")
        total += n
    print("---")
    print(f"total: {total} links rewritten across {len(TARGETS)} files")


if __name__ == "__main__":
    main()