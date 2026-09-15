#!/usr/bin/env python3
"""Copy the current README.md contents to the system clipboard.

Used to paste the new README into PyPI's release-description editor at
https://pypi.org/manage/project/py-idp/releases/ to fix the broken links
on the currently-published py-idp 0.3.1 wheel without bumping the version.

Tries pbcopy (macOS), xclip (Linux), xsel (Linux), then falls back to
printing the content with a marker for manual copy.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

README = Path("README.md")


def copy_via(cmd: list[str]) -> bool:
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        p.communicate(input=README.read_text().encode("utf-8"), timeout=10)
        return p.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    text = README.read_text()
    if shutil.which("pbcopy"):
        if copy_via(["pbcopy"]):
            print(f"copied {len(text)} chars to clipboard (pbcopy)")
            return
    if shutil.which("xclip"):
        if copy_via(["xclip", "-selection", "clipboard"]):
            print(f"copied {len(text)} chars to clipboard (xclip)")
            return
    if shutil.which("xsel"):
        if copy_via(["xsel", "--clipboard", "--input"]):
            print(f"copied {len(text)} chars to clipboard (xsel)")
            return
    print(f"no clipboard tool found; printing README ({len(text)} chars) instead:")
    print("---8<---")
    sys.stdout.write(text)
    print("---8<---")


if __name__ == "__main__":
    main()