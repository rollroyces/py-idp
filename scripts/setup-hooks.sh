#!/usr/bin/env bash
# Setup project git hooks. Run once after cloning.
#
# This configures git to use the tracked hooks in .githooks/
# instead of the per-clone .git/hooks/ directory.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR="$REPO_ROOT/.githooks"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "error: $HOOKS_DIR not found" >&2
    exit 1
fi

# Make sure all hooks are executable. README.md and other docs in
# .githooks/ shouldn't be chmod +x; only files with no extension
# (git hook convention: `pre-commit`, `commit-msg`, etc.) or
# explicit `.hook` extension.
chmod +x "$HOOKS_DIR"/pre-commit "$HOOKS_DIR"/commit-msg "$HOOKS_DIR"/*.hook 2>/dev/null || true

git config core.hooksPath .githooks
echo "git hooks installed: core.hooksPath = .githooks"
echo ""
echo "Active hooks:"
for hook in "$HOOKS_DIR"/pre-commit "$HOOKS_DIR"/commit-msg "$HOOKS_DIR"/*.hook; do
    [ -f "$hook" ] || continue
    [ -x "$hook" ] || continue  # only list executables
    name=$(basename "$hook")
    echo "  - $name"
done
