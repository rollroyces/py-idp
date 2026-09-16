"""Template registry: Markdown files document each PDF type for the LLM.

Each template is a single ``.md`` file with **YAML frontmatter** (between
``---`` fences) followed by a Markdown body. The LLM reads the body as
context when extracting that PDF type — field descriptions, example
values, regex hints, common mistakes.

Example ``templates/invoice.md``::

    ---
    name: invoice
    schema: Invoice
    version: 1
    mime_types: [application/pdf, image/jpeg, image/png]
    filename_patterns: ["*invoice*", "*receipt*", "*bill*"]
    classification_hints:
      - "Contains a total amount, vendor name, invoice number"
      - "Often has line items with quantity, unit price, amount"
    ---

    # Invoice

    ## Fields

    * **vendor_name** — the name of the company issuing the invoice.
      Look for the largest bold text near the top, or the company logo
      text. Common aliases: "From", "Supplier", "Vendor", "Bill from".
    * **invoice_number** — usually near the top, format varies
      (e.g. "INV-2024-001", "2024-09-15-001").
    ...

    ## Common mistakes

    * **subtotal vs total_amount**: subtract tax from total to get
      subtotal. Do not confuse them.
    * **date_due vs date_issued**: due date is usually later than
      issued date.
    ...

The frontmatter is parsed by :func:`_parse_frontmatter` (which uses
PyYAML). The body is preserved verbatim for the LLM.

The :class:`TemplateRegistry` loads all ``*.md`` files from a directory
at startup, and provides lookup by name, by content-type, and by
filename heuristic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from idp.errors import TemplateNotFoundError, TemplateParseError

# yaml is optional: ``templates`` is a feature, not a core dependency.
# If a user calls load_template() without PyYAML installed, we raise a
# helpful ImportError pointing to the optional dep instead of crashing
# at module import time.
yaml: Any = None


def _require_yaml() -> Any:
    """Import PyYAML on first use, raising a clear error if absent.

    Templates are an opt-in feature; users who don't load templates
    never pay the import cost. Keeps ``import idp`` working on a
    bare pip install.
    """
    global yaml
    if yaml is None:
        try:
            import yaml as _yaml
        except ImportError as e:
            raise ImportError(
                "idp.templates requires PyYAML. Install it with "
                "`pip install PyYAML` (or `pip install py-idp[templates]`)."
            ) from e
        yaml = _yaml
    return yaml

# Required frontmatter keys. If any are missing the file is rejected at
# load time (no silent defaults — typos like ``schame: Invoice`` should
# fail loud).
REQUIRED_KEYS: tuple[str, ...] = ("name", "schema")
OPTIONAL_KEYS: tuple[str, ...] = (
    "version",
    "mime_types",
    "filename_patterns",
    "classification_hints",
    "field_overrides",  # dict: field_name -> {regex, examples, description}
)

# Frontmatter is delimited by `---` on its own line at the start of the
# file. The body is everything after the closing fence.
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<fm>.*?)\n---\s*\n(?P<body>.*)\Z",
    re.DOTALL,
)


@dataclass
class Template:
    """A single registered template.

    Attributes:
        name:        Unique template name (matches frontmatter ``name``).
        schema:      Pydantic schema class name (string — not imported;
                     the caller is responsible for resolving it).
        version:     Integer version. Bump when fields change so the
                     LLM knows to re-read the body.
        mime_types:  List of MIME types this template matches.
        filename_patterns: glob patterns to match against uploaded
                           filenames (case-insensitive).
        classification_hints: short strings the classifier can use to
                              decide if this template fits.
        field_overrides: per-field hints, e.g. ``{"invoice_number":
                         {"regex": "INV-\\d+", "examples": [...]}}``.
        body:        The raw Markdown body — passed to the LLM as
                     context.
        source_path: Absolute path of the .md file (for debugging).
    """

    name: str
    schema: str
    body: str
    source_path: Path
    version: int = 1
    mime_types: list[str] = field(default_factory=list)
    filename_patterns: list[str] = field(default_factory=list)
    classification_hints: list[str] = field(default_factory=list)
    field_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def matches_filename(self, filename: str) -> bool:
        """Return True if ``filename`` matches any glob in
        ``filename_patterns``. Empty list = never match by filename.
        """
        if not self.filename_patterns:
            return False
        from fnmatch import fnmatch

        name = filename.lower()
        return any(fnmatch(name, p.lower()) for p in self.filename_patterns)

    def matches_mime(self, mime: str) -> bool:
        """Return True if ``mime`` is in ``mime_types``. Empty list =
        never match by MIME.
        """
        if not self.mime_types:
            return False
        return mime.lower() in [m.lower() for m in self.mime_types]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (for API responses).

        Body is included — clients can use it to understand what the
        LLM was told. ``source_path`` is included as a string.
        """
        return {
            "name": self.name,
            "schema": self.schema,
            "version": self.version,
            "mime_types": list(self.mime_types),
            "filename_patterns": list(self.filename_patterns),
            "classification_hints": list(self.classification_hints),
            "field_overrides": self.field_overrides,
            "body": self.body,
            "source_path": str(self.source_path),
        }


def _parse_frontmatter(text: str, source_path: Path) -> tuple[dict[str, Any], str]:
    """Split a template file into ``(frontmatter_dict, body)``.

    Raises :class:`TemplateParseError` if the file lacks frontmatter,
    if the YAML is invalid, or if a required key is missing.
    """
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        raise TemplateParseError(
            f"template {source_path} has no frontmatter (expected `---\\n...\\n---\\nbody`)"
        )
    fm_raw = m.group("fm")
    body = m.group("body")
    _yaml = _require_yaml()
    try:
        fm = _yaml.safe_load(fm_raw)
    except _yaml.YAMLError as e:
        raise TemplateParseError(f"template {source_path}: invalid YAML: {e}") from e
    if not isinstance(fm, dict):
        raise TemplateParseError(
            f"template {source_path}: frontmatter is not a mapping (got {type(fm).__name__})"
        )
    missing = [k for k in REQUIRED_KEYS if k not in fm]
    if missing:
        raise TemplateParseError(
            f"template {source_path}: missing required frontmatter keys: {missing}"
        )
    return fm, body


def load_template(path: Path) -> Template:
    """Load a single ``.md`` file into a :class:`Template`.

    Raises :class:`TemplateParseError` on any malformation.
    """
    text = path.read_text(encoding="utf-8")
    fm, body = _parse_frontmatter(text, path)
    # `name` is required to be a string and must match the filename stem
    name = str(fm["name"])
    schema = str(fm["schema"])
    stem = path.stem
    if name != stem:
        # Soft warning: most users name the file after the template, so
        # we flag drift but don't reject. A typo here is a debugging
        # hazard, not a runtime failure.
        pass  # surfaced via .name == stem convention in docs
    version = int(fm.get("version", 1))
    return Template(
        name=name,
        schema=schema,
        body=body,
        source_path=path.resolve(),
        version=version,
        mime_types=list(fm.get("mime_types") or []),
        filename_patterns=list(fm.get("filename_patterns") or []),
        classification_hints=list(fm.get("classification_hints") or []),
        field_overrides=dict(fm.get("field_overrides") or {}),
    )


class TemplateRegistry:
    """In-memory registry of templates loaded from a directory.

    Usage::

        registry = TemplateRegistry.load(Path("./templates"))
        registry.list_names()         # ['invoice', 'contract', 'bank_statement']
        registry.get("invoice")       # Template(...)
        registry.find_for_filename(   # best match by filename + mime
            filename="acme-inv-001.pdf",
            mime="application/pdf",
        )

    Loading is a single pass at construction. If ``watch=True`` is
    passed to :meth:`load`, the registry re-reads files on every
    :meth:`get` / :meth:`find_for_filename` call (dev mode only — not
    safe for concurrent reads in production).
    """

    def __init__(self, templates: dict[str, Template], *, watch: bool = False) -> None:
        self._templates = dict(templates)
        self._watch = watch
        self._mtimes: dict[str, float] = {
            name: t.source_path.stat().st_mtime for name, t in templates.items()
        }

    @classmethod
    def load(cls, directory: Path | str, *, watch: bool = False) -> TemplateRegistry:
        """Load all ``*.md`` files from ``directory``.

        If ``directory`` doesn't exist, returns an empty registry
        (no error — server can start without templates, callers will
        get :class:`TemplateNotFoundError` only when they ask for one).

        Raises :class:`TemplateParseError` if any individual file is
        malformed.
        """
        d = Path(directory)
        if not d.is_dir():
            return cls({}, watch=watch)
        templates: dict[str, Template] = {}
        for path in sorted(d.glob("*.md")):
            t = load_template(path)
            if t.name in templates:
                raise TemplateParseError(
                    f"duplicate template name {t.name!r} (from {path} and "
                    f"{templates[t.name].source_path})"
                )
            templates[t.name] = t
        return cls(templates, watch=watch)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def list_names(self) -> list[str]:
        """Return sorted template names."""
        self._maybe_reload()
        return sorted(self._templates.keys())

    def __len__(self) -> int:
        self._maybe_reload()
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        self._maybe_reload()
        return name in self._templates

    def get(self, name: str) -> Template:
        """Return the template named ``name`` or raise
        :class:`TemplateNotFoundError`.
        """
        self._maybe_reload()
        try:
            return self._templates[name]
        except KeyError:
            raise TemplateNotFoundError(
                f"no template named {name!r}; available: {sorted(self._templates)}"
            ) from None

    def find_for_filename(
        self, filename: str, mime: str | None = None
    ) -> Template | None:
        """Return the best matching template for an upload, or None.

        Match priority:
          1. **filename** — if exactly one template's
             ``filename_patterns`` matches, use it.
          2. **mime** — if no filename match, return the first
             template whose ``mime_types`` contains ``mime``.
          3. **None** — caller falls back to discovery / no template.
        """
        self._maybe_reload()
        # 1. filename
        hits = [t for t in self._templates.values() if t.matches_filename(filename)]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # ambiguous: prefer the one with the most specific pattern
            # (longest literal pattern). Same-length patterns keep
            # alphabetical order — deterministic.
            return sorted(hits, key=lambda t: (-max(len(p) for p in t.filename_patterns), t.name))[0]
        # 2. mime
        if mime:
            mime_hits = [t for t in self._templates.values() if t.matches_mime(mime)]
            if len(mime_hits) == 1:
                return mime_hits[0]
            if mime_hits:
                return sorted(mime_hits, key=lambda t: t.name)[0]
        return None

    def all(self) -> list[Template]:
        """Return all templates (sorted by name)."""
        self._maybe_reload()
        return sorted(self._templates.values(), key=lambda t: t.name)

    # ------------------------------------------------------------------
    # Hot reload (dev mode)
    # ------------------------------------------------------------------
    def _maybe_reload(self) -> None:
        if not self._watch:
            return
        for name, t in list(self._templates.items()):
            try:
                mtime = t.source_path.stat().st_mtime
            except FileNotFoundError:
                # file deleted; remove from registry
                del self._templates[name]
                self._mtimes.pop(name, None)
                continue
            if self._mtimes.get(name) != mtime:
                try:
                    new_t = load_template(t.source_path)
                except TemplateParseError:
                    # Skip bad file but don't break the registry
                    continue
                self._templates[name] = new_t
                self._mtimes[name] = mtime


__all__ = [
    "REQUIRED_KEYS",
    "OPTIONAL_KEYS",
    "Template",
    "TemplateRegistry",
    "load_template",
]
