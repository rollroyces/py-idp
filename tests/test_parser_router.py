"""Unit tests for the parser router + auto-pick."""
from idp.core.document import Document
from idp.parse.parser import PlainTextParser, _auto_pick


def test_auto_pick_text_uses_plain(tmp_path):
    f = tmp_path / "foo.txt"
    f.write_text("hi")
    doc = Document.from_path(str(f))
    assert isinstance(_auto_pick(doc), PlainTextParser)


def test_auto_pick_md_uses_plain(tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# hi")
    doc = Document.from_path(str(f))
    assert isinstance(_auto_pick(doc), PlainTextParser)


def test_auto_pick_no_ext_uses_plain(tmp_path):
    f = tmp_path / "Makefile"
    f.write_text("all:")
    doc = Document.from_path(str(f))
    assert isinstance(_auto_pick(doc), PlainTextParser)


def test_document_extension_lowercase(tmp_path):
    f = tmp_path / "Foo.TXT"
    f.write_text("hi")
    doc = Document.from_path(str(f))
    assert doc.extension == "txt"
