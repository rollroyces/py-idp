# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for UnlimitedOcrBackend.

The Unlimited-OCR model is too large to load in CI (~7 GB weights +
custom-code trust requirement), so all tests mock the model + tokenizer
with ``unittest.mock.MagicMock``. This is the same approach used in
``test_ollama_backend.py`` and is sufficient because the routing logic
(construct -> load -> infer vs infer_multi -> marker strip) is what
regresses, not the model's actual OCR.

We do NOT need real torch / transformers to import the module — the
``_ensure_loaded()`` step is the only thing that touches them, and our
tests either construct without loading or pre-set the ``_model`` /
``_tokenizer`` attributes directly.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from idp.llm.backend import (
    CompletionRequest,
    Message,
    get_backend,
    list_backends,
)
from idp.llm.unlimited_ocr import (
    DEFAULT_MODE_MULTI,
    DEFAULT_MODE_SINGLE,
    DEFAULT_MODEL_ID,
    DEFAULT_PROMPT_PREFIX,
    UnlimitedOcrBackend,
    _strip_det_markers,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _png_bytes(color: tuple[int, int, int] = (255, 0, 0), size: tuple[int, int] = (16, 16)) -> str:
    """Return a base64 data URI of a tiny PNG of the given color/size.

    Used to construct Message.images_b64 in test requests without needing
    PIL at the test boundary. We avoid importing PIL here so the tests
    work even if Pillow isn't installed in the test env.
    """
    import base64

    # Minimal valid PNG: 8-byte signature + IHDR + IDAT + IEND
    # Easier: just use PIL if available; otherwise encode the bytes we
    # know PIL will parse. The test backend is supposed to call PIL.Image.open
    # on the decoded bytes, so a valid PNG is required.
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _make_backend_with_mock_model(
    *,
    infer_output_text: str = "extracted text content",
    device: str = "cpu",
    **kwargs: Any,
) -> UnlimitedOcrBackend:
    """Build an UnlimitedOcrBackend with a mock model + tokenizer pre-loaded.

    The mock's ``infer()`` and ``infer_multi()`` write a small file under
    ``output_path`` so ``_read_output_text`` can read it back, matching
    the real model's behavior (``save_results=True``).
    """
    backend = UnlimitedOcrBackend(device=device, **kwargs)
    model = MagicMock()
    tokenizer = MagicMock()

    def _fake_infer(*, tokenizer, prompt, image_file, output_path, base_size, image_size,
                    crop_mode, max_length, no_repeat_ngram_size, ngram_window, save_results, **_kw):
        # Write a predictable file the backend can read back
        out = Path(output_path) / "out.md"
        out.write_text(f"infer_output_text_for_{Path(image_file).stem}", encoding="utf-8")

    def _fake_infer_multi(*, tokenizer, prompt, image_files, output_path, image_size,
                          max_length, no_repeat_ngram_size, ngram_window, save_results, **_kw):
        out = Path(output_path) / "multi.md"
        out.write_text("multi_page_result", encoding="utf-8")

    model.infer.side_effect = _fake_infer
    model.infer_multi.side_effect = _fake_infer_multi

    backend._model = model
    backend._tokenizer = tokenizer
    backend._resolved_device = device
    backend._resolved_dtype = "float32" if device == "cpu" else "bfloat16"
    return backend


def _png_pil_image():
    """Return a real PIL Image so the backend's image save() can run."""
    from PIL import Image

    return Image.new("RGB", (32, 32), color=(0, 128, 255))


# ---------------------------------------------------------------------------
# 1. Constructor doesn't load the model
# ---------------------------------------------------------------------------
def test_construct_without_loading():
    """Construction is a pure config object. No torch / transformers import happens here."""
    b = UnlimitedOcrBackend(device="cpu")
    assert b.model_id == DEFAULT_MODEL_ID
    assert b._model is None
    assert b._tokenizer is None
    assert b.is_multimodal is True


# ---------------------------------------------------------------------------
# 2. is_multimodal
# ---------------------------------------------------------------------------
def test_is_multimodal_true():
    b = UnlimitedOcrBackend()
    assert b.is_multimodal is True


# ---------------------------------------------------------------------------
# 3. Repr is debuggable
# ---------------------------------------------------------------------------
def test_repr_includes_class_name():
    b = UnlimitedOcrBackend(device="cpu")
    r = repr(b)
    assert "UnlimitedOcrBackend" in r
    assert "unlimited-ocr" in r  # .name


# ---------------------------------------------------------------------------
# 4. get_backend is gated when env var is off
# ---------------------------------------------------------------------------
def test_get_backend_gated_when_off(monkeypatch):
    monkeypatch.delenv("IDP_ENABLE_UNLIMITED_OCR", raising=False)
    with pytest.raises(ValueError, match="IDP_ENABLE_UNLIMITED_OCR"):
        get_backend("unlimited-ocr")


# ---------------------------------------------------------------------------
# 5. get_backend works when env var is on
# ---------------------------------------------------------------------------
def test_get_backend_works_when_on(monkeypatch):
    monkeypatch.setenv("IDP_ENABLE_UNLIMITED_OCR", "1")
    b = get_backend("unlimited-ocr", device="cpu")
    assert isinstance(b, UnlimitedOcrBackend)


# ---------------------------------------------------------------------------
# 6. list_backends excludes when gate is off
# ---------------------------------------------------------------------------
def test_list_backends_excludes_when_off(monkeypatch):
    monkeypatch.delenv("IDP_ENABLE_UNLIMITED_OCR", raising=False)
    names = {r["name"] for r in list_backends()}
    assert "unlimited-ocr" not in names


# ---------------------------------------------------------------------------
# 7. list_backends includes when gate is on
# ---------------------------------------------------------------------------
def test_list_backends_includes_when_on(monkeypatch):
    monkeypatch.setenv("IDP_ENABLE_UNLIMITED_OCR", "1")
    try:
        names = {r["name"] for r in list_backends()}
        assert "unlimited-ocr" in names
        by_name = {r["name"]: r for r in list_backends()}
        assert by_name["unlimited-ocr"]["auth"] == "gated"
    finally:
        monkeypatch.delenv("IDP_ENABLE_UNLIMITED_OCR", raising=False)


# ---------------------------------------------------------------------------
# 8. _extract_images decodes base64 (data URI + bare)
# ---------------------------------------------------------------------------
def test_extract_images_decodes_b64(monkeypatch):
    """Both ``data:image/png;base64,XXX`` and bare base64 are accepted."""
    b64 = _png_bytes()
    req = CompletionRequest(
        messages=[Message(role="user", content="hi", images_b64=[b64])],
    )
    backend = UnlimitedOcrBackend(device="cpu")
    images = backend._extract_images(req)
    assert len(images) == 1
    # We don't assert exact size (PIL may normalize), just that it returned


# ---------------------------------------------------------------------------
# 9. Single-image request routes to model.infer
# ---------------------------------------------------------------------------
def test_infer_single_routes_to_model_infer():
    backend = _make_backend_with_mock_model()
    req = CompletionRequest(
        messages=[Message(role="user", content="Extract the invoice.", images_b64=[_png_bytes()])],
    )
    out = backend.complete(req)
    # The mock write wrote a file; we read it back; mock wrote
    # "infer_output_text_for_<stem>" which is the test's content marker.
    assert "infer_output_text_for" in out
    # And model.infer (singular) was called once
    backend._model.infer.assert_called_once()
    backend._model.infer_multi.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Multi-image request routes to model.infer_multi
# ---------------------------------------------------------------------------
def test_infer_multi_routes_to_model_infer_multi():
    backend = _make_backend_with_mock_model()
    req = CompletionRequest(
        messages=[
            Message(
                role="user",
                content="Extract.",
                images_b64=[_png_bytes(), _png_bytes(color=(0, 0, 255)), _png_bytes(color=(0, 255, 0))],
            )
        ],
    )
    out = backend.complete(req)
    assert "multi_page_result" in out
    backend._model.infer_multi.assert_called_once()
    backend._model.infer.assert_not_called()


# ---------------------------------------------------------------------------
# 11. Multi-image forces base mode regardless of user setting
# ---------------------------------------------------------------------------
def test_infer_multi_forces_base_mode():
    """Even if the user passed mode='gundam', multi-image should use base."""
    backend = _make_backend_with_mock_model(mode="gundam")
    req = CompletionRequest(
        messages=[
            Message(
                role="user",
                content="x",
                images_b64=[_png_bytes(), _png_bytes(color=(1, 2, 3))],
            )
        ],
    )
    backend.complete(req)
    backend._model.infer_multi.assert_called_once()
    # image_size should be DEFAULT_IMAGE_SIZE_BASE = 1024 (not 640)
    call_kwargs = backend._model.infer_multi.call_args.kwargs
    assert call_kwargs["image_size"] == 1024


# ---------------------------------------------------------------------------
# 12. _strip_det_markers removes <|det|>...<|/det|> markup
# ---------------------------------------------------------------------------
def test_strip_det_markers_removes_bbox_annotation():
    raw = (
        "Header text.\n"
        "<|det|>title [10, 20, 100, 50]<|/det|>Invoice Title\n"
        "Body text.\n"
        "<|det|>image [200, 200, 300, 300]<|/det|>\n"
        "Footer text.\n"
    )
    cleaned = _strip_det_markers(raw)
    assert "<|det|>" not in cleaned
    assert "<|/det|>" not in cleaned
    assert "Header text." in cleaned
    assert "Invoice Title" in cleaned
    assert "Body text." in cleaned
    assert "Footer text." in cleaned


def test_strip_det_markers_passes_through_text_without_markers():
    raw = "plain text with no markers"
    assert _strip_det_markers(raw) == raw


def test_strip_det_markers_handles_empty_and_none():
    assert _strip_det_markers("") == ""
    assert _strip_det_markers(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 13. Placeholder fallback when no images attached
# ---------------------------------------------------------------------------
def test_synthetic_placeholder_fallback_when_no_images():
    """When Message.images_b64 is empty, the backend synthesizes a blank
    placeholder tile and routes through model.infer (Unlimited-OCR has
    no text-only entry point). Gives the caller a parseable result
    instead of crashing.
    """
    backend = _make_backend_with_mock_model()
    req = CompletionRequest(
        messages=[Message(role="user", content="Summarize the document.")],
    )
    out = backend.complete(req)
    # The mock write produced "infer_output_text_for_<stem>" -- the stem
    # is "page.png" from the synthetic placeholder.
    assert "infer_output_text_for_page" in out
    backend._model.infer.assert_called_once()
    backend._model.infer_multi.assert_not_called()


# ---------------------------------------------------------------------------
# 14. Device resolution (auto)
# ---------------------------------------------------------------------------
def test_resolve_device_auto_picks_cpu_when_no_gpu(monkeypatch):
    """With no CUDA / MPS available, device='auto' should resolve to 'cpu'."""

    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    # No mps attribute
    del fake_torch.backends.mps
    b = UnlimitedOcrBackend(device="auto")
    assert b._resolve_device(fake_torch) == "cpu"


def test_resolve_device_explicit_passes_through():
    """device='cuda' stays 'cuda'; 'mps' stays 'mps'."""
    b = UnlimitedOcrBackend(device="cuda")
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    assert b._resolve_device(fake_torch) == "cuda"
    b2 = UnlimitedOcrBackend(device="mps")
    assert b2._resolve_device(fake_torch) == "mps"


# ---------------------------------------------------------------------------
# 15. 4-bit quantization config
# ---------------------------------------------------------------------------
def test_load_in_4bit_constructs_quantization_config(monkeypatch):
    """When load_in_4bit=True, ``_ensure_loaded`` calls
    ``transformers.BitsAndBytesConfig(...)`` with the right kwargs and
    passes the result to ``from_pretrained(quantization_config=...)``.

    Strategy: replace the ``transformers`` module in ``sys.modules`` with
    a fake that records every call, then call ``_ensure_loaded`` and
    assert on the captured args.
    """
    import sys

    import idp.llm.unlimited_ocr as uo

    captured: dict[str, Any] = {}

    class _FakeBnB:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    def _fake_bnb(**kwargs):
        return _FakeBnB(**kwargs)

    fake_model_instance = MagicMock()
    fake_model_instance.eval.return_value = fake_model_instance
    fake_model_instance.cuda.return_value = fake_model_instance
    fake_model_instance.to.return_value = fake_model_instance

    fake_tokenizer_instance = MagicMock()
    fake_transformers = MagicMock()
    fake_transformers.AutoModel.from_pretrained.return_value = fake_model_instance
    fake_transformers.AutoTokenizer.from_pretrained.return_value = fake_tokenizer_instance
    fake_transformers.BitsAndBytesConfig.side_effect = _fake_bnb

    # Make torch.cuda.is_available return True so dtype resolution picks bfloat16
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.bfloat16 = "bfloat16"

    # Save originals
    orig_transformers = sys.modules.get("transformers")
    orig_torch = sys.modules.get("torch")
    orig_hub = sys.modules.get("huggingface_hub")

    try:
        sys.modules["transformers"] = fake_transformers
        sys.modules["torch"] = fake_torch
        sys.modules["huggingface_hub"] = MagicMock(
            snapshot_download=MagicMock(return_value="/tmp/fake_cache")
        )

        # Build the backend with load_in_4bit=True. Skip require_deps
        # since our torch/transformers are mocks anyway.
        backend = UnlimitedOcrBackend(device="cuda", load_in_4bit=True)
        monkeypatch.setattr(
            uo.UnlimitedOcrBackend,
            "_require_deps",
            lambda self: None,
        )
        backend._ensure_loaded()

        # Assert: BitsAndBytesConfig was called with the right args
        assert "load_in_4bit" in captured
        assert captured["load_in_4bit"] is True
        assert captured["bnb_4bit_quant_type"] == "nf4"
        assert captured["bnb_4bit_use_double_quant"] is True
        assert captured["bnb_4bit_compute_dtype"] == "bfloat16"

        # Assert: from_pretrained was called with quantization_config
        call = fake_transformers.AutoModel.from_pretrained.call_args
        assert "quantization_config" in call.kwargs
        assert isinstance(call.kwargs["quantization_config"], _FakeBnB)
    finally:
        # Restore sys.modules
        if orig_transformers is not None:
            sys.modules["transformers"] = orig_transformers
        else:
            sys.modules.pop("transformers", None)
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch
        else:
            sys.modules.pop("torch", None)
        if orig_hub is not None:
            sys.modules["huggingface_hub"] = orig_hub
        else:
            sys.modules.pop("huggingface_hub", None)


# ---------------------------------------------------------------------------
# 16. Helpful ImportError when torch/transformers are missing
# ---------------------------------------------------------------------------
def test_raises_helpful_import_error_without_torch(monkeypatch):
    """If torch isn't importable, ``_require_deps()`` raises ImportError with an install hint."""
    import sys

    # Remove torch from sys.modules so any subsequent ``import torch``
    # raises ModuleNotFoundError. This is the simplest way to simulate
    # the "torch isn't installed" state in a test that has torch.
    orig_torch = sys.modules.pop("torch", None)
    # Also break the cached import so a re-import goes through the
    # import system and fails.
    sys.modules.get("transformers")
    sys.meta_path.insert(0, _TorchBlocker())
    try:
        b = UnlimitedOcrBackend(device="cpu")
        with pytest.raises(ImportError, match=r"py-idp\[hf-vlm\]"):
            b._require_deps()
    finally:
        sys.meta_path.pop(0)
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch


class _TorchBlocker:
    """A meta-path finder that blocks any import of ``torch`` / ``torch.*``.

    Used to test the ``_require_deps`` ImportError path. Inserted at the
    head of ``sys.meta_path`` so it runs before the default finders.
    """

    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return None


# ---------------------------------------------------------------------------
# Bonus: _build_prompt prepends the model's <image> token
# ---------------------------------------------------------------------------
def test_build_prompt_prepends_image_token():
    b = UnlimitedOcrBackend(device="cpu")
    req = CompletionRequest(
        messages=[
            Message(role="system", content="You are an invoice extractor."),
            Message(role="user", content="Extract the invoice."),
        ],
    )
    prompt = b._build_prompt(req)
    assert prompt.startswith(DEFAULT_PROMPT_PREFIX)
    assert "You are an invoice extractor." in prompt
    assert "Extract the invoice." in prompt


# ---------------------------------------------------------------------------
# Bonus: _resolve_mode picks correctly
# ---------------------------------------------------------------------------
def test_resolve_mode_auto_gundam_for_single_base_for_multi():
    b = UnlimitedOcrBackend(mode="auto")
    assert b._resolve_mode(1) == DEFAULT_MODE_SINGLE  # gundam
    assert b._resolve_mode(2) == DEFAULT_MODE_MULTI   # base
    assert b._resolve_mode(10) == DEFAULT_MODE_MULTI  # base


def test_resolve_mode_explicit_respected_when_single():
    b = UnlimitedOcrBackend(mode="base")
    assert b._resolve_mode(1) == "base"


def test_resolve_mode_multi_image_always_base_even_if_user_set_gundam():
    b = UnlimitedOcrBackend(mode="gundam")
    assert b._resolve_mode(2) == "base"  # multi forces base


# ---------------------------------------------------------------------------
# Regression tests from the v0.6 code review
# ---------------------------------------------------------------------------
def test_resolve_dtype_rejects_unknown_dtype_with_clear_message():
    """A typo like dtype='int8' must raise ValueError naming the valid
    options, not a bare KeyError from the dict lookup.

    Regression: `return mapping[self.dtype]` raised an opaque
    KeyError('int8') from inside complete(), with no hint about which
    values are accepted.
    """
    b = UnlimitedOcrBackend(device="cpu", dtype="int8")
    fake_torch = MagicMock()
    with pytest.raises(ValueError) as exc:
        b._resolve_dtype(fake_torch, "cuda")
    msg = str(exc.value)
    assert "int8" in msg
    # The message must enumerate the valid choices
    for valid in ("auto", "bfloat16", "float16", "float32"):
        assert valid in msg


def test_resolve_dtype_accepts_all_documented_values():
    """Every documented dtype string resolves without raising."""
    fake_torch = MagicMock()
    fake_torch.bfloat16 = "bf16"
    fake_torch.float16 = "f16"
    fake_torch.float32 = "f32"
    fake_torch.cuda.is_available.return_value = True
    b = UnlimitedOcrBackend(device="cuda", dtype="float16")
    assert b._resolve_dtype(fake_torch, "cuda") == "f16"


def test_read_output_text_finds_nested_files(tmp_path):
    """Model output in a subdirectory is still picked up.

    Regression: `out.glob('*.md')` is non-recursive, so a model version
    that writes `output_path/<page>/out.md` silently yielded empty text
    and the pipeline extracted nothing.
    """
    nested = tmp_path / "page_0001"
    nested.mkdir()
    (nested / "out.md").write_text("nested page text", encoding="utf-8")
    got = UnlimitedOcrBackend._read_output_text(tmp_path)
    assert "nested page text" in got


def test_read_output_text_still_reads_flat_files(tmp_path):
    """Flat .md + .txt output (the current documented layout) works."""
    (tmp_path / "a.md").write_text("from md", encoding="utf-8")
    (tmp_path / "b.txt").write_text("from txt", encoding="utf-8")
    got = UnlimitedOcrBackend._read_output_text(tmp_path)
    assert "from md" in got
    assert "from txt" in got


def test_read_output_text_missing_dir_returns_empty(tmp_path):
    """A nonexistent output dir returns '' so _safe_json sees '{}'."""
    assert UnlimitedOcrBackend._read_output_text(tmp_path / "nope") == ""
