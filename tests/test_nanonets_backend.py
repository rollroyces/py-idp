"""Tests for the NanonetsVLBackend.

These tests use a mocked model + processor (no torch/transformers required)
so they run in CI. The real-model test path is on a GPU machine.

Coverage:
  - Construction (no model load at __init__)
  - is_multimodal property
  - _build_prompt: text concatenation from messages
  - _clean_json: strip ```json fences, handle empty
  - _preprocess: resize logic (mocked PIL)
  - _extract_images: base64 decode + data-URI handling
  - Missing-deps path: clear ImportError on .complete()
  - Gate path: get_backend("nanonets") raises without IDP_ENABLE_NANONETS
"""
from __future__ import annotations

import base64
import sys
from unittest.mock import MagicMock, patch

import pytest

from idp.llm.backend import CompletionRequest, Message


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def test_construction_does_not_load_model():
    """Construction must be cheap (no model download / load)."""
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    assert b._model is None
    assert b._processor is None
    assert b.name == "nanonets"
    assert b.is_multimodal is True


def test_custom_model_id():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend(model_id="custom/model")
    assert b.model_id == "custom/model"


def test_default_constants():
    from idp.llm.nanonets import (
        DEFAULT_MAX_IMAGE_SIDE,
        DEFAULT_MODEL_ID,
    )
    assert DEFAULT_MODEL_ID == "nanonets/Nanonets-OCR2-3B"
    assert DEFAULT_MAX_IMAGE_SIDE == 448


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def test_build_prompt_concatenates_messages():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    req = CompletionRequest(messages=[
        Message(role="system", content="You are an extractor."),
        Message(role="user", content="Extract this."),
    ])
    prompt = b._build_prompt(req)
    assert "You are an extractor." in prompt
    assert "Extract this." in prompt


def test_build_prompt_skips_empty_messages():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    req = CompletionRequest(messages=[
        Message(role="system", content=""),
        Message(role="user", content="hi"),
    ])
    prompt = b._build_prompt(req)
    assert prompt == "hi"  # empty system skipped


def test_build_prompt_fallback_when_all_empty():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    req = CompletionRequest(messages=[])
    prompt = b._build_prompt(req)
    # empty request -> default placeholder
    assert prompt == "Extract the document."


# ---------------------------------------------------------------------------
# JSON cleaning
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('  {"a": 1}  ', '{"a": 1}'),
    ('', '{}'),
    ('   ', '{}'),
    ('{"a": 1', '{"a": 1'),  # malformed JSON passes through (downstream _safe_json handles)
])
def test_clean_json(raw, expected):
    from idp.llm.nanonets import NanonetsVLBackend
    out = NanonetsVLBackend._clean_json(raw)
    assert out == expected


# ---------------------------------------------------------------------------
# Image preprocessing (with mocked PIL)
# ---------------------------------------------------------------------------
def test_preprocess_no_resize_when_already_small():
    """Images smaller than max_image_side are returned as-is."""
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend(max_image_side=448)
    # Mock PIL Image
    img = MagicMock()
    img.size = (200, 300)  # already small
    out = b._preprocess(img)
    assert out is img  # same object, no resize


def test_preprocess_resize_when_too_large():
    """Images with longest side > max_image_side get resized."""
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend(max_image_side=448)
    img = MagicMock()
    img.size = (2000, 1000)  # longest = 2000, scale to 448
    resized = MagicMock()
    img.resize.return_value = resized
    out = b._preprocess(img)
    img.resize.assert_called_once()
    # new size should be (2000 * 448/2000, 1000 * 448/2000) = (448, 224)
    call_args = img.resize.call_args
    new_size = call_args[0][0]
    assert new_size[0] == 448
    assert new_size[1] == 224
    assert out is resized


# ---------------------------------------------------------------------------
# Image extraction from CompletionRequest
# ---------------------------------------------------------------------------
def _make_b64_png_1x1() -> str:
    """A minimal valid 1x1 transparent PNG as base64."""
    # 1x1 transparent PNG (smallest valid PNG)
    png = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    return base64.b64encode(png).decode()


def test_extract_images_no_images_returns_empty():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    out = b._extract_images(req)
    assert out == []


def test_extract_images_decodes_bare_b64():
    """Bare base64 (no data: URI prefix) decodes successfully."""
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    b64 = _make_b64_png_1x1()
    req = CompletionRequest(messages=[Message(role="user", content="x", images_b64=[b64])])
    mock_img = MagicMock()
    mock_img.size = (1, 1)  # already small
    with patch("PIL.Image.open", return_value=mock_img):
        out = b._extract_images(req)
    assert len(out) == 1


def test_extract_images_decodes_data_uri():
    """data:image/png;base64,XXX format is handled correctly."""
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    b64 = _make_b64_png_1x1()
    data_uri = f"data:image/png;base64,{b64}"
    req = CompletionRequest(messages=[Message(role="user", content="x", images_b64=[data_uri])])
    mock_img = MagicMock()
    mock_img.size = (1, 1)
    with patch("PIL.Image.open", return_value=mock_img):
        out = b._extract_images(req)
    assert len(out) == 1


def test_extract_images_skips_invalid_b64():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    req = CompletionRequest(messages=[Message(role="user", content="x", images_b64=["not-valid-base64!!!"])])
    out = b._extract_images(req)
    assert out == []  # skipped, not crashed


def test_extract_images_aggregates_across_messages():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    b64_1 = _make_b64_png_1x1()
    b64_2 = _make_b64_png_1x1()
    req = CompletionRequest(messages=[
        Message(role="system", content="sys"),
        Message(role="user", content="page 1", images_b64=[b64_1]),
        Message(role="user", content="page 2", images_b64=[b64_2]),
    ])
    mock_img = MagicMock()
    mock_img.size = (1, 1)
    with patch("PIL.Image.open", return_value=mock_img):
        out = b._extract_images(req)
    assert len(out) == 2  # both pages


# ---------------------------------------------------------------------------
# Missing-deps path
# ---------------------------------------------------------------------------
def test_complete_raises_when_torch_missing(monkeypatch):
    """If torch is not installed, .complete() raises a clear ImportError."""
    from idp.llm.nanonets import NanonetsVLBackend

    # Simulate torch not importable
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` fail

    b = NanonetsVLBackend()
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(ImportError) as exc_info:
        b.complete(req)
    msg = str(exc_info.value)
    assert "py-idp[hf-vlm]" in msg


# ---------------------------------------------------------------------------
# Gate path
# ---------------------------------------------------------------------------
def test_get_backend_nanonets_gated_by_default(monkeypatch):
    """get_backend('nanonets') raises without IDP_ENABLE_NANONETS."""
    monkeypatch.delenv("IDP_ENABLE_NANONETS", raising=False)
    from idp.llm.backend import get_backend
    with pytest.raises(ValueError) as exc_info:
        get_backend("nanonets")
    msg = str(exc_info.value)
    assert "gated" in msg
    assert "IDP_ENABLE_NANONETS=1" in msg


def test_get_backend_nanonets_with_gate_enabled(monkeypatch):
    """With IDP_ENABLE_NANONETS=1, get_backend('nanonets') returns NanonetsVLBackend."""
    monkeypatch.setenv("IDP_ENABLE_NANONETS", "1")
    from idp.llm.backend import get_backend
    from idp.llm.nanonets import NanonetsVLBackend
    b = get_backend("nanonets")
    assert isinstance(b, NanonetsVLBackend)
    # No model loaded yet (lazy)
    assert b._model is None


# ---------------------------------------------------------------------------
# Resolved-device / dtype helpers
# ---------------------------------------------------------------------------
def test_resolve_device_explicit_passes_through():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend(device="cpu")
    fake_torch = MagicMock()
    assert b._resolve_device(fake_torch) == "cpu"


def test_resolve_dtype_bfloat16_default_for_capable():
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.bfloat16 = "bfloat16-marker"
    out = b._resolve_dtype(fake_torch, device="cuda")
    assert out == "bfloat16-marker"


def test_resolve_dtype_float32_on_cpu():
    """CPU + auto dtype -> float32 (bfloat16 SIMD is unreliable on x86)."""
    from idp.llm.nanonets import NanonetsVLBackend
    b = NanonetsVLBackend()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    fake_torch.float32 = "float32-marker"
    out = b._resolve_dtype(fake_torch, device="cpu")
    assert out == "float32-marker"
def test_construction_refuses_windows_arm64(monkeypatch):
    """Windows arm64 raises RuntimeError at construction with a fallback hint."""
    from idp.llm.nanonets import NanonetsVLBackend
    with patch("platform.system", return_value="Windows"), \
         patch("platform.machine", return_value="ARM64"), pytest.raises(RuntimeError) as exc_info:
        NanonetsVLBackend()
    msg = str(exc_info.value)
    assert "Windows" in msg
    assert "ARM64" in msg
    # Should mention the fallback
    assert "docling" in msg or "mock" in msg or "fallback" in msg.lower()


def test_construction_warns_intel_mac(monkeypatch):
    """Intel Mac (Darwin x86_64) raises RuntimeError at construction.

    We treat Intel Mac as a hard-fail: no MPS, eGPU CUDA is unreliable.
    Better to fail loud and have the user pick a real backend than to
    silently run on CPU for 30-60s per page.
    """
    from idp.llm.nanonets import NanonetsVLBackend
    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="x86_64"), pytest.raises(RuntimeError) as exc_info:
        NanonetsVLBackend()
    assert "Intel Mac" in str(exc_info.value) or "x86_64" in str(exc_info.value)


def test_construction_succeeds_on_darwin_arm64(monkeypatch):
    """Apple Silicon (M-series) is the tested target and should construct."""
    from idp.llm.nanonets import NanonetsVLBackend
    with patch("platform.system", return_value="Darwin"), \
         patch("platform.machine", return_value="arm64"):
        b = NanonetsVLBackend()
    assert b is not None


def test_construction_succeeds_on_linux_x86_64(monkeypatch):
    """Linux x86_64 is supported (CPU or CUDA; the choice happens at load)."""
    from idp.llm.nanonets import NanonetsVLBackend
    with patch("platform.system", return_value="Linux"), \
         patch("platform.machine", return_value="x86_64"):
        b = NanonetsVLBackend()
    assert b is not None


def test_unsupported_platforms_table_completeness():
    """The _UNSUPPORTED_PLATFORMS table must cover the known-broken cases."""
    from idp.llm.nanonets import NanonetsVLBackend
    keys = set(NanonetsVLBackend._UNSUPPORTED_PLATFORMS.keys())
    # These are the two we explicitly know are broken
    assert ("Windows", "ARM64") in keys
    assert ("Darwin", "x86_64") in keys
    # macOS arm64 must NOT be in the table (it works)
    assert ("Darwin", "arm64") not in keys
    # Linux x86_64 must NOT be in the table (works with or without CUDA)
    assert ("Linux", "x86_64") not in keys
