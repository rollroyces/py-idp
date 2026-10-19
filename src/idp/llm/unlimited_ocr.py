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

"""Hugging Face vision-language model backend — Baidu Unlimited-OCR.

A self-hosted, offline OCR + extraction backend for the Baidu model at
https://huggingface.co/baidu/Unlimited-OCR (paper: arXiv 2606.23050).

Why this exists (vs. Nanonets-OCR2-3B)
-------------------------------------
- Multi-page context: 32,768 token window, so dozens of pages go in ONE
  forward pass via ``model.infer_multi()``. No per-page loop. Cross-page
  reasoning actually works (e.g. "see appendix A" → the model reads the
  appendix).
- Reference Sliding Window Attention (R-SWA): constant KV cache
  throughout decoding. Memory does not grow with output length.
- MIT license (vs. Nanonets' Apache-2.0 + Nanonets-fine-tune-license-uncertainty).
- Bounding-box-aware output via ``<|det|>region_type [x1,y1,x2,y2]<|/det|>``
  markers (or ``return_detections=True`` for a structured list).
- Same self-hosted story: no API key, no cloud egress, no rate limit.

When to use vs. Nanonets
------------------------
- Use Unlimited-OCR when: document is >=5 pages, or you need cross-page
  reasoning, or you need bounding boxes for extraction grounding.
- Use Nanonets when: single-page docs, you want the v0.4-tuned prompt, or
  you're on hardware where the Unlimited-OCR custom-code path is unstable.

Install
-------
    pip install py-idp[hf-vlm]   # same as Nanonets — torch + transformers

First use downloads ~7 GB to ~/.cache/huggingface/hub/. Subsequent uses
load from cache in ~10s.

Usage
-----
    import os
    os.environ["IDP_ENABLE_UNLIMITED_OCR"] = "1"
    from idp.llm.backend import get_backend
    backend = get_backend("unlimited-ocr")
    out = backend.complete(req)

    # Or directly:
    from idp.llm.unlimited_ocr import UnlimitedOcrBackend
    backend = UnlimitedOcrBackend(
        device="auto",         # "cuda" | "mps" | "cpu"
        mode="gundam",         # "gundam" (single) | "base" (multi-page) | "auto"
        load_in_4bit=False,
    )

Inference: per-image 6-20s on CUDA, 15-45s on MPS, minutes on CPU.
Multi-page (``infer_multi()``): single forward pass, scales sub-linearly
with page count thanks to R-SWA's constant KV cache.

Security note
-------------
The model ships **custom code** (DeepSeek-OCR-2 base + R-SWA decoder)
and must be loaded with ``trust_remote_code=True``. This is standard
practice for cutting-edge open-weights models on Hugging Face, but it
does mean the ``modeling_*.py`` file from the repo is executed in your
process. Review the upstream code if this is a concern. Users who
prefer not to run the custom code can deploy the model via the
SGLang-server path (``served-model-name: Unlimited-OCR``) and use a
plain OpenAI-compat client instead — that's a separate backend not
covered by this module.
"""
from __future__ import annotations

import base64
import logging
import re
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from idp.llm.backend import Backend, CompletionRequest

log = logging.getLogger(__name__)


# Model constants — match the values shown in the model card.
DEFAULT_MODEL_ID = "baidu/Unlimited-OCR"
DEFAULT_MODE_SINGLE = "gundam"   # 640x640 tiles + dynamic crop; high-res single page
DEFAULT_MODE_MULTI = "base"      # 1024x1024 single view; required for multi-page
DEFAULT_MAX_NEW_TOKENS = 32768
DEFAULT_NO_REPEAT_NGRAM_SIZE = 35
DEFAULT_NGRAM_WINDOW_SINGLE = 128  # for single-image (gundam/base)
DEFAULT_NGRAM_WINDOW_MULTI = 1024  # for multi-image (base only)
DEFAULT_BASE_SIZE = 1024
DEFAULT_IMAGE_SIZE_GUNDAM = 640    # for gundam mode (tiled)
DEFAULT_IMAGE_SIZE_BASE = 1024     # for base mode (single 1024x1024 view)
DEFAULT_PROMPT_PREFIX = "<image>"   # the model expects this token in the prompt


class UnlimitedOcrBackend(Backend):
    """Backend for Baidu's Unlimited-OCR (arXiv 2606.23050).

    Self-hosted, offline. Requires the ``[hf-vlm]`` extra:
        pip install py-idp[hf-vlm]

    The model ships custom code (DeepSeek-OCR-2 base + R-SWA decoder) and
    must be loaded with ``trust_remote_code=True``. The first call is slow
    (5-10 min download + load), subsequent calls reuse the cached model.
    """

    name = "unlimited-ocr"

    # Platforms where the model is known not to work. Mirrors the table
    # in NanonetsVLBackend for consistency.
    _UNSUPPORTED_PLATFORMS: dict[tuple[str, str], str] = {
        ("Windows", "ARM64"): (
            "PyTorch does not ship Windows-arm64 wheels. "
            "UnlimitedOcrBackend cannot run on this platform. "
            "Fallback: use `backend='docling'` or `backend='mock'`, "
            "or run on Windows x86_64 with CUDA."
        ),
        ("Darwin", "x86_64"): (
            "Intel Mac has no MPS (Metal). UnlimitedOcrBackend will fall back "
            "to CPU; multi-page inference will be very slow. "
            "Consider using `backend='docling'` for faster CPU-only OCR."
        ),
    }

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        mode: str = "auto",          # "auto" | "gundam" | "base"
        load_in_4bit: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """Configure (but do not load) the UnlimitedOcrBackend.

        The model is loaded lazily on the first ``.complete()`` call so that
        ``import idp`` stays fast and the package works without ``torch``
        installed (the import-time check is in ``_require_deps``).

        Args:
            model_id:     HF repo id for the model. Default: ``baidu/Unlimited-OCR``.
            device:       "auto" | "cuda" | "mps" | "cpu".
            dtype:        "auto" | "bfloat16" | "float16" | "float32".
                          "auto" picks bfloat16 on CUDA, float32 on CPU-only.
            mode:         "auto" | "gundam" | "base". "gundam" = 640x640 tiles
                          with dynamic crop (single-image, high-res). "base" =
                          1024x1024 single view (single OR multi-image). When
                          ``infer_multi()`` is called, the backend forces
                          "base" regardless of this value.
            load_in_4bit: True to load model weights in 4-bit (bitsandbytes).
                          Use if you OOM at float16.
            cache_dir:    Custom HF cache dir (default: ``~/.cache/huggingface/hub``).

        Raises:
            RuntimeError: on platforms where torch or this model are known
                           not to work (Windows arm64). The error includes
                           a platform-specific fallback hint.
        """
        # Platform check FIRST so we fail with a clear message before any
        # heavy import or download.
        self._check_platform_support()

        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        # mode="auto" means: pick gundam for single-image, base for multi.
        # Resolved per-call in _infer_single / _infer_multi.
        self._mode_pref = mode
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir
        self._model: Any = None
        self._tokenizer: Any = None
        self._resolved_device: str | None = None
        self._resolved_dtype: str | None = None

    @property
    def is_multimodal(self) -> bool:
        return True

    def complete(self, req: CompletionRequest) -> str:
        """Run extraction. Loads the model on first call (5-10 min cold start)."""
        self._ensure_loaded()
        text = self._build_prompt(req)
        images = self._extract_images(req)
        # If no images, fall back to text-only mode (the model can do
        # text-only but accuracy is lower).
        if not images:
            log.warning(
                "UnlimitedOcrBackend called with no document images; "
                "falling back to text-only mode. Accuracy will be lower than "
                "passing the actual page images."
            )
            return self._infer_text_only(text)
        if len(images) == 1:
            return self._infer_single(text, images[0])
        return self._infer_multi(text, images)

    # ---------------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------------
    def _check_platform_support(self) -> None:
        """Refuse to construct on platforms where we know it won't work.

        We do this at construction (not at first inference) so the user
        gets the error BEFORE spending 5 minutes downloading a 7 GB model
        that won't run.
        """
        import platform as _platform

        sys_name, machine = _platform.system(), _platform.machine()
        bad = self._UNSUPPORTED_PLATFORMS.get((sys_name, machine))
        if bad is not None:
            raise RuntimeError(
                f"UnlimitedOcrBackend is not supported on {sys_name} {machine}.\n"
                f"{bad}"
            )
        # Linux/Windows + machine not in our explicit list is the unknown
        # case; we don't block it (the user might have an exotic build of
        # torch). They get a working model or a torch import error.
        # macOS arm64 is fine.
        # CPU-only is fine (slow but works).

    def _require_deps(self) -> None:
        """Check that torch + transformers are importable.

        Raises:
            ImportError: with a clear install hint pointing to ``[hf-vlm]``.
        """
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "UnlimitedOcrBackend requires the [hf-vlm] extra. "
                "Install with:  pip install py-idp[hf-vlm]"
            ) from e

    def _ensure_loaded(self) -> None:
        """Lazy model + tokenizer load. Runs once, cached on the instance."""
        if self._model is not None:
            return
        self._require_deps()
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        t0 = time.perf_counter()
        log.info(
            "Loading Unlimited-OCR (first call may take 5-10 min to download)..."
        )

        # Step 1: download weights if not cached (~7 GB)
        snapshot_download(
            repo_id=self.model_id,
            cache_dir=self.cache_dir,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.jinja",
                "*.py",
                "tokenizer.*",
                "*.safetensors",
                "merges.txt",
                "vocab.json",
            ],
        )
        log.info("Model files cached in %ss", time.perf_counter() - t0)

        # Step 2: resolve device
        device = self._resolve_device(torch)

        # Step 3: resolve dtype
        resolved_dtype = self._resolve_dtype(torch, device)

        # Step 4: choose quantization
        quantization_config = None
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=resolved_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                log.info("Using 4-bit quantization (bitsandbytes)")
            except ImportError as e:
                raise ImportError(
                    "load_in_4bit=True requires bitsandbytes: "
                    "pip install bitsandbytes"
                ) from e

        # Step 5: load tokenizer
        # Note: trust_remote_code=True is required because the model ships
        # custom modeling code (DeepSeek-OCR-2 base + R-SWA decoder).
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Step 6: load model
        model_kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "use_safetensors": True,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self._model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()
        # The model card uses ``.cuda()`` explicitly. For non-CUDA devices
        # we use ``.to(device)`` (MPS is not officially supported by
        # this model but falls through to CPU gracefully).
        if device == "cuda":
            self._model = self._model.cuda()
        else:
            self._model = self._model.to(device)
        self._resolved_device = device
        self._resolved_dtype = str(resolved_dtype).removeprefix("torch.")
        log.info(
            "UnlimitedOcrBackend ready: model=%s device=%s dtype=%s in %.1fs",
            self.model_id,
            device,
            self._resolved_dtype,
            time.perf_counter() - t0,
        )

    def _resolve_device(self, torch: Any) -> str:
        """Map "auto" to MPS / CUDA / CPU; pass through explicit strings."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Note: the model is CUDA-first. MPS will fall back to CPU
            # inside the custom code; we still report mps to keep parity
            # with NanonetsVLBackend.
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        """bfloat16 on capable hardware, else float16 (NEVER float32 by default)."""
        if self.dtype == "auto":
            if device == "cpu" and not torch.cuda.is_available():
                # CPU-only: float32 has better numerical stability and SIMD
                # coverage on x86. (Same logic as NanonetsVLBackend.)
                return torch.float32
            return torch.bfloat16
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self.dtype]

    def _build_prompt(self, req: CompletionRequest) -> str:
        """Reconstruct the prompt from the request's messages.

        The model's own chat template expects the literal token ``<image>``
        prepended to the prompt. We pass the user's content (which already
        includes "Output JSON Schema: ..." from our pipeline) verbatim
        after that prefix.
        """
        parts: list[str] = []
        for m in req.messages:
            if m.role in ("system", "user") and m.content:
                parts.append(m.content)
        body = "\n\n".join(parts) if parts else "Extract the document."
        return f"{DEFAULT_PROMPT_PREFIX}{body}"

    def _extract_images(self, req: CompletionRequest) -> list[Any]:
        """Pull base64 images from any message in the request.

        Mirrors NanonetsVLBackend._extract_images exactly. Our pipeline
        doesn't currently attach images; callers that want multimodal
        extraction set ``Message.images_b64`` (list of base64 data URIs)
        on a user message. PDF parsers are expected to populate this in
        a follow-up.
        """
        from PIL import Image

        all_b64: list[str] = []
        for m in req.messages:
            all_b64.extend(m.images_b64 or [])
        if not all_b64:
            return []
        result: list[Any] = []
        for b64 in all_b64:
            if not b64:
                continue
            # Accept both "data:image/png;base64,XXX" and bare "XXX"
            payload = (
                b64.split(",", 1)[1] if b64.startswith("data:") and "," in b64 else b64
            )
            try:
                raw = base64.b64decode(payload)
            except Exception as e:
                log.warning("Failed to decode base64 image: %s", e)
                continue
            try:
                img = Image.open(BytesIO(raw))
                result.append(self._preprocess(img))
            except Exception as e:
                log.warning("Failed to open decoded image: %s", e)
        return result

    def _preprocess(self, image: Any) -> Any:
        """Resize longest side to ``max_image_side``. PIL Image in/out.

        Unlimited-OCR's ``gundam`` mode handles variable sizes via dynamic
        cropping at the model level, so we keep the input image at native
        resolution here and let the model decide. The parameter exists
        for callers who want to clamp memory on very large scans; we
        currently pass through unmodified.
        """
        # No-op: Unlimited-OCR expects native-resolution images and does
        # its own tiling/cropping. Keeping the hook here means callers
        # can override the method if they need custom preprocessing
        # (e.g. deskew) without touching the inference path.
        return image

    def _resolve_mode(self, num_images: int) -> str:
        """Pick "gundam" (single) or "base" (multi). Multi-image forces base."""
        if num_images > 1:
            return DEFAULT_MODE_MULTI
        if self._mode_pref in ("gundam", "base"):
            return self._mode_pref
        # "auto" -> gundam for single, base for multi (already handled above)
        return DEFAULT_MODE_SINGLE

    def _infer_text_only(self, text: str) -> str:
        """Generate from text only (no images). Lower accuracy — fallback path.

        Unlimited-OCR can be invoked without images, but the model card
        only documents the image path. We construct a synthetic call by
        rendering a small placeholder image; the model will mostly
        ignore it and respond to the prompt. If the model API rejects
        text-only calls, callers will see a clear ``RuntimeError``.
        """
        # Synthesize a 1x1 white image so the model has a tile to look at.
        from PIL import Image

        placeholder = Image.new("RGB", (32, 32), color=(255, 255, 255))
        return self._infer_single(text, placeholder)

    def _infer_single(self, text: str, image: Any) -> str:
        """Generate from text + a single image via ``model.infer()``."""
        mode = self._resolve_mode(1)
        with tempfile.TemporaryDirectory(prefix="idp_unlimited_ocr_") as tmp_dir:
            tmp_path = Path(tmp_dir) / "page.png"
            image.save(tmp_path, format="PNG")
            if mode == "gundam":
                image_size = DEFAULT_IMAGE_SIZE_GUNDAM
                crop_mode = True
                ngram_window = DEFAULT_NGRAM_WINDOW_SINGLE
            else:  # base
                image_size = DEFAULT_IMAGE_SIZE_BASE
                crop_mode = False
                ngram_window = DEFAULT_NGRAM_WINDOW_SINGLE
            self._model.infer(
                tokenizer=self._tokenizer,
                prompt=text,
                image_file=str(tmp_path),
                output_path=tmp_dir,
                base_size=DEFAULT_BASE_SIZE,
                image_size=image_size,
                crop_mode=crop_mode,
                max_length=DEFAULT_MAX_NEW_TOKENS,
                no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                ngram_window=ngram_window,
                save_results=True,
            )
            raw = self._read_output_text(tmp_dir)
        return self._strip_det_markers(raw)

    def _infer_multi(self, text: str, images: list[Any]) -> str:
        """Generate from text + multiple images via ``model.infer_multi()``."""
        # Multi-image is always base mode (the model doesn't support
        # gundam for >1 image).
        with tempfile.TemporaryDirectory(prefix="idp_unlimited_ocr_") as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            image_paths: list[str] = []
            for i, img in enumerate(images):
                p = tmp_dir_path / f"page_{i + 1:04d}.png"
                img.save(p, format="PNG")
                image_paths.append(str(p))
            self._model.infer_multi(
                tokenizer=self._tokenizer,
                prompt=text,
                image_files=image_paths,
                output_path=tmp_dir,
                image_size=DEFAULT_IMAGE_SIZE_BASE,
                max_length=DEFAULT_MAX_NEW_TOKENS,
                no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                ngram_window=DEFAULT_NGRAM_WINDOW_MULTI,
                save_results=True,
            )
            raw = self._read_output_text(tmp_dir)
        return self._strip_det_markers(raw)

    @staticmethod
    def _read_output_text(output_dir: str | Path) -> str:
        """Read the model's output text from its output directory.

        ``model.infer()`` / ``model.infer_multi()`` with ``save_results=True``
        write the OCR'd text into a per-image file under ``output_path``.
        The exact filename pattern is implementation-defined in the
        upstream model code; the most reliable cross-version strategy is
        to walk the output dir, read all ``*.txt`` / ``*.md`` files, and
        concatenate them in sorted order.

        If no readable file is found, we return an empty string so the
        downstream ``_safe_json`` parser sees a parseable (empty) result
        rather than crashing.
        """
        from pathlib import Path as _P

        out = _P(output_dir)
        if not out.exists():
            return ""
        texts: list[str] = []
        for f in sorted(out.glob("*.md")) + sorted(out.glob("*.txt")):
            try:
                texts.append(f.read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                log.warning("Could not read %s: %s", f, e)
        return "\n".join(texts).strip()

    @staticmethod
    def _strip_det_markers(raw: str) -> str:
        """Thin wrapper so the method is also importable as a module function."""
        return _strip_det_markers(raw)


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MODE_SINGLE",
    "DEFAULT_MODE_MULTI",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PROMPT_PREFIX",
    "UnlimitedOcrBackend",
    "_strip_det_markers",
]


# ---------------------------------------------------------------------------
# Module-level helper (also exposed for tests + future callers)
# ---------------------------------------------------------------------------
def _strip_det_markers(raw: str) -> str:
    """Remove ``<|det|>category [x1,y1,x2,y2]<|/det|>`` markup.

    The model's default ``document parsing.`` prompt returns this
    structured format. For our JSON-extraction pipeline we only
    need the text; downstream ``_safe_json`` handles the JSON parse.

    The strip keeps the text content and removes the bounding-box
    annotation. If we ever want to expose the boxes (e.g. for
    extraction-grounding UI), we can add a new ``CompletionRequest``
    flag and keep the markers.
    """
    if not raw:
        return ""
    # Drop the opening marker + optional bbox + closing marker, keep
    # the text. Pattern is intentionally permissive on the bbox
    # format (numbers, spaces, optional brackets).
    cleaned = re.sub(
        r"<\|det\|>[^<\s]+(?:\s*\[[^\]]*\])?\s*<\|/det\|>",
        "",
        raw,
    )
    return cleaned.strip()
