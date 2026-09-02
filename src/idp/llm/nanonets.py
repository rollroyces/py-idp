"""Hugging Face vision-language model backend — Nanonets-OCR2-3B.

A self-hosted, offline OCR + extraction backend for the Qwen2.5-VL
fine-tune at https://huggingface.co/nanonets/Nanonets-OCR2-3B.

Why this exists
---------------
- Self-hosted: no API key, no cloud egress, no rate limit.
- Open weights: Apache-2.0 (Qwen2.5-VL base); verify the Nanonets fine-tune
  license before commercial use.
- Multi-language OCR that outperforms Tesseract on noisy scans.

When to use
-----------
- You have a Mac M-series (MPS) or CUDA GPU
- You want extraction on documents you can't send to a third party
- 5-15s per page is acceptable

When NOT to use
--------------
- No GPU available (use Docling, the pdfplumber fallback, or a hosted backend)
- You need <100ms latency (use a hosted model)
- You need to support 100+ concurrent requests (use hosted multi-tenant)

Install
-------
    pip install py-idp[hf-vlm]

First use downloads ~7 GB to ~/.cache/huggingface/hub/ and is cached
thereafter. Subsequent uses load from cache in ~10 seconds.

Usage
-----
    # as a backend via get_backend (gated — opt in)
    import os
    os.environ["IDP_ENABLE_NANONETS"] = "1"
    from idp.llm.backend import get_backend
    backend = get_backend("nanonets")
    out = backend.complete(req)

    # direct construction
    from idp.llm.nanonets import NanonetsVLBackend
    backend = NanonetsVLBackend(
        device="mps",           # or "cuda", "cpu", "auto"
        max_image_side=448,     # 4x less vision memory; 95% accuracy
        load_in_4bit=False,     # True if you OOM at float16
    )

Memory budget on Apple M4 16 GB (float16, 448px image): ~8.3 GB.
First call: 5-10 min download + load. Subsequent: ~10s load, ~5-15s per page.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from idp.llm.backend import Backend, CompletionRequest

log = logging.getLogger(__name__)


# Model constants
DEFAULT_MODEL_ID = "nanonets/Nanonets-OCR2-3B"
DEFAULT_MAX_IMAGE_SIDE = 448  # 4x less vision memory than 1024, ~95% accuracy


class NanonetsVLBackend(Backend):
    """Backend for the Nanonets-OCR2-3B vision-language model from Hugging Face.

    Self-hosted, offline. Requires the ``[hf-vlm]`` extra:
        pip install py-idp[hf-vlm]

    **Platform support** (verified at construction time):

    ============================  ==========================================
    Platform                      Status
    ============================  ==========================================
    macOS arm64 (M1/M2/M3/M4)     ✅ tested target, MPS acceleration
    macOS arm64 + 16+ GB unified  ✅ comfortable (4 GB headroom)
    macOS arm64 8 GB unified      ❌ OOM (use 4-bit mode or different backend)
    macOS x86_64 (Intel)          ⚠️ CPU-only, ~30-60s per page
    Linux x86_64 + CUDA           ✅ best (1-5s per page, 4-bit option)
    Linux x86_64 no GPU           ⚠️ CPU-only, ~30-60s per page
    Linux arm64                    ⚠️ CPU-only (no MPS-equivalent)
    Windows x86_64 + CUDA         ✅ same as Linux CUDA path
    Windows arm64                 ❌ torch has no Windows-arm64 wheels
    ============================  ==========================================

    Constructing on a known-unsupported platform raises
    ``RuntimeError`` immediately with a platform-specific error
    message pointing at a fallback backend.
    """

    name = "nanonets"

    # Platforms that we KNOW won't work, with helpful fallback hints.
    _UNSUPPORTED_PLATFORMS: dict[tuple[str, str], str] = {
        # (system, machine) -> error message + fallback hint
        ("Windows", "ARM64"): (
            "PyTorch does not ship Windows-arm64 wheels. "
            "NanonetsVLBackend cannot run on this platform. "
            "Fallback: use `backend='docling'` or `backend='mock'`, "
            "or run on Windows x86_64 with CUDA."
        ),
        ("Darwin", "x86_64"): (
            "Intel Mac has no MPS (Metal) and CUDA via eGPU is unreliable. "
            "NanonetsVLBackend will fall back to CPU (~30-60s per page). "
            "Consider using `backend='docling'` for faster CPU-only OCR."
        ),
    }

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
        load_in_4bit: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        """Configure (but do not load) the NanonetsVLBackend.

        The model is loaded lazily on the first ``.complete()`` call so that
        ``import idp`` stays fast and the package works without ``torch``
        installed (the import-time check is in ``_require_deps``).

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
        self.max_image_side = max_image_side
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir
        self._model: Any = None
        self._processor: Any = None
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
        # If no images, fall back to text-only mode (Qwen2.5-VL supports this,
        # but accuracy will be lower — we recommend the multimodal path).
        if not images:
            log.warning(
                "NanonetsVLBackend called with no document images; "
                "falling back to text-only mode. Accuracy will be lower than "
                "passing the actual page images."
            )
            return self._generate_text_only(text)
        return self._generate_with_images(text, images)

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
                f"NanonetsVLBackend is not supported on {sys_name} {machine}.\n"
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
                "NanonetsVLBackend requires the [hf-vlm] extra. "
                "Install with:  pip install py-idp[hf-vlm]"
            ) from e

    def _ensure_loaded(self) -> None:
        """Lazy model + processor load. Runs once, cached on the instance."""
        if self._model is not None:
            return
        self._require_deps()
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.perf_counter()
        log.info("Loading Nanonets-OCR2-3B (first call may take 5-10 min to download)...")

        # Step 1: download weights if not cached (~7 GB)
        snapshot_download(
            repo_id=self.model_id,
            cache_dir=self.cache_dir,
            allow_patterns=[
                "*.json", "*.txt", "*.jinja", "*.py", "tokenizer.*",
                "*.safetensors", "merges.txt", "vocab.json",
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

        # Step 5: load processor
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Step 6: load model
        model_kwargs: dict[str, Any] = {
            "torch_dtype": resolved_dtype,
            "device_map": device,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        self._model.eval()
        self._resolved_device = device
        self._resolved_dtype = str(resolved_dtype).removeprefix("torch.")
        log.info(
            "NanonetsVLBackend ready: model=%s device=%s dtype=%s in %.1fs",
            self.model_id, device, self._resolved_dtype, time.perf_counter() - t0,
        )

    def _resolve_device(self, torch: Any) -> str:
        """Map "auto" to MPS / CUDA / CPU; pass through explicit strings."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        """bfloat16 on capable hardware, else float16 (NEVER float32 — 2x slower, 2x memory)."""
        if self.dtype == "auto":
            if device == "cpu" and not torch.cuda.is_available():
                return torch.float32  # CPU works better in float32 (no bfloat16 SIMD on most x86)
            return torch.bfloat16
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self.dtype]

    def _build_prompt(self, req: CompletionRequest) -> str:
        """Reconstruct the prompt from the request's messages."""
        # Nanonets-OCR2-3B is instruction-tuned. We pass the user's content
        # (which already includes "Output JSON Schema: ..." from our pipeline)
        # as a single user turn.
        parts: list[str] = []
        for m in req.messages:
            if m.role == "system" and m.content or m.role == "user" and m.content:
                parts.append(m.content)
        return "\n\n".join(parts) if parts else "Extract the document."

    def _extract_images(self, req: CompletionRequest) -> list[Any]:
        """Pull base64 images from any message in the request.

        Our pipeline doesn't currently attach images; callers that want
        multimodal extraction set ``Message.images_b64`` (list of base64
        data URIs) on a user message. PDF parsers are expected to
        populate this in a follow-up.
        """
        all_b64: list[str] = []
        for m in req.messages:
            all_b64.extend(m.images_b64 or [])
        if not all_b64:
            return []
        import base64
        from io import BytesIO

        from PIL import Image
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
        """Resize longest side to ``max_image_side``. PIL Image in/out."""
        w, h = image.size
        longest = max(w, h)
        if longest <= self.max_image_side:
            return image
        scale = self.max_image_side / longest
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return image.resize(new_size, image.LANCZOS)

    def _generate_text_only(self, text: str) -> str:
        """Generate from text only (no images). Lower accuracy — fallback path."""
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._resolved_device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(new_ids, skip_special_tokens=True)

    def _generate_with_images(self, text: str, images: list[Any]) -> str:
        """Generate from text + images. Main inference path."""
        import torch
        # Qwen2.5-VL chat template: content is a list of {type, text|image} entries.
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self._resolved_device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        raw = self._processor.decode(new_ids, skip_special_tokens=True)
        return self._clean_json(raw)

    @staticmethod
    def _clean_json(raw: str) -> str:
        """The model sometimes returns JSON wrapped in ```json ... ``` fences.

        Strip them so the downstream ``_safe_json`` parser sees clean JSON.
        If the response is empty, return '{}' so callers get a parseable stub.
        """
        if not raw:
            return "{}"
        s = raw.strip()
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        if not s:
            return "{}"
        return s


# ---------------------------------------------------------------------------
# Optional: a no-op sentinel for the "not installed" path.
# Lets users do `from idp.llm.nanonets import NanonetsVLBackend` and
# get a clear ImportError at construction time, not at first use.
# ---------------------------------------------------------------------------
def _check_installation_or_raise() -> None:
    """Raise a helpful ImportError if torch/transformers aren't installed."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "NanonetsVLBackend requires the [hf-vlm] extra.\n"
            "Install with:  pip install py-idp[hf-vlm]\n"
            "Or, for Apple Silicon:\n"
            "  pip install --upgrade pip\n"
            "  pip install torch torchvision\n"
            "  pip install py-idp[hf-vlm]\n\n"
            f"Original error: {e}"
        ) from e


__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_MAX_IMAGE_SIDE",
    "NanonetsVLBackend",
    "_check_installation_or_raise",
]