"""Configuration, model defaults, thresholds, and device auto-detection."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pathlib import Path
from dotenv import load_dotenv

if TYPE_CHECKING:
    import torch

_ENV_PATH = Path(__file__).resolve().parent / ".env"
_ENV_EXAMPLE_PATH = Path(__file__).resolve().parent / ".env.example"
load_dotenv(dotenv_path=_ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# Model & API Configuration
# ---------------------------------------------------------------------------
TROCR_MODEL_NAME: str = os.getenv(
    "TROCR_MODEL_NAME", "microsoft/trocr-base-handwritten"
)
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-lite")
GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
)
GEMINI_FAST_GRADING_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
)
LOCAL_VLM_MODEL_NAME: str = os.getenv(
    "LOCAL_VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"
)
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

_CLIENT_LOCK = threading.Lock()
_GEMINI_CLIENT_CACHE: dict[tuple[str, int], Any] = {}
_COOLDOWN_LOCK = threading.Lock()
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}


def _ordered_healthy_models(models: tuple[str, ...] | list[str]) -> list[str]:
    """Return deduplicated models prioritizing healthy endpoints not currently in 503/rate cooldown."""
    unique = list(dict.fromkeys(models))
    now = time.monotonic()
    with _COOLDOWN_LOCK:
        ready = [m for m in unique if _MODEL_COOLDOWN_UNTIL.get(m, 0.0) <= now]
        cooling = sorted(
            [m for m in unique if m not in ready],
            key=lambda m: _MODEL_COOLDOWN_UNTIL.get(m, 0.0),
        )
    return ready + cooling


def _mark_model_transient_cooldown(model_name: str, exc: Exception) -> None:
    """Put transient 503/504/429 models on a brief cooldown so parallel calls skip slow stalls."""
    msg = str(exc).upper()
    if any(tok in msg for tok in ("503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED", "429", "RESOURCE_EXHAUSTED")):
        ttl = 90.0
        with _COOLDOWN_LOCK:
            _MODEL_COOLDOWN_UNTIL[model_name] = time.monotonic() + ttl


def get_gemini_client(api_key: str, timeout_ms: int = 15000) -> Any:
    """Return a cached thread-safe Gemini Client configured for fast fail-over without retry backoff stalls."""
    eff_timeout = max(10500, int(timeout_ms))
    cache_key = (api_key, eff_timeout)
    with _CLIENT_LOCK:
        client = _GEMINI_CLIENT_CACHE.get(cache_key)
        if client is None:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=eff_timeout,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            _GEMINI_CLIENT_CACHE[cache_key] = client
        return client


def build_fast_gemini_config(model_name: str, response_schema: Any = None) -> Any:
    """Build low-latency structured JSON config with optimal zero/minimal thinking per Flash-Lite variant."""
    from google.genai import types

    kwargs: dict[str, Any] = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }
    if response_schema is not None:
        kwargs["response_schema"] = response_schema

    lower_m = (model_name or "").lower()
    if "3.5-flash-lite" in lower_m:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")

    return types.GenerateContentConfig(**kwargs)


def generate_content_fast(
    api_key: str,
    models: tuple[str, ...] | list[str],
    contents: Any,
    response_schema: Any = None,
    timeout_ms: int = 14000,
) -> Any:
    """Execute Gemini generate_content using adaptive cooldown routing across Flash-Lite models."""
    client = get_gemini_client(api_key, timeout_ms=timeout_ms)
    last_err: Exception | None = None
    for m_name in _ordered_healthy_models(models):
        try:
            return client.models.generate_content(
                model=m_name,
                contents=contents,
                config=build_fast_gemini_config(m_name, response_schema=response_schema),
            )
        except Exception as exc:
            _mark_model_transient_cooldown(m_name, exc)
            last_err = exc
    raise RuntimeError(f"Gemini fast generation failed across models: {last_err}")


def _read_key_from_file(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val and val != "your_gemini_api_key_here":
                    return val
    except Exception:
        pass
    return None


def get_gemini_api_key() -> str | None:
    """Hot-reload .env (and .env.example if edited by user) to return the latest GEMINI_API_KEY."""
    candidates: list[tuple[float, str]] = []
    for p in (_ENV_PATH, _ENV_EXAMPLE_PATH):
        k = _read_key_from_file(p)
        if k:
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            candidates.append((mtime, k))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        latest_key = candidates[0][1]
        os.environ["GEMINI_API_KEY"] = latest_key
        return latest_key
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    key = os.getenv("GEMINI_API_KEY")
    return key.strip() if key and key.strip() and key != "your_gemini_api_key_here" else None


def get_device_and_dtype() -> tuple[str, "torch.dtype"]:
    """Auto-detect compute device (cuda / mps / cpu) and select optimal dtype."""
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


@dataclass(frozen=True)
class SegmentationConfig:
    """Thresholds for OpenCV preprocessing, ruled-line removal, and layout analysis."""

    target_min_width: int = 1000
    target_max_width: int = 1600

    # Denoising & Adaptive Thresholding
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    adaptive_block_size: int = 31
    adaptive_c: int = 15

    # Deskew constraints (degrees)
    min_deskew_angle_deg: float = 0.25
    max_deskew_angle_deg: float = 15.0

    # Ruled-line suppression
    ruled_line_min_width_ratio: float = 0.06
    vertical_stroke_preserve_height: int = 7

    # Morphological grouping for text lines
    line_dilate_width_ratio: float = 0.025
    line_dilate_min_width: int = 24
    line_dilate_max_width: int = 55
    line_dilate_height: int = 3

    # Region filtering
    min_region_width: int = 18
    min_region_height: int = 12
    min_region_area: int = 220

    # Text vs. Diagram classification heuristics
    diagram_height_multiplier: float = 3.0
    diagram_max_aspect_ratio: float = 5.0
    diagram_min_area: int = 4000
    diagram_min_stroke_density: float = 0.008
    diagram_max_stroke_density: float = 0.65
    diagram_merge_gap_px: int = 20

    # Cropping & Reading order
    text_line_padding_px: int = 6
    diagram_padding_px: int = 15
    reading_order_y_jitter_px: int = 15

    # TrOCR inference
    trocr_batch_size: int = 8
    trocr_max_new_tokens: int = 128


@dataclass(frozen=True)
class GradingConfig:
    """Thresholds and weights for rubric scoring."""

    keypoint_hit_threshold: float = 0.54
    keypoint_partial_threshold: float = 0.36
    diagram_weight_when_required: float = 0.30
    semantic_blend_weight: float = 0.20


SEGMENTATION_CONFIG = SegmentationConfig()
GRADING_CONFIG = GradingConfig()
