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
GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
GEMINI_FAST_GRADING_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
LOCAL_VLM_MODEL_NAME: str = os.getenv(
    "LOCAL_VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"
)
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Groq API configuration for ultra-fast fallback & multi-model evaluation
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_TEXT_MODEL: str = os.getenv("GROQ_TEXT_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "llama-3.1-8b-instant",
)

_CLIENT_LOCK = threading.Lock()
_GEMINI_CLIENT_CACHE: dict[tuple[str, int], Any] = {}
_COOLDOWN_LOCK = threading.Lock()
_MODEL_COOLDOWN_UNTIL: dict[str, float] = {}

# Multi-key round-robin rotation for Gemini
_KEY_LOCK = threading.Lock()
_KEY_INDEX = 0
_KEY_LIST: list[str] = []
_BAD_KEY_COOLDOWN: dict[str, float] = {}


def mark_gemini_key_bad(key: str, ttl_seconds: float = 300.0) -> None:
    """Mark an unauthorized (403/401) or depleted key temporarily bad so failover routes to working keys."""
    if not key:
        return
    with _KEY_LOCK:
        _BAD_KEY_COOLDOWN[key] = time.monotonic() + ttl_seconds


class FastGenResponse:
    """Wrapper response matching Google GenAI generate_content interface."""
    def __init__(self, parsed: Any = None, text: str = ""):
        self.parsed = parsed
        self.text = text


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
    """Put transient 503/504/429 models on cooldown so parallel calls skip slow stalls."""
    msg = str(exc).upper()
    if any(tok in msg for tok in ("429", "RESOURCE_EXHAUSTED", "QUOTA")):
        ttl = 900.0  # 15 min cooldown for quota exhaustion
        with _COOLDOWN_LOCK:
            _MODEL_COOLDOWN_UNTIL[model_name] = time.monotonic() + ttl
    elif any(tok in msg for tok in ("503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED")):
        ttl = 6.0
        with _COOLDOWN_LOCK:
            _MODEL_COOLDOWN_UNTIL[model_name] = time.monotonic() + ttl


def get_gemini_client(api_key: str, timeout_ms: int = 15000) -> Any:
    """Return a cached thread-safe Gemini Client configured for fast fail-over with robust retry on 503."""
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
                    retry_options=types.HttpRetryOptions(attempts=3),
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
    if "flash-lite" in lower_m or "3.8-flash" in lower_m or "3.5-flash" in lower_m:
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="MINIMAL")

    return types.GenerateContentConfig(**kwargs)


def generate_content_groq(
    contents: Any,
    response_schema: Any = None,
    models: tuple[str, ...] | list[str] = GROQ_FALLBACK_MODELS,
    timeout_s: float = 18.0,
) -> FastGenResponse:
    """Generate structured JSON content using Groq LLM API with schema enforcement and model fallback."""
    import json
    groq_key = get_groq_api_key()
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not configured.")

    from groq import Groq
    client = Groq(api_key=groq_key, timeout=timeout_s)

    # Extract text from contents
    text_content = ""
    if isinstance(contents, str):
        text_content = contents
    elif isinstance(contents, (list, tuple)):
        text_parts = []
        for p in contents:
            if isinstance(p, str):
                text_parts.append(p)
            elif hasattr(p, "text") and p.text:
                text_parts.append(p.text)
        text_content = "\n\n".join(text_parts)
    else:
        text_content = str(contents)

    system_instr = (
        "You are an expert academic examiner and evaluator. "
        "Assess student answers with high precision, fair step-wise marks, and strict adherence to rubric key points. "
        "Always respond with strict, valid JSON matching the required schema."
    )
    if response_schema is not None:
        try:
            schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
            text_content = (
                f"{text_content}\n\n"
                f"CRITICAL REQUIREMENT: Output must strictly conform to this JSON Schema:\n{schema_json}\n"
                "Respond ONLY with valid JSON without any markdown formatting or commentary."
            )
        except Exception:
            pass

    last_err = None
    for model_name in models:
        for attempt in range(2):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_instr},
                        {"role": "user", "content": text_content},
                    ],
                    model=model_name,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                raw_text = chat_completion.choices[0].message.content or "{}"
                clean_raw = raw_text.strip()
                if clean_raw.startswith("```json"):
                    clean_raw = clean_raw[7:]
                elif clean_raw.startswith("```"):
                    clean_raw = clean_raw[3:]
                if clean_raw.endswith("```"):
                    clean_raw = clean_raw[:-3]
                clean_raw = clean_raw.strip()

                parsed_obj = None
                if response_schema is not None:
                    parsed_obj = response_schema.model_validate_json(clean_raw)
                return FastGenResponse(parsed=parsed_obj, text=clean_raw)
            except Exception as exc:
                last_err = exc
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break
    raise RuntimeError(f"Groq generation failed across models: {last_err}")


def generate_content_fast(
    api_key: str,
    models: tuple[str, ...] | list[str],
    contents: Any,
    response_schema: Any = None,
    timeout_ms: int = 14000,
) -> Any:
    """Execute generate_content using adaptive cooldown routing across Gemini models with transient retry, multi-key failover, and Groq fallback."""
    keys_to_try = [api_key]
    for alt_k in get_all_valid_gemini_api_keys():
        if alt_k not in keys_to_try:
            keys_to_try.append(alt_k)

    last_err: Exception | None = None
    for cur_key in keys_to_try:
        client = get_gemini_client(cur_key, timeout_ms=timeout_ms)
        key_auth_error = False
        for m_name in _ordered_healthy_models(models):
            for attempt in range(2):
                try:
                    return client.models.generate_content(
                        model=m_name,
                        contents=contents,
                        config=build_fast_gemini_config(m_name, response_schema=response_schema),
                    )
                except Exception as exc:
                    last_err = exc
                    msg = str(exc).upper()
                    if any(tok in msg for tok in ("PERMISSION_DENIED", "ACCESS_DENIED", "UNAUTHENTICATED", "API_KEY_INVALID", "403", "401")):
                        mark_gemini_key_bad(cur_key, 600.0)
                        key_auth_error = True
                        break
                    if any(tok in msg for tok in ("503", "UNAVAILABLE", "429")) and attempt == 0:
                        time.sleep(1.0)
                        continue
                    _mark_model_transient_cooldown(m_name, exc)
                    break
            if key_auth_error:
                break
        if not key_auth_error and last_err is None:
            break

    # If Gemini failed across all models, try Groq fallback if configured
    groq_k = get_groq_api_key()
    if groq_k:
        try:
            return generate_content_groq(contents, response_schema=response_schema)
        except Exception as groq_exc:
            last_err = f"Gemini error: {last_err}; Groq error: {groq_exc}"

    raise RuntimeError(f"Fast generation failed across Gemini & Groq models: {last_err}")


def _extract_all_keys(val: str) -> list[str]:
    """Split comma or newline separated API keys."""
    keys: list[str] = []
    for k in val.replace("\n", ",").split(","):
        k = k.strip().strip('"').strip("'")
        if k and k != "your_gemini_api_key_here" and len(k) > 10:
            keys.append(k)
    return keys


def _read_keys_from_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    keys: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEYS=") or line.startswith("GEMINI_API_KEY="):
                val = line.split("=", 1)[1].strip()
                keys.extend(_extract_all_keys(val))
    except Exception:
        pass
    return keys


def get_all_gemini_api_keys() -> list[str]:
    """Return all configured Gemini API keys (supporting comma-separated rotation like Irfan's Evaluator)."""
    keys: list[str] = []
    for p in (_ENV_PATH, _ENV_EXAMPLE_PATH):
        file_keys = _read_keys_from_file(p)
        if file_keys:
            keys.extend(file_keys)
            break
    if not keys:
        load_dotenv(dotenv_path=_ENV_PATH, override=True)
        raw = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")
        keys = _extract_all_keys(raw)
    return list(dict.fromkeys(keys))


def get_all_valid_gemini_api_keys() -> list[str]:
    """Return all configured Gemini API keys that are healthy and not in temporary cooldown."""
    all_k = get_all_gemini_api_keys()
    now = time.monotonic()
    with _KEY_LOCK:
        valid_k = [k for k in all_k if _BAD_KEY_COOLDOWN.get(k, 0.0) <= now]
    return valid_k if valid_k else all_k


def get_next_gemini_api_key() -> str | None:
    """Return next Gemini API key in round-robin rotation across healthy configured keys."""
    global _KEY_INDEX, _KEY_LIST
    active_keys = get_all_valid_gemini_api_keys()
    if not active_keys:
        return None
    with _KEY_LOCK:
        if _KEY_LIST != active_keys:
            _KEY_LIST = list(active_keys)
            _KEY_INDEX = 0
        key = _KEY_LIST[_KEY_INDEX % len(_KEY_LIST)]
        _KEY_INDEX += 1
        return key


def get_gemini_api_key() -> str | None:
    """Hot-reload .env to return an active GEMINI_API_KEY (with round-robin rotation)."""
    return get_next_gemini_api_key()


def get_google_cloud_vision_api_key() -> str | None:
    """Return active Google Cloud Vision API key if configured."""
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    k = os.getenv("GOOGLE_CLOUD_VISION_API_KEY", "").strip()
    if k and len(k) > 10:
        return k
    for key in get_all_gemini_api_keys():
        if key.startswith("AIzaSy"):
            return key
    return None


def get_groq_api_key() -> str | None:
    """Hot-reload .env to return GROQ_API_KEY if configured."""
    load_dotenv(dotenv_path=_ENV_PATH, override=True)
    k = os.getenv("GROQ_API_KEY", "").strip()
    return k if k and k != "your_groq_api_key_here" else None


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
    bilateral_d: int = 5
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
