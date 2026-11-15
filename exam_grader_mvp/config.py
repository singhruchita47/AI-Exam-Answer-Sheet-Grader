"""Configuration, model defaults, thresholds, and device auto-detection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
)
LOCAL_VLM_MODEL_NAME: str = os.getenv(
    "LOCAL_VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct"
)
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")


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
    target_max_width: int = 2000

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
