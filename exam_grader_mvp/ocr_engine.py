"""Image preprocessing, ruled-line suppression, layout segmentation, and TrOCR inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from config import (
    SEGMENTATION_CONFIG,
    TROCR_MODEL_NAME,
    SegmentationConfig,
    get_device_and_dtype,
)

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in image pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    stroke_density: float = 0.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        return self.w / max(self.h, 1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass
class SegmentedRegion:
    """A classified layout region saved to disk for downstream OCR or VLM grading."""

    region_id: str
    kind: Literal["text_line", "diagram"]
    page_number: int
    bbox: BoundingBox
    crop_path: Path
    text: str = ""


@dataclass
class PageOCRResult:
    """Segmentation and OCR outputs for a single exam page."""

    page_number: int
    source_path: Path
    layout_image_path: Path
    skew_angle_deg: float
    text_lines: list[SegmentedRegion] = field(default_factory=list)
    diagrams: list[SegmentedRegion] = field(default_factory=list)
    transcript: str = ""


@dataclass
class OCRPipelineResult:
    """Aggregated OCR and layout segmentation output across all pages."""

    pages: list[PageOCRResult]
    full_transcript: str
    debug_dir: Path


def _resize_if_needed(
    image: np.ndarray, cfg: SegmentationConfig
) -> np.ndarray:
    """Normalize extreme image widths so pixel-based heuristics remain consistent."""
    h, w = image.shape[:2]
    if w < cfg.target_min_width:
        scale = cfg.target_min_width / float(w)
        return cv2.resize(
            image,
            (cfg.target_min_width, int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    if w > cfg.target_max_width:
        scale = cfg.target_max_width / float(w)
        return cv2.resize(
            image,
            (cfg.target_max_width, int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _rotate_image(
    image: np.ndarray,
    angle_deg: float,
    border_value: int | tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Rotate an image around its center without introducing edge contrast seams."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    if border_value is None:
        return cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _normalize_min_area_angle(raw_angle: float) -> float:
    """Normalize cv2.minAreaRect angle into (-45, 45]."""
    angle = float(raw_angle)
    while angle > 45.0:
        angle -= 90.0
    while angle <= -45.0:
        angle += 90.0
    return angle


def _estimate_skew_angle(gray_denoised: np.ndarray, cfg: SegmentationConfig) -> float:
    """Estimate page skew using minAreaRect on text-line contours and ink pixels."""
    h, w = gray_denoised.shape[:2]
    # Otsu isolates darker pen/ink strokes from lighter paper and faint ruled lines
    _, ink_bin = cv2.threshold(
        gray_denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Mask outer 3% border so image frame edges never bias minAreaRect
    pad_y, pad_x = max(10, int(h * 0.03)), max(10, int(w * 0.03))
    ink_bin[:pad_y, :] = 0
    ink_bin[h - pad_y :, :] = 0
    ink_bin[:, :pad_x] = 0
    ink_bin[:, w - pad_x :] = 0

    if np.count_nonzero(ink_bin) < 200:
        return 0.0

    # Primary: minAreaRect on horizontally dilated text-line contours
    dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    dilated = cv2.dilate(ink_bin, dil_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_angles: list[float] = []
    for cnt in contours:
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bw > 120 and (bw / max(bh, 1)) > 3.0:
            ang = _normalize_min_area_angle(cv2.minAreaRect(cnt)[2])
            if abs(ang) <= cfg.max_deskew_angle_deg:
                line_angles.append(ang)

    if len(line_angles) >= 2:
        angle = float(np.median(line_angles))
    else:
        ys, xs = np.where(ink_bin > 0)
        if len(xs) > 50_000:
            step = len(xs) // 50_000
            xs = xs[::step]
            ys = ys[::step]
        points = np.column_stack((xs, ys)).astype(np.float32)
        angle = _normalize_min_area_angle(cv2.minAreaRect(points)[2])

    if abs(angle) < cfg.min_deskew_angle_deg or abs(angle) > cfg.max_deskew_angle_deg:
        return 0.0

    # Disambiguate rotation sign by comparing horizontal projection sharpness
    small_h = min(h, 700)
    scale = small_h / float(h)
    small_w = max(1, int(round(w * scale)))
    small_bin = cv2.resize(ink_bin, (small_w, small_h), interpolation=cv2.INTER_NEAREST)

    cand_pos = _rotate_image(small_bin, angle, border_value=0)
    cand_neg = _rotate_image(small_bin, -angle, border_value=0)

    var_orig = float(np.var(np.sum(small_bin > 0, axis=1)))
    var_pos = float(np.var(np.sum(cand_pos > 0, axis=1)))
    var_neg = float(np.var(np.sum(cand_neg > 0, axis=1)))

    if var_pos >= var_neg and var_pos > var_orig:
        return angle
    if var_neg > var_pos and var_neg > var_orig:
        return -angle
    return 0.0


def _suppress_ruled_lines(
    deskewed_bgr: np.ndarray,
    deskewed_gray: np.ndarray,
    cfg: SegmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove horizontal ruled notebook lines and margin lines while preserving handwriting."""
    h, w = deskewed_gray.shape[:2]

    # Compute pen-adapted threshold (notebook ruled lines are lighter than pen ink)
    otsu_val, _ = cv2.threshold(
        deskewed_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    # Pen ink is significantly darker than ruled notebook paper lines
    pen_thresh = int(max(40, min(220, otsu_val * 0.90)))
    _, pen_bin = cv2.threshold(
        deskewed_gray, pen_thresh, 255, cv2.THRESH_BINARY_INV
    )

    # Zero out outer 1.5% edge frame
    pad_y = max(8, int(h * 0.015))
    pad_x = max(8, int(w * 0.015))
    pen_bin[:pad_y, :] = 0
    pen_bin[h - pad_y :, :] = 0
    pen_bin[:, :pad_x] = 0
    pen_bin[:, w - pad_x :] = 0

    # 1. Detect straight horizontal ruled lines
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (38, 1))
    h_lines = cv2.morphologyEx(pen_bin, cv2.MORPH_OPEN, hk)

    # 2. Detect straight vertical margin lines
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 38))
    v_lines = cv2.morphologyEx(pen_bin, cv2.MORPH_OPEN, vk)

    # Preserve natural handwriting stroke intersections and crossings
    crossings = cv2.bitwise_and(
        cv2.dilate(h_lines, np.ones((3, 3), np.uint8)),
        cv2.dilate(v_lines, np.ones((3, 3), np.uint8)),
    )
    lines_mask = cv2.bitwise_or(h_lines, v_lines)
    lines_to_remove = cv2.subtract(lines_mask, crossings)

    clean_binary = cv2.subtract(pen_bin, lines_to_remove)

    # Suppress prominent vertical notebook margin column (x in 7% to 18% of page width)
    margin_min = int(w * 0.07)
    margin_max = int(w * 0.18)
    col_ink = np.sum(clean_binary > 0, axis=0)
    for cx in range(margin_min, margin_max):
        if col_ink[cx] > h * 0.18:
            clean_binary[:, max(0, cx - 4) : min(w, cx + 5)] = 0

    # Inpaint erased line pixels on the color image for clean TrOCR line crops
    if np.any(lines_to_remove > 0):
        clean_for_ocr = cv2.inpaint(deskewed_bgr, lines_to_remove, 2, cv2.INPAINT_TELEA)
    else:
        clean_for_ocr = deskewed_bgr.copy()

    return clean_binary, clean_for_ocr


def sort_reading_order(
    boxes: list[BoundingBox], y_jitter_px: int = 15
) -> list[BoundingBox]:
    """Sort bounding boxes top-to-bottom and left-to-right within a +/- y_jitter_px threshold."""
    if not boxes:
        return []

    sorted_by_y = sorted(boxes, key=lambda b: (b.y, b.x))
    rows: list[list[BoundingBox]] = []
    current_row: list[BoundingBox] = [sorted_by_y[0]]
    row_anchor_y = sorted_by_y[0].y

    for box in sorted_by_y[1:]:
        if abs(box.y - row_anchor_y) <= y_jitter_px:
            current_row.append(box)
        else:
            rows.append(sorted(current_row, key=lambda b: b.x))
            current_row = [box]
            row_anchor_y = box.y

    rows.append(sorted(current_row, key=lambda b: b.x))
    return [box for row in rows for box in row]


def _boxes_overlap_or_close(a: BoundingBox, b: BoundingBox, gap: int) -> bool:
    """Check whether two bounding boxes overlap or are within `gap` pixels."""
    return not (
        a.x2 + gap < b.x
        or b.x2 + gap < a.x
        or a.y2 + gap < b.y
        or b.y2 + gap < a.y
    )


def _merge_boxes(boxes: list[BoundingBox], clean_binary: np.ndarray, gap: int) -> list[BoundingBox]:
    """Iteratively merge overlapping or adjacent bounding boxes."""
    if len(boxes) <= 1:
        return boxes

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_merged: list[BoundingBox] = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            curr = merged[i]
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _boxes_overlap_or_close(curr, merged[j], gap):
                    nx = min(curr.x, merged[j].x)
                    ny = min(curr.y, merged[j].y)
                    nx2 = max(curr.x2, merged[j].x2)
                    ny2 = max(curr.y2, merged[j].y2)
                    nw, nh = nx2 - nx, ny2 - ny
                    fg = float(np.count_nonzero(clean_binary[ny:ny2, nx:nx2]))
                    curr = BoundingBox(
                        x=nx,
                        y=ny,
                        w=nw,
                        h=nh,
                        stroke_density=fg / max(nw * nh, 1),
                    )
                    used[j] = True
                    changed = True
            next_merged.append(curr)
        merged = next_merged
    return merged


def _intersection_over_box_area(inner: BoundingBox, outer: BoundingBox) -> float:
    """Compute fraction of `inner`'s area that lies inside `outer` (expanded slightly)."""
    ix1 = max(inner.x, outer.x - 10)
    iy1 = max(inner.y, outer.y - 10)
    ix2 = min(inner.x2, outer.x2 + 10)
    iy2 = min(inner.y2, outer.y2 + 10)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    return inter_area / max(inner.area, 1)


def _segment_regions(
    clean_binary: np.ndarray, cfg: SegmentationConfig
) -> tuple[list[BoundingBox], list[BoundingBox]]:
    """Segment foreground into sorted `text_line` and `diagram` bounding boxes.
    
    Robust layout analysis:
    1. Mask outer 2.5% image borders to eliminate scanner edge shadows.
    2. Uses horizontal line projection across the body to detect individual text line baselines.
    3. Distinguishes text sentence lines (high line width and ink density) from diagram drawing elements.
    4. Clusters diagram drawings (circuits, leaves, organs) into unified diagrams.
    5. Returns strictly sorted text lines and diagrams.
    """
    page_h, page_w = clean_binary.shape[:2]

    # Mask outer 2.5% frame to avoid scanner/border artifacts
    pad_y = max(15, int(page_h * 0.025))
    pad_x = max(15, int(page_w * 0.025))
    masked_bin = clean_binary.copy()
    masked_bin[:pad_y, :] = 0
    masked_bin[page_h - pad_y :, :] = 0
    masked_bin[:, :pad_x] = 0
    masked_bin[:, page_w - pad_x :] = 0

    import scipy.signal

    # Project ink across page body (columns 50 to w - 50)
    body_bin = masked_bin[:, 50 : page_w - 50]
    row_ink = np.sum(body_bin > 0, axis=1).astype(np.float32)
    sm = np.convolve(row_ink, np.ones(15, dtype=np.float32) / 15.0, mode="same")
    peaks, _ = scipy.signal.find_peaks(
        sm, distance=int(page_h * 0.015), prominence=18
    )

    row_boxes: list[tuple[int, int, int, int, int, float]] = []
    for pk in peaks:
        if sm[pk] < 20:
            continue
        y1 = pk
        while y1 > 0 and sm[y1] > max(10.0, sm[pk] * 0.15) and pk - y1 < int(page_h * 0.025):
            y1 -= 1
        y2 = pk
        while y2 < page_h - 1 and sm[y2] > max(10.0, sm[pk] * 0.15) and y2 - pk < int(page_h * 0.025):
            y2 += 1

        line_strip = masked_bin[y1:y2, 50 : page_w - 50]
        nz_cols = np.where(np.sum(line_strip > 0, axis=0) > 0)[0]
        if len(nz_cols) >= 30:
            lx = 50 + int(nz_cols[0])
            lw = int(nz_cols[-1] - nz_cols[0])
            lh = y2 - y1
            ink = int(np.count_nonzero(line_strip))
            row_boxes.append((lx, y1, lw, lh, ink, float(sm[pk])))

    text_candidates: list[BoundingBox] = []
    diagram_strips: list[BoundingBox] = []

    for lx, ly, lw, lh, ink, peak_val in row_boxes:
        density = ink / float(lw * lh) if lw * lh > 0 else 0.0
        # Sentence text line: wide line with solid ink density
        is_sentence = (lw >= int(page_w * 0.40) and peak_val >= 120) or (
            lw >= 250 and peak_val >= 80 and density >= 0.075
        )
        if is_sentence:
            text_candidates.append(
                BoundingBox(x=lx, y=ly, w=lw, h=lh, stroke_density=density)
            )
        else:
            diagram_strips.append(
                BoundingBox(x=lx, y=ly, w=lw, h=lh, stroke_density=density)
            )

    # Cluster diagram strips into unified diagram boxes
    unified_diagrams: list[BoundingBox] = []
    if diagram_strips:
        curr_d: list[BoundingBox] = [diagram_strips[0]]
        for s in diagram_strips[1:]:
            prev = curr_d[-1]
            if s.y - prev.y2 <= int(page_h * 0.08):
                curr_d.append(s)
            else:
                d_h = curr_d[-1].y2 - curr_d[0].y
                if d_h >= max(90, int(page_h * 0.08)):
                    min_y = min(b.y for b in curr_d)
                    max_y = max(b.y2 for b in curr_d)
                    crop_d = masked_bin[min_y:max_y, 50 : page_w - 50]
                    if np.count_nonzero(crop_d) >= 1000:
                        row_p = np.sum(crop_d > 0, axis=1)
                        act_r = np.where(row_p > 20)[0]
                        if len(act_r) >= 40:
                            ay1 = min_y + int(act_r[0])
                            ay2 = min_y + int(act_r[-1])
                            col_p = np.sum(masked_bin[ay1:ay2, 50 : page_w - 50] > 0, axis=0)
                            act_c = np.where(col_p > 15)[0]
                            if len(act_c) >= 60:
                                min_x = 50 + int(act_c[0])
                                max_x = 50 + int(act_c[-1])
                                pad = 20
                                if (max_x - min_x) >= max(320, int(page_w * 0.22)):
                                    dw = (max_x - min_x) + 2 * pad
                                    dh = (ay2 - ay1) + 2 * pad
                                    dx = max(0, min_x - pad)
                                    dy = max(0, ay1 - pad)
                                    unified_diagrams.append(
                                        BoundingBox(
                                            x=dx,
                                            y=dy,
                                            w=min(page_w - dx, dw),
                                            h=min(page_h - dy, dh),
                                            stroke_density=float(np.count_nonzero(crop_d)) / max(dw * dh, 1),
                                        )
                                    )
                curr_d = [s]
        if curr_d:
            d_h = curr_d[-1].y2 - curr_d[0].y
            if d_h >= max(90, int(page_h * 0.08)):
                min_y = min(b.y for b in curr_d)
                max_y = max(b.y2 for b in curr_d)
                crop_d = masked_bin[min_y:max_y, 50 : page_w - 50]
                if np.count_nonzero(crop_d) >= 1000:
                    row_p = np.sum(crop_d > 0, axis=1)
                    act_r = np.where(row_p > 20)[0]
                    if len(act_r) >= 40:
                        ay1 = min_y + int(act_r[0])
                        ay2 = min_y + int(act_r[-1])
                        col_p = np.sum(masked_bin[ay1:ay2, 50 : page_w - 50] > 0, axis=0)
                        act_c = np.where(col_p > 15)[0]
                        if len(act_c) >= 60:
                            min_x = 50 + int(act_c[0])
                            max_x = 50 + int(act_c[-1])
                            pad = 20
                            if (max_x - min_x) >= max(320, int(page_w * 0.22)):
                                dw = (max_x - min_x) + 2 * pad
                                dh = (ay2 - ay1) + 2 * pad
                                dx = max(0, min_x - pad)
                                dy = max(0, ay1 - pad)
                                unified_diagrams.append(
                                    BoundingBox(
                                        x=dx,
                                        y=dy,
                                        w=min(page_w - dx, dw),
                                        h=min(page_h - dy, dh),
                                        stroke_density=float(np.count_nonzero(crop_d)) / max(dw * dh, 1),
                                    )
                                )

    # Final Text Lines: Retain lines NOT inside diagram boxes (check vertical span)
    final_text_lines: list[BoundingBox] = []
    for l in text_candidates:
        l_cy = l.y + l.h / 2.0
        inside = any(
            (d.y - 5 <= l_cy <= d.y2 + 5)
            for d in unified_diagrams
        )
        if not inside:
            final_text_lines.append(l)

    sorted_text_lines = sort_reading_order(
        final_text_lines, y_jitter_px=cfg.reading_order_y_jitter_px
    )
    sorted_diagrams = sort_reading_order(
        unified_diagrams, y_jitter_px=cfg.reading_order_y_jitter_px
    )
    return sorted_text_lines, sorted_diagrams


def _split_if_multiline_block(
    box: BoundingBox, clean_binary: np.ndarray, median_line_h: float
) -> list[BoundingBox]:
    """Split tall multi-line text blocks into individual text lines using horizontal projection."""
    if box.h < max(80, int(median_line_h * 1.8)):
        return [box]

    crop = clean_binary[box.y : box.y2, box.x : box.x2]
    proj = np.sum(crop > 0, axis=1).astype(np.float32)
    smoothed = np.convolve(proj, np.ones(7, dtype=np.float32) / 7.0, mode="same")

    import scipy.signal
    peaks, _ = scipy.signal.find_peaks(smoothed, distance=30, prominence=20)
    if len(peaks) < 2:
        return [box]

    valleys = [0]
    for i in range(len(peaks) - 1):
        v = peaks[i] + np.argmin(smoothed[peaks[i] : peaks[i + 1]])
        valleys.append(int(v))
    valleys.append(box.h)

    sub_boxes: list[BoundingBox] = []
    for i in range(len(valleys) - 1):
        y1, y2 = valleys[i], valleys[i + 1]
        sub = crop[y1:y2, :]
        nz_r = np.where(np.sum(sub > 0, axis=1) > 0)[0]
        nz_c = np.where(np.sum(sub > 0, axis=0) > 0)[0]
        if len(nz_r) > 0 and len(nz_c) > 0:
            sw = nz_c[-1] - nz_c[0] + 1
            sh = nz_r[-1] - nz_r[0] + 1
            if sw >= 15 and sh >= 12:
                sx = box.x + nz_c[0]
                sy = box.y + y1 + nz_r[0]
                fg = float(np.count_nonzero(clean_binary[sy : sy + sh, sx : sx + sw]))
                sub_boxes.append(
                    BoundingBox(
                        x=sx,
                        y=sy,
                        w=sw,
                        h=sh,
                        stroke_density=fg / max(sw * sh, 1),
                    )
                )

    return sub_boxes if sub_boxes else [box]


def _merge_collinear_lines(
    boxes: list[BoundingBox],
    clean_binary: np.ndarray,
    diagrams: list[BoundingBox],
    max_gap: int = 65,
) -> list[BoundingBox]:
    """Merge horizontally adjacent fragments belonging to the same text line."""
    if len(boxes) <= 1:
        return boxes

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        next_merged: list[BoundingBox] = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            curr = merged[i]
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                other = merged[j]
                vert_overlap = min(curr.y2, other.y2) - max(curr.y, other.y)
                cy_diff = abs((curr.y + curr.h / 2.0) - (other.y + other.h / 2.0))
                same_row = vert_overlap > 0.45 * min(curr.h, other.h) or cy_diff <= 12.0
                horiz_gap = max(0, max(curr.x - other.x2, other.x - curr.x2))
                if same_row and horiz_gap <= max_gap:
                    nx = min(curr.x, other.x)
                    ny = min(curr.y, other.y)
                    nx2 = max(curr.x2, other.x2)
                    ny2 = max(curr.y2, other.y2)
                    candidate = BoundingBox(x=nx, y=ny, w=nx2 - nx, h=ny2 - ny)
                    if any(_boxes_overlap_or_close(candidate, d, gap=0) for d in diagrams):
                        continue
                    fg = float(np.count_nonzero(clean_binary[ny:ny2, nx:nx2]))
                    curr = BoundingBox(
                        x=nx,
                        y=ny,
                        w=nx2 - nx,
                        h=ny2 - ny,
                        stroke_density=fg / max((nx2 - nx) * (ny2 - ny), 1),
                    )
                    used[j] = True
                    changed = True
            next_merged.append(curr)
        merged = next_merged
    return merged


def _crop_with_padding(
    image: np.ndarray, box: BoundingBox, pad: int
) -> np.ndarray:
    """Crop a bounding box from an image with safe boundary clamping."""
    h, w = image.shape[:2]
    x1 = max(0, box.x - pad)
    y1 = max(0, box.y - pad)
    x2 = min(w, box.x2 + pad)
    y2 = min(h, box.y2 + pad)
    return image[y1:y2, x1:x2]


class OCREngine:
    """Handles OpenCV layout segmentation and batched TrOCR handwriting transcription."""

    def __init__(
        self,
        trocr_model_name: str = TROCR_MODEL_NAME,
        cfg: SegmentationConfig = SEGMENTATION_CONFIG,
    ) -> None:
        self.trocr_model_name = trocr_model_name
        self.cfg = cfg
        self.device, self.dtype = get_device_and_dtype()
        self._processor: TrOCRProcessor | None = None
        self._model: VisionEncoderDecoderModel | None = None

    def _ensure_model_loaded(self) -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
        """Lazy-load TrOCR processor and model onto the detected compute device."""
        if self._processor is None or self._model is None:
            logger.info(
                "Loading TrOCR model '%s' on %s (%s)...",
                self.trocr_model_name,
                self.device,
                self.dtype,
            )
            self._processor = TrOCRProcessor.from_pretrained(self.trocr_model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(
                self.trocr_model_name,
                torch_dtype=self.dtype,
            )
            self._model.to(self.device)
            self._model.eval()
        return self._processor, self._model

    def segment_page(
        self, page_path: str | Path, page_number: int, debug_dir: Path
    ) -> PageOCRResult:
        """Preprocess one page, suppress ruled lines, segment regions, and save debug crops."""
        src_path = Path(page_path)
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read page image: {src_path}")

        bgr = _resize_if_needed(bgr, self.cfg)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(
            gray,
            d=self.cfg.bilateral_d,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space,
        )

        skew_angle = _estimate_skew_angle(denoised, self.cfg)
        if abs(skew_angle) > 0.0:
            deskewed_bgr = _rotate_image(bgr, skew_angle)
            deskewed_gray = _rotate_image(denoised, skew_angle)
        else:
            deskewed_bgr = bgr
            deskewed_gray = denoised

        clean_binary, clean_for_ocr = _suppress_ruled_lines(
            deskewed_bgr, deskewed_gray, self.cfg
        )
        line_boxes, diagram_boxes = _segment_regions(clean_binary, self.cfg)

        page_crop_dir = debug_dir / f"page_{page_number}"
        if page_crop_dir.exists():
            for old_crop in page_crop_dir.glob("*.png"):
                old_crop.unlink()
        page_crop_dir.mkdir(parents=True, exist_ok=True)

        annotated = deskewed_bgr.copy()
        text_regions: list[SegmentedRegion] = []
        diagram_regions: list[SegmentedRegion] = []

        for idx, box in enumerate(line_boxes, start=1):
            region_id = f"line_{idx:03d}"
            crop_path = page_crop_dir / f"{region_id}.png"
            crop = _crop_with_padding(clean_for_ocr, box, self.cfg.text_line_padding_px)
            cv2.imwrite(str(crop_path), crop)

            text_regions.append(
                SegmentedRegion(
                    region_id=region_id,
                    kind="text_line",
                    page_number=page_number,
                    bbox=box,
                    crop_path=crop_path,
                )
            )
            cv2.rectangle(annotated, (box.x, box.y), (box.x2, box.y2), (0, 200, 0), 2)
            cv2.putText(
                annotated,
                region_id,
                (box.x, max(20, box.y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 160, 0),
                2,
                cv2.LINE_AA,
            )

        for idx, box in enumerate(diagram_boxes, start=1):
            region_id = f"diagram_{idx:03d}"
            crop_path = page_crop_dir / f"{region_id}.png"
            crop = _crop_with_padding(deskewed_bgr, box, self.cfg.diagram_padding_px)
            cv2.imwrite(str(crop_path), crop)

            diagram_regions.append(
                SegmentedRegion(
                    region_id=region_id,
                    kind="diagram",
                    page_number=page_number,
                    bbox=box,
                    crop_path=crop_path,
                )
            )
            cv2.rectangle(annotated, (box.x, box.y), (box.x2, box.y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                region_id,
                (box.x, max(24, box.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 230),
                2,
                cv2.LINE_AA,
            )

        layout_path = debug_dir / f"page_{page_number}_layout.png"
        cv2.imwrite(str(layout_path), annotated)

        return PageOCRResult(
            page_number=page_number,
            source_path=src_path,
            layout_image_path=layout_path,
            skew_angle_deg=round(skew_angle, 2),
            text_lines=text_regions,
            diagrams=diagram_regions,
        )

    def transcribe_lines(self, crop_paths: list[Path]) -> list[str]:
        """Batch-transcribe cropped single text-line images using TrOCR."""
        if not crop_paths:
            return []

        processor, model = self._ensure_model_loaded()
        transcripts: list[str] = []
        batch_size = self.cfg.trocr_batch_size

        for start in range(0, len(crop_paths), batch_size):
            batch_paths = crop_paths[start : start + batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device=self.device, dtype=self.dtype)

            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=self.cfg.trocr_max_new_tokens,
                )
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for d_idx, text in enumerate(decoded):
                cleaned = text.strip()
                if not cleaned:
                    try:
                        from rapidocr_onnxruntime import RapidOCR

                        fast_ocr = RapidOCR()
                        res, _ = fast_ocr(str(batch_paths[d_idx]))
                        if res:
                            cleaned = " ".join(r[1].strip() for r in res if r[1].strip())
                    except Exception:
                        pass
                transcripts.append(cleaned)

        return transcripts

    def process_pages(
        self, page_paths: list[str | Path], debug_dir: str | Path = "debug_crops"
    ) -> OCRPipelineResult:
        """Segment all input pages and run batched TrOCR on all detected text lines."""
        debug_root = Path(debug_dir)
        debug_root.mkdir(parents=True, exist_ok=True)

        pages: list[PageOCRResult] = []
        all_line_regions: list[SegmentedRegion] = []

        for idx, page_path in enumerate(page_paths, start=1):
            page_result = self.segment_page(page_path, page_number=idx, debug_dir=debug_root)
            pages.append(page_result)
            all_line_regions.extend(page_result.text_lines)

        if all_line_regions:
            line_texts = self.transcribe_lines([r.crop_path for r in all_line_regions])
            for region, text in zip(all_line_regions, line_texts):
                region.text = text

        for page in pages:
            lines = [r.text for r in page.text_lines if r.text]
            page.transcript = "\n".join(lines)

        full_transcript = "\n".join(p.transcript for p in pages if p.transcript)
        return OCRPipelineResult(
            pages=pages,
            full_transcript=full_transcript,
            debug_dir=debug_root,
        )
