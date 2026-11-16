"""Image preprocessing, ruled-line suppression, layout segmentation, and TrOCR inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
from PIL import Image

from config import (
    SEGMENTATION_CONFIG,
    TROCR_MODEL_NAME,
    SegmentationConfig,
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
    need_inpaint: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove horizontal ruled notebook lines and margin lines while preserving handwriting."""
    h, w = deskewed_gray.shape[:2]

    # Compute pen-adapted threshold (rejects faint reverse-side bleed-through and light ruled lines)
    otsu_val, _ = cv2.threshold(
        deskewed_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    pen_thresh = int(max(45, min(178, otsu_val * 0.83)))
    _, global_bin = cv2.threshold(
        deskewed_gray, pen_thresh, 255, cv2.THRESH_BINARY_INV
    )
    adapt_bin = cv2.adaptiveThreshold(
        deskewed_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )
    pen_bin = cv2.bitwise_and(global_bin, adapt_bin)

    # Zero out outer 1.5% edge frame
    pad_y = max(8, int(h * 0.015))
    pad_x = max(8, int(w * 0.015))
    pen_bin[:pad_y, :] = 0
    pen_bin[h - pad_y :, :] = 0
    pen_bin[:, :pad_x] = 0
    pen_bin[:, w - pad_x :] = 0

    # 1. Detect straight horizontal ruled lines (bridge tiny 3px gaps first so broken scan lines are caught)
    h_closed = cv2.morphologyEx(
        pen_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    )
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (42, 1))
    h_lines = cv2.morphologyEx(h_closed, cv2.MORPH_OPEN, hk)
    h_lines = cv2.dilate(h_lines, np.ones((3, 1), np.uint8), iterations=1)

    # 2. Detect straight vertical margin/table lines
    v_closed = cv2.morphologyEx(
        pen_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    )
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 42))
    v_lines = cv2.morphologyEx(v_closed, cv2.MORPH_OPEN, vk)
    v_lines = cv2.dilate(v_lines, np.ones((1, 3), np.uint8), iterations=1)

    # Preserve handwriting strokes that are thicker than a thin ruled line (vertical thickness >= 6px)
    vert_stroke = cv2.morphologyEx(
        pen_bin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    )
    vert_stroke = cv2.dilate(vert_stroke, np.ones((3, 3), np.uint8), iterations=1)
    lines_mask = cv2.bitwise_or(h_lines, v_lines)
    lines_to_remove = cv2.subtract(lines_mask, vert_stroke)

    clean_binary = cv2.subtract(pen_bin, lines_to_remove)

    # Suppress prominent vertical notebook margin column (x in 7% to 18% of page width)
    margin_min = int(w * 0.07)
    margin_max = int(w * 0.18)
    col_ink = np.sum(clean_binary > 0, axis=0)
    for cx in range(margin_min, margin_max):
        if col_ink[cx] > h * 0.18:
            clean_binary[:, max(0, cx - 4) : min(w, cx + 5)] = 0

    # Inpaint erased line pixels on the color image only when saving line crops for local TrOCR
    if need_inpaint and np.any(lines_to_remove > 0):
        clean_for_ocr = cv2.inpaint(deskewed_bgr, lines_to_remove, 2, cv2.INPAINT_TELEA)
    else:
        clean_for_ocr = deskewed_bgr

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
    clean_binary: np.ndarray,
    cfg: SegmentationConfig,
    deskewed_gray: np.ndarray | None = None,
    page_number: int = 1,
) -> tuple[list[BoundingBox], list[BoundingBox]]:
    """Segment foreground into sorted `text_line` and `diagram` bounding boxes.
    
    Eliminates false diagram detections on normal handwritten lines, underlined headings,
    top printed page header frames, university/college cover introduction blocks, and
    notebook margin columns while accurately detecting true non-text visual structures.
    """
    page_h, page_w = clean_binary.shape[:2]

    # Determine top_header_y dynamically:
    # - On portrait pages (page_h > page_w), check if the top half is an unruled University/College Cover Introduction block
    #   (like Scope Global Skills University Answer Book on Page 1); if so, mask down to the first full-width divider line (~48%).
    #   Otherwise, on normal portrait continuation pages, start near the top (3.5%).
    # - On landscape board booklet pages (page_w >= page_h), mask the top 15.5% board header zone.
    if page_h > page_w:
        top_header_y = max(20, int(page_h * 0.035))
        if deskewed_gray is not None:
            # Use strict dark black threshold (125) and wide (80px) horizontal kernel for solid printed
            # University Cover dividers so pale-blue notebook rules NEVER trigger false top-half masking.
            _, b_rule = cv2.threshold(deskewed_gray, 125, 255, cv2.THRESH_BINARY_INV)
            b_dil = cv2.dilate(b_rule, np.ones((3, 5), np.uint8), iterations=1)
            hl = cv2.morphologyEx(
                b_dil, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
            )
            row_w = np.sum(hl > 0, axis=1) / float(page_w)
            top_rules_rows = int(np.sum(row_w[int(page_h * 0.04) : int(page_h * 0.35)] > 0.55))
            mid_div = [
                int(y)
                for y in np.where(row_w > 0.68)[0]
                if int(page_h * 0.38) <= y <= int(page_h * 0.58)
            ]
            if page_number == 1 and top_rules_rows <= 12 and mid_div:
                top_header_y = mid_div[0]
    else:
        top_header_y = max(25, int(page_h * 0.155))

    pad_y = max(15, int(page_h * 0.025))
    pad_x = max(15, int(page_w * 0.025))
    masked_bin = clean_binary.copy()
    masked_bin[:top_header_y, :] = 0
    masked_bin[page_h - pad_y :, :] = 0
    masked_bin[:, :pad_x] = 0
    masked_bin[:, page_w - pad_x :] = 0

    import scipy.signal

    # Exclude left notebook margin column and right edge border when detecting text lines
    x_left_body = (
        max(60, int(page_w * 0.102)) if page_h > page_w else max(70, int(page_w * 0.115))
    )
    x_right_body = page_w - max(35, int(page_w * 0.035))
    body_bin = masked_bin[:, x_left_body:x_right_body]

    row_ink = np.sum(body_bin > 0, axis=1).astype(np.float32)
    sm = np.convolve(row_ink, np.ones(13, dtype=np.float32) / 13.0, mode="same")
    peaks, _ = scipy.signal.find_peaks(
        sm, distance=max(28, int(page_h * 0.023)), prominence=6.5
    )

    row_boxes: list[tuple[int, int, int, int, int, float]] = []
    for pk in peaks:
        if sm[pk] < 7.0:
            continue
        y1 = pk
        while y1 > top_header_y and sm[y1] > max(3.5, sm[pk] * 0.16) and pk - y1 < int(page_h * 0.016):
            y1 -= 1
        y2 = pk
        while y2 < page_h - 1 and sm[y2] > max(3.5, sm[pk] * 0.16) and y2 - pk < int(page_h * 0.016):
            y2 += 1

        line_strip = masked_bin[y1:y2, x_left_body:x_right_body]
        strip_h, strip_w = line_strip.shape[:2]
        if strip_h < 8 or strip_w < 40:
            continue

        # Remove any residual thin horizontal underline tails inside the strip using vertical column span
        h_dil = cv2.dilate(line_strip, np.ones((3, 5), np.uint8), iterations=1)
        h_line = cv2.morphologyEx(
            h_dil, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        )
        h_line = cv2.dilate(h_line, np.ones((5, 5), np.uint8), iterations=1)
        has_ink = line_strip > 0
        col_any = np.any(has_ink, axis=0)
        col_min_y = np.where(col_any, np.argmax(has_ink, axis=0), 0)
        col_max_y = np.where(col_any, strip_h - 1 - np.argmax(has_ink[::-1, :], axis=0), 0)
        col_span = np.where(col_any, col_max_y - col_min_y + 1, 0)
        thin_cols = (col_span <= 8)[np.newaxis, :]
        clean_strip = np.where((h_line > 0) & thin_cols, 0, line_strip).astype(np.uint8)

        # Also keep only columns where vertical character span >= 9 (excluding full-strip vertical table rules)
        v_rule_col = col_span >= max(24, strip_h - 2)
        char_col_bool = (col_span >= 9) & (~v_rule_col)
        if int(np.sum(char_col_bool)) < 20:
            continue
        char_cols_mask = char_col_bool.astype(np.float32)
        char_cols_sm = np.convolve(char_cols_mask, np.ones(15, dtype=np.float32), mode="same")
        col_ink = np.sum(clean_strip > 0, axis=0).astype(np.float32)
        col_sm = np.convolve(col_ink, np.ones(9, dtype=np.float32), mode="same")
        nz_cols = np.where((col_sm >= 5.0) & (char_cols_sm >= 3.0))[0]

        if len(nz_cols) >= 18:
            active_spans = col_span[nz_cols]
            p85_span = float(np.percentile(active_spans, 85)) if len(active_spans) > 0 else 0.0
            raw_w = int(nz_cols[-1] - nz_cols[0])
            # Reject flat horizontal underline strips that have no character height
            if p85_span <= 9.0 and raw_w >= int(page_w * 0.25):
                continue

            pad_left = 12
            pad_right = 24
            pad_top = 10
            pad_bottom = 6
            lx = max(x_left_body - 8, x_left_body + int(nz_cols[0]) - pad_left)
            rx = min(page_w - 15, x_left_body + int(nz_cols[-1]) + pad_right)
            by1 = max(top_header_y - 6, y1 - pad_top)
            by2 = min(page_h - pad_y, y2 + pad_bottom)
            lw = max(25, rx - lx)
            lh = max(12, by2 - by1)
            ink = int(np.count_nonzero(clean_strip[:, nz_cols[0] : nz_cols[-1] + 1]))
            if ink >= 45:
                row_boxes.append((lx, by1, lw, lh, ink, float(sm[pk])))

    text_candidates: list[BoundingBox] = []
    diagram_strips: list[BoundingBox] = []

    for lx, ly, lw, lh, ink, peak_val in row_boxes:
        density = ink / float(lw * lh) if lw * lh > 0 else 0.0
        # Wide sentence text line
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

    # Cluster non-sentence strips and validate whether they form a TRUE 2D diagram
    # vs short handwritten lines (headings like 'Answer no. - 01', MCQ options, True/False)
    unified_diagrams: list[BoundingBox] = []
    clusters: list[list[BoundingBox]] = []
    if diagram_strips:
        curr_c: list[BoundingBox] = [diagram_strips[0]]
        for s in diagram_strips[1:]:
            if s.y - curr_c[-1].y2 <= int(page_h * 0.08):
                curr_c.append(s)
            else:
                clusters.append(curr_c)
                curr_c = [s]
        clusters.append(curr_c)

    margin_x_cut = max(70, int(page_w * 0.13))

    for curr_d in clusters:
        min_y = min(b.y for b in curr_d)
        max_y = max(b.y2 for b in curr_d)
        d_h = max_y - min_y
        is_valid_diagram = False

        # True diagrams must not start in the top printed header zone and must have 2D height
        if d_h >= max(160, int(page_h * 0.20)) and min_y >= int(page_h * 0.15):
            crop_d = masked_bin[min_y:max_y, margin_x_cut : page_w - 50]
            if np.count_nonzero(crop_d) >= 1200:
                row_p = np.sum(crop_d > 0, axis=1)
                act_r = np.where(row_p > 20)[0]
                if len(act_r) >= 60:
                    ay1 = min_y + int(act_r[0])
                    ay2 = min_y + int(act_r[-1])
                    span_h = max(1, ay2 - ay1)
                    # Check vertical row continuity: stacked short text lines have blank valleys (<0.36)
                    active_rows = np.sum(np.sum(masked_bin[ay1:ay2, margin_x_cut : page_w - 50] > 0, axis=1) > 5)
                    vert_continuity = float(active_rows) / float(span_h)

                    col_p = np.sum(masked_bin[ay1:ay2, margin_x_cut : page_w - 50] > 0, axis=0)
                    act_c = np.where(col_p > 15)[0]
                    if len(act_c) >= 80 and vert_continuity >= 0.38:
                        min_x = margin_x_cut + int(act_c[0])
                        max_x = margin_x_cut + int(act_c[-1])
                        pad = 20
                        dw = (max_x - min_x) + 2 * pad
                        dh = (ay2 - ay1) + 2 * pad
                        ar = dw / float(max(dh, 1))
                        # Reject wide text paragraphs/tables (ar > 1.95 or full page width)
                        if (
                            0.30 <= ar <= 1.95
                            and dw >= max(300, int(page_w * 0.22))
                            and dw <= int(page_w * 0.78)
                            and dh >= int(page_h * 0.22)
                        ):
                            dil_d = cv2.dilate(
                                masked_bin[ay1:ay2, min_x:max_x],
                                np.ones((9, 9), np.uint8),
                                iterations=1,
                            )
                            num_cc, _, stats_cc, _ = cv2.connectedComponentsWithStats(
                                dil_d, connectivity=8
                            )
                            has_tall_stroke = any(
                                stats_cc[i, cv2.CC_STAT_HEIGHT] >= 85
                                and stats_cc[i, cv2.CC_STAT_WIDTH] >= 75
                                for i in range(1, num_cc)
                            )
                            if has_tall_stroke:
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
                                is_valid_diagram = True

        # Promote short handwritten text lines (headings, MCQ options, True/False) to text_candidates!
        if not is_valid_diagram:
            for b in curr_d:
                if b.y >= top_header_y - 8 and b.w >= 45:
                    text_candidates.append(b)

    # Also detect single-component large 2D drawings (e.g. full-page maps/charts on portrait pages)
    if not unified_diagrams and page_h > page_w:
        x_start = int(page_w * 0.10)
        x_end = int(page_w * 0.92)
        dil_full = cv2.dilate(
            masked_bin[:, x_start:x_end], np.ones((7, 7), np.uint8), iterations=1
        )
        num_f, _, stats_f, _ = cv2.connectedComponentsWithStats(dil_full, connectivity=8)
        for i in range(1, num_f):
            cw = int(stats_f[i, cv2.CC_STAT_WIDTH])
            ch = int(stats_f[i, cv2.CC_STAT_HEIGHT])
            carea = int(stats_f[i, cv2.CC_STAT_AREA])
            cx = int(stats_f[i, cv2.CC_STAT_LEFT]) + x_start
            cy = int(stats_f[i, cv2.CC_STAT_TOP])
            ar_f = cw / float(max(ch, 1))
            if (
                ch >= int(page_h * 0.35)
                and cw >= int(page_w * 0.45)
                and 0.60 <= ar_f <= 1.65
                and cy >= int(page_h * 0.15)
            ):
                unified_diagrams.append(
                    BoundingBox(
                        x=cx,
                        y=cy,
                        w=cw,
                        h=ch,
                        stroke_density=float(carea) / max(cw * ch, 1),
                    )
                )

    # Final Text Lines: Retain lines NOT inside diagram boxes (check vertical & horizontal overlap)
    final_text_lines: list[BoundingBox] = []
    for l in text_candidates:
        l_cy = l.y + l.h / 2.0
        inside = any(
            (d.y - 5 <= l_cy <= d.y2 + 5) and (_intersection_over_box_area(l, d) > 0.35)
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
    """Handles OpenCV layout segmentation and fast hybrid handwriting transcription."""

    def __init__(
        self,
        trocr_model_name: str = TROCR_MODEL_NAME,
        cfg: SegmentationConfig = SEGMENTATION_CONFIG,
    ) -> None:
        self.trocr_model_name = trocr_model_name
        self.cfg = cfg
        self.device: str | None = None
        self.dtype: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._rapid_ocr: Any = None

    def _get_rapid_ocr(self) -> Any:
        if self._rapid_ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._rapid_ocr = RapidOCR()
            except Exception as exc:
                logger.warning("RapidOCR initialization failed: %s", exc)
        return self._rapid_ocr

    def _ensure_model_loaded(self) -> tuple[Any, Any]:
        """Lazy-load TrOCR processor and model onto the detected compute device."""
        if self._processor is None or self._model is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            from config import get_device_and_dtype

            self.device, self.dtype = get_device_and_dtype()
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
        self,
        page_path: str | Path,
        page_number: int,
        debug_dir: Path,
        save_line_crops: bool = True,
        fast_layout_only: bool = False,
    ) -> PageOCRResult:
        """Preprocess one page, suppress ruled lines, segment regions, and save debug crops."""
        src_path = Path(page_path)
        bgr = cv2.imread(str(src_path))
        if bgr is None:
            raise FileNotFoundError(f"Could not read page image: {src_path}")

        if fast_layout_only:
            h, w = bgr.shape[:2]
            if w > 900:
                scale = 900.0 / float(w)
                bgr = cv2.resize(bgr, (900, int(round(h * scale))), interpolation=cv2.INTER_AREA)
            layout_path = debug_dir / f"page_{page_number}_layout.png"
            cv2.imwrite(str(layout_path), bgr)
            return PageOCRResult(
                page_number=page_number,
                source_path=src_path,
                layout_image_path=layout_path,
                skew_angle_deg=0.0,
                text_lines=[],
                diagrams=[],
            )

        bgr = _resize_if_needed(bgr, self.cfg)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        skew_angle = _estimate_skew_angle(denoised, self.cfg)
        if abs(skew_angle) > 0.0:
            deskewed_bgr = _rotate_image(bgr, skew_angle)
            deskewed_gray = _rotate_image(denoised, skew_angle)
        else:
            deskewed_bgr = bgr
            deskewed_gray = denoised

        clean_binary, clean_for_ocr = _suppress_ruled_lines(
            deskewed_bgr, deskewed_gray, self.cfg, need_inpaint=save_line_crops
        )
        line_boxes, diagram_boxes = _segment_regions(
            clean_binary, self.cfg, deskewed_gray=deskewed_gray, page_number=page_number
        )

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
            if save_line_crops:
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

    def _transcribe_page_rapidocr(self, page_path: str | Path) -> str:
        """Whole-page RapidOCR transcription (runs 1 pass per page instead of 20 passes on crops)."""
        fast_ocr = self._get_rapid_ocr()
        if fast_ocr is None:
            return ""
        bgr = cv2.imread(str(page_path))
        if bgr is None:
            return ""
        h, w = bgr.shape[:2]
        if w > 1100:
            scale = 1100.0 / float(w)
            bgr = cv2.resize(bgr, (1100, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        res, _ = fast_ocr(bgr)
        if not res:
            return ""
        lines = [r[1].strip() for r in res if r[1] and len(r[1].strip()) >= 2]
        return "\n".join(lines)

    def _transcribe_pages_gemini(
        self,
        pages: list[PageOCRResult],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> bool:
        """Parallel multi-page Vision OCR using Gemini with retry and per-batch resilience."""
        import time
        from concurrent.futures import ThreadPoolExecutor
        from pydantic import BaseModel
        from google import genai
        from google.genai import types
        from config import GEMINI_FALLBACK_MODELS, GEMINI_MODEL_NAME, get_gemini_api_key

        active_key = get_gemini_api_key()
        if not active_key:
            return False

        class PageItem(BaseModel):
            page_number: int
            transcript: str

        class BatchTranscript(BaseModel):
            pages: list[PageItem]

        client = genai.Client(api_key=active_key)
        batch_size = 5
        batches = [pages[i : i + batch_size] for i in range(0, len(pages), batch_size)]

        def _process_batch(batch: list[PageOCRResult]) -> dict[int, str]:
            contents: list[Any] = [
                "You are an expert handwriting recognition specialist. Transcribe the student's handwritten English answers from each attached exam sheet page into clean, accurate, properly spelled digital text.\n"
                "CRITICAL RULES:\n"
                "1. IGNORE COVER / INTRODUCTION / COLLEGE METADATA: Completely ignore all cover page sections, university/college/school names (e.g. 'SCOPE GLOBAL SKILLS UNIVERSITY', 'ANSWER BOOK FOR MST / PUT EXAMINATION', 'MP Board'), student introduction/metadata fields (Roll No., Name, Course, Branch, Year/Sem, Subject Code, Date, Marks Awarded, Examiner/Invigilator Signature), printed Hindi headers, and crossed-out words. Focus ONLY on the student's main handwritten exam answers.\n"
                "2. QUESTION & PART NUMBERING:\n"
                "   - Standard sequential exams (Answer no. 01, Q2, Ans 3, etc.): start each main question on its own line with 'Q1:', 'Q2:', 'Q3:', etc.\n"
                "   - OBJECTIVE / MULTI-ITEM SUB-ANSWERS (MCQ, Fill in blanks, True/False, One-word items): put EACH sub-item on its OWN separate line in strict top-to-bottom order using '(i) ...', '(ii) ...', '(iii) ...', etc. (never merge multiple objective answers onto a single line).\n"
                "   - SHORT / LONG DESCRIPTIVE ANSWERS: keep each sentence/paragraph flowing naturally in clean continuous English text without breaking every visual line unnecessarily.\n"
                "   - Multi-part exams with 'Part-I', 'Part-II', 'Part-III' headings where question numbers restart in each part:\n"
                "     * Under Part-I: start Q.1(A) on its own line with 'Q1A:' and write its sub-answers line-by-line top-to-bottom as '(i) ...', '(ii) ...', '(iii) ...', '(iv) ...', '(v) ...' (even if the student wrote A-1..A-5). Start Q.1(B) / (B) on its own line with 'Q1B:' and write its sub-answers line-by-line as '(i) ...', '(ii) ...', '(iii) ...', '(iv) ...', '(v) ...' (if (iii)..(v) continue at the top of the next page before Part-II, write them as '(iii) ...', '(iv) ...', '(v) ...' without a new main question tag so they continue Q1B).\n"
                "     * Under Part-II: start each main answer ('A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'Q.1', etc.) on its own line with 'P2_Q1:', 'P2_Q2:', 'P2_Q3:', 'P2_Q4:', 'P2_Q5:', 'P2_Q6:', preserving the exact number written by the student.\n"
                "     * Under Part-III: start each main answer ('A-1', 'A-2', 'A-3', 'Q.1', 'Q.2', 'Q.3') on its own line with 'P3_Q1:', 'P3_Q2:', 'P3_Q3:' preserving the EXACT number written in the margin by the student (e.g., if the student writes 'A-2', output 'P3_Q2:' — NEVER renumber 'A-2' to 'P3_Q1:'; if the student writes 'A-3', output 'P3_Q3:'). Keep numbered sub-items inside an answer like '1-> Echo', '2-> printf', '3-> who', '4-> date', '5-> cal' on separate lines formatted as '(1) Echo...', '(2) printf...', etc.\n"
                "   - If a page starts with a continuation of an answer from the previous page before any new question marker appears, transcribe the continuation lines first before starting the next question tag.\n"
                "3. Convert messy cursive handwriting carefully into clean, accurate digital English words, numbers, and scientific formulas (e.g., V = IR, CO2, H2O). Do NOT hallucinate or invent words not written on the page.\n"
                "4. Return strict JSON mapping each page_number to its clean digital transcript."
            ]
            for p in batch:
                pil_img = Image.open(p.source_path).convert("RGB")
                if pil_img.width > 1500:
                    ratio = 1500.0 / float(pil_img.width)
                    pil_img = pil_img.resize((1500, int(pil_img.height * ratio)))
                contents.append(f"--- PAGE {p.page_number} ---")
                contents.append(pil_img)

            models_to_try = [GEMINI_MODEL_NAME, *GEMINI_FALLBACK_MODELS]
            for m_name in dict.fromkeys(models_to_try):
                try:
                    resp = client.models.generate_content(
                        model=m_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=BatchTranscript,
                            temperature=0.0,
                        ),
                    )
                    parsed = (
                        BatchTranscript.model_validate(resp.parsed)
                        if resp.parsed is not None
                        else BatchTranscript.model_validate_json(resp.text)
                    )
                    return {item.page_number: item.transcript.strip() for item in parsed.pages}
                except Exception as exc:
                    time.sleep(0.6)
                    logger.warning("Gemini OCR batch attempt (%s) failed: %s", m_name, exc)
            return {}

        page_map: dict[int, str] = {}
        try:
            with ThreadPoolExecutor(max_workers=max(1, min(4, len(batches)))) as pool:
                for batch_res in pool.map(_process_batch, batches):
                    page_map.update(batch_res)
        except Exception as exc:
            logger.warning("Gemini multi-page Vision OCR fallback triggered: %s", exc)

        for p in pages:
            p.transcript = page_map.get(p.page_number, "")
            if p.transcript and p.text_lines:
                clean_lines = [ln.strip() for ln in p.transcript.splitlines() if ln.strip()]
                for idx_l, reg in enumerate(p.text_lines):
                    if idx_l < len(clean_lines):
                        reg.text = clean_lines[idx_l]
        return any(bool(p.transcript) for p in pages)

    def transcribe_lines(self, crop_paths: list[Path]) -> list[str]:
        """Fast hybrid line transcription: cached RapidOCR with selective TrOCR fallback."""
        if not crop_paths:
            return []

        fast_ocr = self._get_rapid_ocr()
        transcripts: list[str] = [""] * len(crop_paths)
        needs_trocr_indices: list[int] = []

        for idx, path in enumerate(crop_paths):
            if fast_ocr is not None and path.exists():
                try:
                    res, _ = fast_ocr(str(path))
                    if res:
                        txt = " ".join(r[1].strip() for r in res if r[1].strip())
                        if len(txt) >= 2:
                            transcripts[idx] = txt
                            continue
                except Exception:
                    pass
            needs_trocr_indices.append(idx)

        # Only invoke heavy CPU TrOCR if RapidOCR missed most lines on a short 1-2 page submission
        hit_count = sum(1 for t in transcripts if t)
        if needs_trocr_indices and len(crop_paths) <= 20 and hit_count < len(crop_paths) // 2:
            try:
                import torch

                processor, model = self._ensure_model_loaded()
                batch_size = self.cfg.trocr_batch_size
                for start in range(0, len(needs_trocr_indices), batch_size):
                    batch_target_indices = needs_trocr_indices[start : start + batch_size]
                    batch_paths = [crop_paths[i] for i in batch_target_indices]
                    pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
                    inputs = processor(images=pil_images, return_tensors="pt")
                    pixel_values = inputs.pixel_values.to(device=self.device, dtype=self.dtype)

                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values,
                            max_new_tokens=min(64, self.cfg.trocr_max_new_tokens),
                        )
                    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
                    for d_idx, text in enumerate(decoded):
                        target_pos = batch_target_indices[d_idx]
                        transcripts[target_pos] = text.strip()
            except Exception as exc:
                logger.warning("TrOCR fallback failed: %s", exc)

        return transcripts

    def process_pages(
        self,
        page_paths: list[str | Path],
        debug_dir: str | Path = "debug_crops",
        online_fast: bool = True,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> OCRPipelineResult:
        """Segment all input pages and run fast Vision/RapidOCR/TrOCR transcription."""
        debug_root = Path(debug_dir)
        debug_root.mkdir(parents=True, exist_ok=True)

        total_pages = len(page_paths)
        is_single_or_pair = total_pages <= 2
        save_crops = is_single_or_pair and not online_fast

        pages: list[PageOCRResult] = []
        all_line_regions: list[SegmentedRegion] = []

        for idx, page_path in enumerate(page_paths, start=1):
            if progress_callback:
                pct = int(5 + ((idx - 1) / max(total_pages, 1)) * 45)
                msg = (
                    f"Evaluating Page {idx} of {total_pages}..."
                    if idx == 1
                    else f"Page {idx - 1} Done ✔ → Now Evaluating Page {idx} of {total_pages}..."
                )
                progress_callback(
                    {
                        "type": "progress",
                        "step": "page_segment",
                        "page": idx,
                        "total_pages": total_pages,
                        "percent": pct,
                        "message": msg,
                    }
                )

            page_result = self.segment_page(
                page_path,
                page_number=idx,
                debug_dir=debug_root,
                save_line_crops=save_crops,
                fast_layout_only=False,
            )
            pages.append(page_result)
            all_line_regions.extend(page_result.text_lines)

            if progress_callback:
                pct = int(5 + (idx / max(total_pages, 1)) * 45)
                next_part = (
                    f" → Now Evaluating Page {idx + 1} of {total_pages}..."
                    if idx < total_pages
                    else " → Layout Segmentation Complete!"
                )
                progress_callback(
                    {
                        "type": "progress",
                        "step": "page_done",
                        "page": idx,
                        "total_pages": total_pages,
                        "percent": pct,
                        "lines_detected": len(page_result.text_lines),
                        "diagrams_detected": len(page_result.diagrams),
                        "message": f"Page {idx} of {total_pages} Done ✔ ({len(page_result.text_lines)} lines, {len(page_result.diagrams)} diagram){next_part}",
                    }
                )

        # Path 1: Fast Parallel Gemini Vision OCR (Online)
        if online_fast:
            if progress_callback:
                progress_callback(
                    {
                        "type": "progress",
                        "step": "ocr_vision",
                        "page": total_pages,
                        "total_pages": total_pages,
                        "percent": 58,
                        "message": f"Running Multimodal Vision Handwriting OCR across {total_pages} page(s)...",
                    }
                )
            if self._transcribe_pages_gemini(pages, progress_callback=progress_callback):
                # Fill any single missed page with RapidOCR if needed
                for page in pages:
                    if not page.transcript.strip() and page.text_lines:
                        page.transcript = self._transcribe_page_rapidocr(page.source_path)
                full_transcript = "\n".join(p.transcript for p in pages if p.transcript)
                return OCRPipelineResult(
                    pages=pages,
                    full_transcript=full_transcript,
                    debug_dir=debug_root,
                )

        # Path 2: Offline Multi-page Whole-Page RapidOCR (1 pass per page instead of 200 crop calls)
        if not save_crops:
            for idx, page in enumerate(pages, start=1):
                if progress_callback:
                    progress_callback(
                        {
                            "type": "progress",
                            "step": "ocr_local",
                            "page": idx,
                            "total_pages": total_pages,
                            "percent": int(55 + (idx / max(total_pages, 1)) * 20),
                            "message": f"Transcribing Page {idx} of {total_pages} (Local OCR)...",
                        }
                    )
                page.transcript = self._transcribe_page_rapidocr(page.source_path)
            full_transcript = "\n".join(p.transcript for p in pages if p.transcript)
            return OCRPipelineResult(
                pages=pages,
                full_transcript=full_transcript,
                debug_dir=debug_root,
            )

        # Path 3: Offline 1-2 page crop-level RapidOCR + TrOCR fallback
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
