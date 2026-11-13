"""Generate realistic ruled handwritten exam answer sheets (sample_page1.jpg, sample_page2.jpg)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE_W = 1400
PAGE_H = 1850
INK_COLOR = (30, 40, 78)  # Blue-black ballpoint pen (RGB)
RULED_LINE_COLOR = (178, 198, 222)
MARGIN_LINE_COLOR = (225, 175, 175)
PAPER_BASE_RGB = (249, 247, 241)


def _load_handwriting_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Locate a handwriting-style TTF font across Windows/Linux/macOS with fallback."""
    candidates = [
        r"C:\Windows\Fonts\segoepr.ttf",
        r"C:\Windows\Fonts\Inkfree.ttf",
        r"C:\Windows\Fonts\comic.ttf",
        r"C:\Windows\Fonts\segoesc.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    ]
    for path_str in candidates:
        if Path(path_str).exists():
            return ImageFont.truetype(path_str, size=size)
    return ImageFont.load_default()


def _create_ruled_paper(seed: int) -> Image.Image:
    """Create off-white notebook paper with horizontal ruled lines, margin, and subtle texture."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=2.2, size=(PAGE_H, PAGE_W, 3))
    arr = np.clip(np.array(PAPER_BASE_RGB, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Top header divider
    draw.line([(0, 170), (PAGE_W, 170)], fill=(165, 185, 215), width=2)
    # Left vertical margin line
    draw.line([(125, 0), (125, PAGE_H)], fill=MARGIN_LINE_COLOR, width=2)

    # Horizontal ruled lines spaced 76 px apart
    for y in range(250, PAGE_H - 90, 76):
        draw.line([(0, y), (PAGE_W, y)], fill=RULED_LINE_COLOR, width=2)

    return img


def _draw_hand_line(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    rng: np.random.Generator,
    width: int = 4,
) -> None:
    """Draw a slightly imperfect hand-drawn straight stroke between two points."""
    x1, y1 = p1
    x2, y2 = p2
    dist = max(1.0, math.hypot(x2 - x1, y2 - y1))
    steps = max(4, int(dist / 24))
    pts = []
    for i in range(steps + 1):
        t = i / steps
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        if 0 < i < steps:
            px += float(rng.uniform(-1.2, 1.2))
            py += float(rng.uniform(-1.2, 1.2))
        pts.append((px, py))
    draw.line(pts, fill=INK_COLOR, width=width)


def _draw_circuit_diagram(
    draw: ImageDraw.ImageDraw,
    top_left: tuple[int, int],
    label_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    rng: np.random.Generator,
) -> None:
    """Draw a labeled hand-drawn style series electric circuit diagram."""
    ox, oy = top_left
    left, right = ox + 80, ox + 660
    top, bottom = oy + 70, oy + 410

    # Top wire broken by switch
    _draw_hand_line(draw, (left, top), (ox + 290, top), rng, width=4)
    # Switch open/closed angled lever
    draw.ellipse((ox + 285, top - 6, ox + 297, top + 6), fill=INK_COLOR)
    draw.ellipse((ox + 375, top - 6, ox + 387, top + 6), fill=INK_COLOR)
    _draw_hand_line(draw, (ox + 292, top), (ox + 372, top - 34), rng, width=4)
    _draw_hand_line(draw, (ox + 382, top), (right, top), rng, width=4)
    draw.text((ox + 295, top - 80), "Switch", font=label_font, fill=INK_COLOR)

    # Right vertical wire with Ammeter circle (A)
    mid_y = (top + bottom) // 2
    _draw_hand_line(draw, (right, top), (right, mid_y - 42), rng, width=4)
    draw.ellipse(
        (right - 42, mid_y - 42, right + 42, mid_y + 42),
        outline=INK_COLOR,
        width=4,
    )
    draw.text((right - 14, mid_y - 24), "A", font=label_font, fill=INK_COLOR)
    _draw_hand_line(draw, (right, mid_y + 42), (right, bottom), rng, width=4)
    draw.text((right + 55, mid_y - 20), "Ammeter", font=label_font, fill=INK_COLOR)

    # Bottom wire with zig-zag Resistor
    res_start, res_end = ox + 260, ox + 480
    _draw_hand_line(draw, (left, bottom), (res_start, bottom), rng, width=4)
    peaks = 6
    step_x = (res_end - res_start) / peaks
    prev = (res_start, bottom)
    for idx in range(1, peaks):
        nx = int(round(res_start + idx * step_x))
        ny = bottom + (-22 if idx % 2 == 1 else 22)
        _draw_hand_line(draw, prev, (nx, ny), rng, width=4)
        prev = (nx, ny)
    _draw_hand_line(draw, prev, (res_end, bottom), rng, width=4)
    _draw_hand_line(draw, (res_end, bottom), (right, bottom), rng, width=4)
    draw.text((ox + 305, bottom + 32), "Resistor (R)", font=label_font, fill=INK_COLOR)

    # Left vertical wire with Battery symbol (long + / short thick -)
    _draw_hand_line(draw, (left, top), (left, mid_y - 20), rng, width=4)
    _draw_hand_line(draw, (left - 34, mid_y - 20), (left + 34, mid_y - 20), rng, width=4)
    _draw_hand_line(draw, (left - 18, mid_y + 12), (left + 18, mid_y + 12), rng, width=7)
    _draw_hand_line(draw, (left, mid_y + 12), (left, bottom), rng, width=4)
    draw.text((left + 44, mid_y - 44), "+", font=label_font, fill=INK_COLOR)
    draw.text((left + 44, mid_y + 4), "-", font=label_font, fill=INK_COLOR)
    draw.text((left - 125, mid_y - 22), "Battery", font=label_font, fill=INK_COLOR)

    # Current arrow along top wire
    _draw_hand_line(draw, (ox + 490, top - 16), (ox + 545, top - 16), rng, width=3)
    _draw_hand_line(draw, (ox + 530, top - 25), (ox + 545, top - 16), rng, width=3)
    _draw_hand_line(draw, (ox + 530, top - 7), (ox + 545, top - 16), rng, width=3)
    draw.text((ox + 495, top - 56), "I", font=label_font, fill=INK_COLOR)


def _apply_scan_skew(img: Image.Image, angle_deg: float) -> Image.Image:
    """Apply subtle scanner/camera skew using OpenCV with clean paper border."""
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(PAPER_BASE_RGB[2], PAPER_BASE_RGB[1], PAPER_BASE_RGB[0]),
    )
    return Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))


def generate_sample_pages(output_dir: Path | str = ".") -> tuple[Path, Path]:
    """Create sample_page1.jpg and sample_page2.jpg in `output_dir`."""
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    body_font = _load_handwriting_font(34)
    label_font = _load_handwriting_font(28)

    # -----------------------------------------------------------------------
    # Page 1: Question 1 (Text-only STEM answer on ruled paper)
    # -----------------------------------------------------------------------
    rng1 = np.random.default_rng(42)
    page1 = _create_ruled_paper(seed=42)
    draw1 = ImageDraw.Draw(page1)

    page1_lines = [
        "Q1. Newton's Second Law of Motion states that",
        "the rate of change of momentum of a body is",
        "directly proportional to the applied unbalanced",
        "force and acts in the direction of the force.",
        "Mathematically, F = m * a, where m is mass",
        "and a is acceleration. The SI unit of force",
        "is the Newton (N), equal to 1 kg m/s^2.",
    ]
    for idx, line_text in enumerate(page1_lines):
        y_baseline = 250 + idx * 76
        x_pos = 155 + int(rng1.integers(-4, 6))
        y_pos = y_baseline - 48 + int(rng1.integers(-2, 3))
        draw1.text((x_pos, y_pos), line_text, font=body_font, fill=INK_COLOR)

    page1_skewed = _apply_scan_skew(page1, angle_deg=1.8)
    page1_path = out_root / "sample_page1.jpg"
    page1_skewed.save(page1_path, quality=94)

    # -----------------------------------------------------------------------
    # Page 2: Question 2 (Text explanation + hand-drawn labeled circuit diagram)
    # -----------------------------------------------------------------------
    rng2 = np.random.default_rng(99)
    page2 = _create_ruled_paper(seed=99)
    draw2 = ImageDraw.Draw(page2)

    page2_lines = [
        "Q2. In a simple series electric circuit, the",
        "battery provides voltage across the closed loop.",
        "The resistor opposes electric current following",
        "Ohm's Law V = I * R, and the ammeter is",
        "connected in series to measure current.",
    ]
    for idx, line_text in enumerate(page2_lines):
        y_baseline = 250 + idx * 76
        x_pos = 155 + int(rng2.integers(-4, 6))
        y_pos = y_baseline - 48 + int(rng2.integers(-2, 3))
        draw2.text((x_pos, y_pos), line_text, font=body_font, fill=INK_COLOR)

    _draw_circuit_diagram(
        draw2,
        top_left=(290, 740),
        label_font=label_font,
        rng=rng2,
    )

    page2_skewed = _apply_scan_skew(page2, angle_deg=-1.5)
    page2_path = out_root / "sample_page2.jpg"
    page2_skewed.save(page2_path, quality=94)

    return page1_path, page2_path


if __name__ == "__main__":
    p1, p2 = generate_sample_pages(Path(__file__).resolve().parent)
    print(f"Generated sample exam pages:\n  - {p1}\n  - {p2}")
