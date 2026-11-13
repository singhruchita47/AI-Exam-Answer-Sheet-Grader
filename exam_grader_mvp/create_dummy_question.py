"""Generate realistic sample question paper images (sample_question_paper.jpg) for testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE_W = 1200
PAGE_H = 1000
TEXT_COLOR = (25, 30, 45)
HEADER_COLOR = (15, 23, 42)
DIVIDER_COLOR = (180, 195, 215)
PAPER_BASE = (252, 251, 248)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def generate_sample_question_paper(output_path: Path | str = "sample_question_paper.jpg") -> Path:
    """Create a realistic printed Question Paper image."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Subtle paper texture
    rng = np.random.default_rng(101)
    noise = rng.normal(loc=0.0, scale=1.5, size=(PAGE_H, PAGE_W, 3))
    arr = np.clip(np.array(PAPER_BASE, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)

    title_font = _load_font(28, bold=True)
    meta_font = _load_font(18, bold=False)
    q_title_font = _load_font(22, bold=True)
    body_font = _load_font(20, bold=False)

    # Header
    draw.text((PAGE_W // 2 - 250, 60), "DEPARTMENT OF HIGHER SECONDARY EDUCATION", font=title_font, fill=HEADER_COLOR)
    draw.text((PAGE_W // 2 - 180, 105), "Mid-Term Physics & Circuits Examination", font=meta_font, fill=(70, 80, 95))
    draw.text((100, 155), "Subject: Physics", font=meta_font, fill=HEADER_COLOR)
    draw.text((PAGE_W - 320, 155), "Max Marks: 10 | Time: 1 Hour", font=meta_font, fill=HEADER_COLOR)

    # Header horizontal line
    draw.line([(90, 190), (PAGE_W - 90, 190)], fill=DIVIDER_COLOR, width=2)
    draw.line([(90, 194), (PAGE_W - 90, 194)], fill=DIVIDER_COLOR, width=1)

    # Question 1
    y = 240
    draw.text((100, y), "Q1. [5 Marks]", font=q_title_font, fill=HEADER_COLOR)
    q1_lines = [
        "State Newton's Second Law of Motion, write its mathematical expression,",
        "and define the SI unit of force.",
    ]
    for line in q1_lines:
        y += 34
        draw.text((140, y), line, font=body_font, fill=TEXT_COLOR)

    # Question 2
    y += 80
    draw.text((100, y), "Q2. [5 Marks]", font=q_title_font, fill=HEADER_COLOR)
    q2_lines = [
        "Draw a neat labeled diagram of a simple series electric circuit to verify Ohm's Law,",
        "and explain the role of the resistor and ammeter.",
    ]
    for line in q2_lines:
        y += 34
        draw.text((140, y), line, font=body_font, fill=TEXT_COLOR)

    # Instructions box at bottom
    y += 110
    draw.rectangle([(90, y), (PAGE_W - 90, y + 140)], outline=DIVIDER_COLOR, width=1, fill=(245, 248, 252))
    draw.text((110, y + 16), "Instructions to Candidates:", font=_load_font(18, bold=True), fill=HEADER_COLOR)
    draw.text((110, y + 50), "• Answer all questions clearly in blue or black ink.", font=meta_font, fill=(80, 90, 105))
    draw.text((110, y + 80), "• Diagrams must be neatly drawn with sharp pencil/pen and properly labeled.", font=meta_font, fill=(80, 90, 105))

    img.save(dest, quality=95)
    return dest


if __name__ == "__main__":
    p = generate_sample_question_paper(Path(__file__).resolve().parent / "sample_question_paper.jpg")
    print(f"Generated sample question paper at: {p}")
