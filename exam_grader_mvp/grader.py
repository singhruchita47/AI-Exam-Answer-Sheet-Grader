"""Question boundary segmentation, semantic similarity, VLM diagram evaluation, and rubric grading."""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

from config import (
    EMBEDDING_MODEL_NAME,
    GEMINI_API_KEY,
    GEMINI_FALLBACK_MODELS,
    GEMINI_FAST_GRADING_MODELS,
    GEMINI_MODEL_NAME,
    GRADING_CONFIG,
    LOCAL_VLM_MODEL_NAME,
    GradingConfig,
    generate_content_fast,
    get_gemini_api_key,
)

if TYPE_CHECKING:
    from ocr_engine import OCRPipelineResult, PageOCRResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class KeyPoint(BaseModel):
    """Weighted rubric sub-point or step."""

    point_id: str
    description: str
    weight: float = Field(default=1.0, ge=0.0)


class RubricQuestion(BaseModel):
    """Single question definition in the exam rubric."""

    question_id: str
    question_text: str
    question_type: Literal[
        "mcq", "fill_blank", "one_word", "short_answer", "long_answer"
    ] = "short_answer"
    model_answer: str
    key_points: list[KeyPoint]
    requires_diagram: bool = False
    diagram_criteria: str = ""
    max_marks: float = Field(default=5.0, ge=0.0)


class ExamRubric(BaseModel):
    """Top-level exam rubric loaded from JSON."""

    exam_title: str = "Automated Exam Grading"
    detected_subject: str = ""
    printed_max_marks: float = 0.0
    reference_mode: Literal["dual_reference", "auto_inferred"] = "auto_inferred"
    questions: list[RubricQuestion]


class DiagramEvaluation(BaseModel):
    """Structured VLM evaluation of a hand-drawn student diagram."""

    is_relevant: bool
    labels_accurate: bool
    structure_score: float = Field(ge=0.0, le=1.0)
    critique: str


class StepBreakdownItem(BaseModel):
    """Step-wise rubric grading result for an individual key point."""

    point_id: str
    description: str
    max_point_marks: float = Field(ge=0.0)
    awarded_point_marks: float = Field(ge=0.0)
    status: Literal["Hit", "Partial", "Missed"]
    similarity_score: float = Field(ge=0.0, le=1.0)
    notes: str


class LLMGradeOutput(BaseModel):
    """Structured output returned by Gemini during step-wise grading."""

    step_wise_breakdown: list[StepBreakdownItem]
    missing_key_points: list[str]
    marks_awarded: float = Field(ge=0.0)
    student_feedback: str


class QuestionGradeResult(BaseModel):
    """Complete grading output for one question."""

    question_id: str
    question_type: str = "short_answer"
    student_answer: str
    is_attempted: bool = True
    page_numbers: list[int] = Field(default_factory=list)
    page_texts: dict[str, str] = Field(default_factory=dict)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    diagram_path: str | None = None
    diagram_evaluation: DiagramEvaluation | None = None
    step_wise_breakdown: list[StepBreakdownItem]
    missing_key_points: list[str]
    marks_awarded: float = Field(ge=0.0)
    max_marks: float = Field(gt=0.0)
    student_feedback: str


class ExamGradingReport(BaseModel):
    """Final grading report across all rubric questions."""

    exam_title: str
    mode: Literal["online_gemini", "offline_local"]
    strictness_level: int = Field(default=5, ge=1, le=10)
    total_marks_awarded: float
    total_max_marks: float
    printed_max_marks: float = 0.0
    mapped_answers: dict[str, str]
    question_results: list[QuestionGradeResult]


# ---------------------------------------------------------------------------
# Regex Question Boundary Segmentation
# ---------------------------------------------------------------------------
_PART_PREFIX_PATTERN = re.compile(
    r"^\s*(?:P(?:art)?[\s_\-]*(II|III|I|2|3|1)[\s_\-]*)Q(?:uestion)?\s*(?:no\.?|number)?\s*[\.\-:#]?\s*0*([0-9]{1,2}[ABab]?)\s*[\)\.\-:#]*(?:\s+(.*))?$",
    re.IGNORECASE,
)
_SUB_AB_PATTERN = re.compile(
    r"^\s*Q(?:uestion)?\s*[\.\-:#]?\s*0*1\s*[:\-#]?\s*\(?\s*([ABab])\s*\)?\s*[\)\.\-:#]*(?:\s+(.*))?$",
    re.IGNORECASE,
)

_EXPLICIT_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Matches: प्रश्न 1, प्रश्न क्रमांक 1, प्रश्न क्र. 1, प्र. 1, प्र० 1, प्र.क्र. 1, प्र 1
    re.compile(
        r"^\s*(?:प्रश्न|प्र[\.\s\:\-]*|प्र(?:श्न)?[\s\.\:\-]*क्र(?:मांक|\.)?|प्र[\s\.\:\-]*सं(?:ख्या|\.)?)\s*0*([0-9]{1,2})\s*(?:[:\)\-]|\.(?!\.))*\s*(.*)$",
        re.IGNORECASE,
    ),
    # Matches: उत्तर 1, उत्तर क्रमांक 1, उत्तर क्र. 1, उ. 1, उ० 1, उ.क्र. 1, उ 1
    re.compile(
        r"^\s*(?:उत्तर|उ[\.\s\:\-]*|उ(?:त्तर)?[\s\.\:\-]*क्र(?:मांक|\.)?|उ[\s\.\:\-]*सं(?:ख्या|\.)?)\s*0*([0-9]{1,2})\s*(?:[:\)\-]|\.(?!\.))*\s*(.*)$",
        re.IGNORECASE,
    ),
    # Matches: Q1, Q.1, Q. No. 1, Q. No. 23 [or], Question no. - 01, Q1), Question 1:
    re.compile(
        r"^\s*Q(?:uestion)?[\.\s]*(?:no\.?|number)?\s*[\.\-:#]?\s*0*([0-9]{1,2}|[!lI|])\s*(?:\[\s*or\s*\]|\(\s*or\s*\))?\s*(?:[:\)\-]|\.(?!\.))*\s*(.*)$",
        re.IGNORECASE,
    ),
    # Matches: Answer no. - 01, Answer no.-06 [or], Ans. No. 2, Ans 1, A-1, A-2
    re.compile(
        r"^\s*(?:(?:ans(?:wer)?|an[a-z]{2,5})[\.\s]*(?:no\.?|m?no\.?|number)?[\.\-:#\s]*|[Aa]\s*(?:(?:no\.?|number)[\.\-:#\s]*|[\-\.:#]\s*))0*([0-9]{1,2}|[!lI|])\s*(?:\[\s*or\s*\]|\(\s*or\s*\))?\s*(?:[:\)\-]|\.(?!\.))*\s*(.*)$",
        re.IGNORECASE,
    ),
    # Also matches Ans1, Ans2 (no space) with digits
    re.compile(
        r"^\s*(?:ans(?:wer)?|[a-z]{0,2}[ns]|[a-z]?s|[Aa])0*([0-9]{1,2})\s*(?:\[?\s*or\s*\]?)?\s*(?:[:\)\-]|\.(?!\.))*\s*(.*)$",
        re.IGNORECASE,
    ),
)

_BARE_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Matches: (1), (2), (10)
    re.compile(r"^\s*\(0*([0-9]{1,2}|[!lI|])\)\s*(.*)$"),
    # Matches: 1), 2:, 1 ) or 1 - Text
    re.compile(r"^\s*0*([0-9]{1,2})\s*[\)\:\-–]\s*(.*)$"),
    # Matches: 1. Text... (requires dot + space to avoid decimals like 1.5)
    re.compile(r"^\s*0*([0-9]{1,2})\.\s+(.*)$"),
)

_QUESTION_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    *_EXPLICIT_MARKER_PATTERNS,
    *_BARE_MARKER_PATTERNS,
)


_TROCR_HALLUCINATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Categories\s*\d{4}.*?(?:births|deaths|actors|state)", re.IGNORECASE),
    re.compile(r"(?:American\s+male\s+(?:film|stage|television)\s+actors|People\s+from\s+New\s+Brunswick)", re.IGNORECASE),
    re.compile(r"(?:e2){3,}|(?:0\s*){4,}", re.IGNORECASE),
    re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE),
    re.compile(r"qualads\s+first\s+hugci.*?views\s+of\s+all\s+of\s+ten\s+centi", re.IGNORECASE),
    re.compile(r"(?:SCOPE\s+GLOBAL\s+SKILLS\s+UNIVERSITY|CENTRAL\s+INDIA'S\s+1st\s+SKILLS\s+UNIVERSITY|ANSWER\s+BOOK\s+FOR\s+MST|Marks\s+Awared|Marks\s+Obtained|Invigilators\s*/\s*Internal|Examiner\s+Name)", re.IGNORECASE),
)


def _normalize_qid(qid: str) -> str:
    """Normalize question IDs across rubric and OCR parser (e.g. PartII_Q1 -> P2_Q1, Q.1(A) -> Q1A)."""
    q = qid.strip()
    q = re.sub(
        r"^(?:Part|Section|P)[\s_\-]*I(?:1)?[\s_\-]*Q\.?\s*1\s*\(?\s*([AB])\s*\)?$",
        r"Q1\1",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(r"^Q\.?\s*1\s*\(?\s*([AB])\s*\)?$", r"Q1\1", q, flags=re.IGNORECASE)
    q = re.sub(
        r"^(?:Part|Section|P)[\s_\-]*(?:II|2)[\s_\-]*Q(?:uestion)?[\s_\-\.]*0*(\d+)$",
        r"P2_Q\1",
        q,
        flags=re.IGNORECASE,
    )
    q = re.sub(
        r"^(?:Part|Section|P)[\s_\-]*(?:III|3)[\s_\-]*Q(?:uestion)?[\s_\-\.]*0*(\d+)$",
        r"P3_Q\1",
        q,
        flags=re.IGNORECASE,
    )
    return q.upper()


def _clean_ocr_handwriting_text(text: str) -> str:
    """Fix common OCR misrecognitions while preserving all English and Hindi/Devanagari characters."""
    # Strip known TrOCR Wikipedia/IAM noise patterns
    for pat in _TROCR_HALLUCINATION_PATTERNS:
        text = pat.sub(" ", text)
    # Fix IDhm's -> Ohm's
    text = re.sub(r"\bI?Dhm(?:'s)?\b", "Ohm's", text, flags=re.IGNORECASE)
    # Fix appropriational -> proportional
    text = re.sub(r"\bappropriational\b", "proportional", text, flags=re.IGNORECASE)
    # Fix following -> flowing
    text = re.sub(r"\bfollowing through\b", "flowing through", text, flags=re.IGNORECASE)
    # Fix jenergy -> energy
    text = re.sub(r"\bjenergy\b", "energy", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _match_question_marker(
    line: str, explicit_only: bool = False
) -> tuple[str, str] | None:
    """Return normalized (question_id, remainder_text) if line starts with a marker."""
    stripped = line.strip()
    if not stripped:
        return None
    # Priority 1: Check Part-prefixed pattern (P2_Q1, P3_Q2, PartII_Q1, Part-II Q.1)
    m_part = _PART_PREFIX_PATTERN.match(stripped)
    if m_part and m_part.lastindex and m_part.lastindex >= 2:
        part_raw = (m_part.group(1) or "").strip().upper()
        q_raw = (m_part.group(2) or "").strip().upper()
        rem_raw = (m_part.group(3) or "").strip() if m_part.lastindex >= 3 else ""
        part_num = 3 if part_raw in ("III", "3") else (2 if part_raw in ("II", "2") else 1)
        rem_clean = _clean_ocr_handwriting_text(rem_raw)
        if part_num == 1 and q_raw in ("1A", "1B"):
            return f"Q{q_raw}", rem_clean
        return f"P{part_num}_Q{q_raw}", rem_clean

    # Priority 2: Check Q1A / Q1B / Q.1(A) / Q.1(B)
    m_ab = _SUB_AB_PATTERN.match(stripped)
    if m_ab:
        sub_ab = (m_ab.group(1) or "").strip().upper()
        rem_raw = (m_ab.group(2) or "").strip() if m_ab.lastindex and m_ab.lastindex >= 2 else ""
        return f"Q1{sub_ab}", _clean_ocr_handwriting_text(rem_raw)

    patterns = _EXPLICIT_MARKER_PATTERNS if explicit_only else _QUESTION_MARKER_PATTERNS
    for pattern in patterns:
        match = pattern.match(stripped)
        if match:
            raw_num = match.group(1).strip()
            if raw_num in ("!", "l", "I", "|"):
                q_num = 1
            else:
                try:
                    q_num = int(raw_num)
                except ValueError:
                    continue
            if 1 <= q_num <= 50:
                raw_rem = (match.group(2) or "").strip() if match.lastindex and match.lastindex >= 2 else ""
                if raw_rem in ("..", "(..)", ".") or raw_rem.startswith("..") or raw_rem.startswith("(..)"):
                    remainder = raw_rem
                else:
                    m_dot = re.match(r"^[:\-–!|]\s*(\(\.\.\)|\.\.)\s*$", raw_rem)
                    if m_dot:
                        remainder = m_dot.group(1)
                    else:
                        remainder = re.sub(
                            r"^(?:\[?\s*or\s*\]?|\(\s*or\s*\)|[\.\(\)\-–!|:\s])+(?:\d{1,2}\s*[\.\)]\s*)?",
                            "",
                            raw_rem,
                            flags=re.IGNORECASE,
                        ).strip()
                remainder = _clean_ocr_handwriting_text(remainder)
                if remainder:
                    nested = _match_question_marker(remainder, explicit_only=True)
                    if nested is not None and nested[0] == f"Q{q_num}":
                        remainder = nested[1]
                return f"Q{q_num}", remainder
    return None


_INLINE_Q_SPLIT_RE = re.compile(
    r"(?<=\S)\s+(?=(?:"
    r"P(?:art)?[\s_\-]*(?:II|III|I|2|3|1)[\s_\-]*Q(?:uestion)?\s*(?:no\.?)?[\.\-:#]?\s*0*[0-9]{1,2}\s*[:\-]"
    r"|Q(?:uestion)?[\.\s]*(?:no\.?|number)?\s*[\.\-:#]?\s*0*[0-9]{1,2}\s*(?:\[\s*or\s*\]|\(\s*or\s*\))?\s*[:\-]"
    r"|(?:Answer|Ans)[\.\s]*(?:no\.?|number)\s*[\.\-:#\s]*0*[0-9]{1,2}\s*(?:\[\s*or\s*\]|\(\s*or\s*\))?(?:\s*[:\-]|(?=\s+\S))"
    r"|(?:प्रश्न|उत्तर)[\s\.\:\-]*(?:क्र(?:मांक|\.)?)?\s*0*[0-9]{1,2}\s*[:\-]"
    r"))",
    re.IGNORECASE,
)


def _split_inline_question_lines(text: str) -> list[str]:
    """Break inline question boundary tags (e.g. '...end. Q11: Answer no. 11 ...') onto separate lines."""
    lines: list[str] = []
    for raw_ln in text.splitlines():
        s = raw_ln.strip()
        if not s:
            continue
        for sub in _INLINE_Q_SPLIT_RE.split(s):
            sub_s = sub.strip()
            if sub_s:
                lines.append(sub_s)
    return lines


def parse_questions(full_text: str) -> dict[str, str]:
    """Split continuous reconstructed OCR text into a {question_id: answer_text} map."""
    mapped: dict[str, list[str]] = {}
    current_qid: str = "Q1"  # Default start to Q1 so top text is never discarded
    lines = _split_inline_question_lines(full_text)
    has_explicit = any(_match_question_marker(l, explicit_only=True) is not None for l in lines)

    for line in lines:
        marker = _match_question_marker(line, explicit_only=has_explicit)
        if marker is not None:
            current_qid, remainder = marker
            mapped.setdefault(current_qid, [])
            if remainder:
                mapped[current_qid].append(remainder)
        else:
            cleaned_line = _clean_ocr_handwriting_text(line)
            if cleaned_line:
                mapped.setdefault(current_qid, []).append(cleaned_line)

    if not mapped and full_text.strip():
        return {"Q1": _clean_ocr_handwriting_text(" ".join(lines))}

    return {qid: " ".join(parts).strip() for qid, parts in mapped.items()}


def _join_answer_parts(parts: list[str]) -> str:
    """Format answer text: keep objective sub-items line-by-line top-to-bottom; keep prose continuous."""
    non_empty = [p.strip() for p in parts if p and p.strip()]
    if not non_empty:
        return ""

    # Already split line-by-line by OCR (e.g., (i) ..., (ii) ..., 1) ..., a) ..., (क) ..., (अ) ...)
    line_start_re = re.compile(
        r"^\s*(?:\(\s*(?:[ivx]{1,4}|\d{1,2}|[a-f]|[कखगघङचछजझञटठडढणतथदधनपफबभम]|[अआइईउऊऋएऐओऔ])\s*\)|(?:[ivx]{1,4}|\d{1,2}|[a-f]|[कखगघङचछजझञटठडढणतथदधनपफबभम]|[अआइईउऊऋएऐओऔ])\s*[\)\.])\s+",
        re.IGNORECASE,
    )
    sub_line_count = sum(1 for ln in non_empty if line_start_re.match(ln))
    if sub_line_count >= 2:
        return "\n".join(non_empty)

    joined_space = " ".join(non_empty)
    # Prefer roman or numeric sub-item markers so MCQ option letters (a)..(d) remain attached to their item
    roman_num_re = re.compile(
        r"(?:^|\s)(\(\s*(?:[ivx]{1,4}|\d{1,2})\s*\))\s*",
        re.IGNORECASE,
    )
    rn_matches = list(roman_num_re.finditer(joined_space))
    if len(rn_matches) >= 2:
        first_tok = rn_matches[0].group(1).strip("() ").lower()
        second_tok = rn_matches[1].group(1).strip("() ").lower()
        looks_sequential = (
            (first_tok in ("i", "1"))
            or (first_tok in ("ii", "2") and second_tok in ("iii", "3"))
        )
        if looks_sequential:
            formatted = roman_num_re.sub(lambda m: f"\n{m.group(1).strip()} ", joined_space)
            clean_lines = [ln.strip() for ln in formatted.splitlines() if ln.strip()]
            return "\n".join(clean_lines)

    # Fallback to alpha markers only if no roman/numeric markers exist
    inline_paren_re = re.compile(
        r"(?:^|\s)(\(\s*[a-f]\s*\))\s+",
        re.IGNORECASE,
    )
    paren_matches = list(inline_paren_re.finditer(joined_space))
    if len(paren_matches) >= 2:
        first_tok = paren_matches[0].group(1).strip("() ").lower()
        second_tok = paren_matches[1].group(1).strip("() ").lower()
        looks_sequential = (
            (first_tok == "a")
            or (first_tok == "b" and second_tok == "c")
        )
        if looks_sequential:
            formatted = inline_paren_re.sub(lambda m: f"\n{m.group(1).strip()} ", joined_space)
            clean_lines = [ln.strip() for ln in formatted.splitlines() if ln.strip()]
            return "\n".join(clean_lines)

    return joined_space.strip()


def parse_questions_with_page_texts(
    pages: list["PageOCRResult"],
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, dict[str, str]]]:
    """Parse questions across ordered pages while tracking page numbers and per-page text per question."""
    mapped: dict[str, list[str]] = {}
    question_pages: dict[str, list[int]] = {}
    page_texts_list: dict[str, dict[str, list[str]]] = {}
    current_qid: str = "Q1"  # Default start to Q1 so opening text is NEVER lost
    current_part: int | None = None
    explicit_opened: set[str] = set()
    int_to_roman = {1: "i", 2: "ii", 3: "iii", 4: "iv", 5: "v", 6: "vi"}

    part_header_re = re.compile(
        r"^\s*\[?\s*(?:Part|Section)[\s\-_]*([I123V]+)\s*\]?\s*$",
        re.IGNORECASE,
    )
    sub_b_header_re = re.compile(
        r"^\s*(?:Q\.?\s*1\s*[:\-#]*)?\s*\(\s*([AB])\s*\)\s*(.*)$",
        re.IGNORECASE,
    )
    sub_roman_ans_re = re.compile(
        r"^\s*A[\s\-_]*\(\s*([ivx]{1,4})\s*\)\s*(.*)$",
        re.IGNORECASE,
    )

    all_lines = [
        ln
        for page in pages
        for ln in _split_inline_question_lines(page.transcript)
    ]
    has_explicit = any(
        _match_question_marker(l, explicit_only=True) is not None for l in all_lines
    )

    for page in pages:
        p_str = str(page.page_number)
        for line in _split_inline_question_lines(page.transcript):

            # Detect Part-I / Part-II / Part-III section headers
            pm = part_header_re.match(line)
            if pm:
                p_tok = pm.group(1).strip().upper()
                if p_tok in ("I", "1"):
                    current_part = 1
                elif p_tok in ("II", "2"):
                    current_part = 2
                elif p_tok in ("III", "3"):
                    current_part = 3
                continue

            # Detect (A) or (B) sub-headers under Part-I
            sbm = sub_b_header_re.match(line)
            if sbm and (current_part == 1 or current_qid in ("Q1", "Q1A", "Q1B")):
                sub_letter = sbm.group(1).upper()
                rem_sub = _clean_ocr_handwriting_text(sbm.group(2) or "")
                current_part = 1
                current_qid = f"Q1{sub_letter}"
                explicit_opened.add(current_qid)
                mapped.setdefault(current_qid, [])
                question_pages.setdefault(current_qid, [])
                page_texts_list.setdefault(current_qid, {})
                if page.page_number not in question_pages[current_qid]:
                    question_pages[current_qid].append(page.page_number)
                if rem_sub:
                    mapped[current_qid].append(rem_sub)
                    page_texts_list[current_qid].setdefault(p_str, []).append(rem_sub)
                continue

            # Detect A-(i), A-(ii), A-(iii) sub-item lines under Q1B
            srm = sub_roman_ans_re.match(line)
            if srm:
                r_tok = srm.group(1).lower()
                rem_r = _clean_ocr_handwriting_text(srm.group(2) or "")
                if current_qid in ("Q1", "Q1A"):
                    current_qid = "Q1B"
                formatted_sub = f"({r_tok}) {rem_r}".strip()
                mapped.setdefault(current_qid, [])
                question_pages.setdefault(current_qid, [])
                page_texts_list.setdefault(current_qid, {})
                if page.page_number not in question_pages[current_qid]:
                    question_pages[current_qid].append(page.page_number)
                mapped[current_qid].append(formatted_sub)
                page_texts_list[current_qid].setdefault(p_str, []).append(formatted_sub)
                continue

            marker = _match_question_marker(line, explicit_only=has_explicit)
            if marker is not None:
                marker_qid, remainder = marker
                # Track part state from explicit P2_Q* / P3_Q* / Q1A / Q1B tags
                if marker_qid.startswith("P2_"):
                    current_part = 2
                elif marker_qid.startswith("P3_"):
                    current_part = 3
                elif marker_qid in ("Q1A", "Q1B"):
                    current_part = 1
                elif marker_qid.startswith("Q") and marker_qid[1:].isdigit():
                    q_num_val = int(marker_qid[1:])
                    # If inside Q1A (Part-I MCQ block) and student wrote A-1..A-5, keep inside Q1A as (i)..(v)!
                    if (
                        current_part == 1
                        and current_qid in ("Q1A", "Q1")
                        and 1 <= q_num_val <= 6
                        and re.match(r"^\s*A[\s\-_]*0*[1-6]\b", line, flags=re.IGNORECASE)
                    ):
                        r_lbl = int_to_roman.get(q_num_val, str(q_num_val))
                        sub_line = f"({r_lbl}) {remainder}".strip()
                        if page.page_number not in question_pages[current_qid]:
                            question_pages[current_qid].append(page.page_number)
                        mapped[current_qid].append(sub_line)
                        page_texts_list[current_qid].setdefault(p_str, []).append(sub_line)
                        continue
                    elif current_part == 2:
                        marker_qid = f"P2_Q{q_num_val}"
                    elif current_part == 3:
                        marker_qid = f"P3_Q{q_num_val}"

                # If Q1 was only opened implicitly on an earlier cover page and now
                # explicitly starts on a later page, clear pre-marker cover noise.
                if (
                    has_explicit
                    and marker_qid in ("Q1", "Q1A")
                    and marker_qid not in explicit_opened
                    and question_pages.get("Q1")
                    and all(p < page.page_number for p in question_pages["Q1"])
                ):
                    mapped["Q1"] = []
                    question_pages["Q1"] = []
                    page_texts_list["Q1"] = {}

                explicit_opened.add(marker_qid)
                current_qid = marker_qid
                mapped.setdefault(current_qid, [])
                question_pages.setdefault(current_qid, [])
                page_texts_list.setdefault(current_qid, {})
                if page.page_number not in question_pages[current_qid]:
                    question_pages[current_qid].append(page.page_number)
                if remainder:
                    mapped[current_qid].append(remainder)
                    page_texts_list[current_qid].setdefault(p_str, []).append(remainder)
            else:
                cleaned_line = _clean_ocr_handwriting_text(line)
                if not cleaned_line:
                    continue
                # If the previous line was a bare sub-item label like "(v)" and current line is the answer e.g. ".."
                if mapped.get(current_qid) and re.match(r"^\s*\([ivxIVX]{1,4}\)\s*$", mapped[current_qid][-1]):
                    mapped[current_qid][-1] = f"{mapped[current_qid][-1]} {cleaned_line}".strip()
                    if page_texts_list.get(current_qid, {}).get(p_str) and re.match(r"^\s*\([ivxIVX]{1,4}\)\s*$", page_texts_list[current_qid][p_str][-1]):
                        page_texts_list[current_qid][p_str][-1] = f"{page_texts_list[current_qid][p_str][-1]} {cleaned_line}".strip()
                    continue
                mapped.setdefault(current_qid, [])
                question_pages.setdefault(current_qid, [])
                page_texts_list.setdefault(current_qid, {})
                if page.page_number not in question_pages[current_qid]:
                    question_pages[current_qid].append(page.page_number)
                mapped[current_qid].append(cleaned_line)
                page_texts_list[current_qid].setdefault(p_str, []).append(cleaned_line)

    clean_answers = {
        qid: _join_answer_parts(parts)
        for qid, parts in mapped.items()
        if _join_answer_parts(parts)
    }
    clean_page_texts = {
        qid: {
            p_str: _join_answer_parts(p_lines)
            for p_str, p_lines in p_map.items()
            if _join_answer_parts(p_lines)
        }
        for qid, p_map in page_texts_list.items()
    }

    # Auto-split combined Q1 into Q1A and Q1B if text contains A: and B: or (A) and (B)
    if "Q1" in clean_answers and ("Q1A" not in clean_answers or "Q1B" not in clean_answers):
        q1_text = clean_answers["Q1"]
        m_split = re.search(r"(?:^|\n)\s*(?:Q\.?\s*1\s*[:\-]*)?\(?B\)?\s*[:\-]\s*", q1_text, re.IGNORECASE)
        if m_split:
            split_idx = m_split.start()
            part_a = q1_text[:split_idx].strip()
            part_b = q1_text[m_split.end():].strip()
            # Clean A: marker if present
            part_a = re.sub(r"^\s*(?:Q\.?\s*1\s*[:\-]*)?\(?A\)?\s*[:\-]\s*", "", part_a, flags=re.IGNORECASE).strip()
            clean_answers["Q1A"] = part_a
            clean_answers["Q1B"] = part_b
            if "Q1" in question_pages:
                question_pages["Q1A"] = list(question_pages["Q1"])
                question_pages["Q1B"] = list(question_pages["Q1"])
            if "Q1" in clean_page_texts:
                clean_page_texts["Q1A"] = dict(clean_page_texts["Q1"])
                clean_page_texts["Q1B"] = dict(clean_page_texts["Q1"])

    return clean_answers, question_pages, clean_page_texts


def parse_questions_with_pages(
    pages: list["PageOCRResult"],
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """Parse questions across ordered pages while tracking page numbers per question."""
    clean_answers, question_pages, _ = parse_questions_with_page_texts(pages)
    return clean_answers, question_pages


# ---------------------------------------------------------------------------
# Semantic Similarity (Sentence Transformers)
# ---------------------------------------------------------------------------
_ROMAN_TO_INT: dict[str, int] = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
    "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10,
}

_META_STOPWORDS: set[str] = {
    "the", "a", "an", "of", "in", "to", "for", "and", "or", "is", "are", "was", "were",
    "by", "on", "at", "with", "from", "as", "into", "any", "valid", "name", "write",
    "state", "define", "explain", "correct", "option", "answer", "following", "blank",
    "point", "mention", "give", "example", "one", "two", "three", "four", "five",
    "core", "concept", "concepts", "accuracy", "completeness", "statement",
}


class SemanticScorer:
    """Computes hybrid dense cosine similarity (all-MiniLM-L6-v2) + exact/sub-item lexical matching."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading sentence-transformer '%s'...", self.model_name)
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _extract_positional_subitems(text: str) -> dict[int, str]:
        """Extract sub-answers indexed 1..N from markers like (i), (ii), (1), (2), (a), (b)."""
        pattern = re.compile(
            r"(?:^|\s)(?:\(\s*([ivx]{1,4}|\d{1,2}|[a-f])\s*\)|(?<!\w)([ivx]{1,4}|\d{1,2})\s*[\)\.])\s*",
            re.IGNORECASE,
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return {}

        sub_items: dict[int, str] = {}
        for i, m in enumerate(matches):
            raw_tok = (m.group(1) or m.group(2) or "").strip().lower()
            idx_num: int | None = None
            if raw_tok in _ROMAN_TO_INT:
                idx_num = _ROMAN_TO_INT[raw_tok]
            elif raw_tok.isdigit():
                idx_num = int(raw_tok)
            elif len(raw_tok) == 1 and "a" <= raw_tok <= "f":
                idx_num = ord(raw_tok) - ord("a") + 1

            start_pos = m.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            raw_seg = text[start_pos:end_pos].strip(" \t\r\n")
            if raw_seg.startswith("..") or raw_seg == "." or raw_seg.startswith("(..)"):
                seg = raw_seg.strip(" ,;:\n")
            else:
                seg = raw_seg.strip(" ,;.-:\n")
            if idx_num is not None and 1 <= idx_num <= 15 and seg:
                sub_items[idx_num] = seg
        return sub_items

    @classmethod
    def _split_chunks(cls, text: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"[.!?\n;]+", text) if p.strip()]
        chunks = [p for p in parts if p]
        sub_map = cls._extract_positional_subitems(text)
        for seg in sub_map.values():
            if seg and seg not in chunks:
                chunks.append(seg)
        if text.strip() and text.strip() not in chunks:
            chunks.append(text.strip())
        return chunks if chunks else [text.strip()]

    @staticmethod
    def _compute_lexical_kp_similarity(
        kp: KeyPoint,
        kp_idx: int,
        total_kps: int,
        student_answer: str,
        sub_items: dict[int, str],
    ) -> float:
        """Compute direct lexical / token / positional sub-item match score in [0.0, 1.0]."""
        raw_desc = kp.description.strip()
        # Strip leading sub-item prefix from description e.g. "(i) 100" -> "100"
        clean_desc = re.sub(
            r"^\s*(?:\(\s*(?:[ivx]{1,4}|\d{1,2}|[a-f])\s*\)|(?:[ivx]{1,4}|\d{1,2}|[a-f])[\.\)\-:])\s*",
            "",
            raw_desc,
            flags=re.IGNORECASE,
        ).strip()
        # Also handle "Label / Blank (i): 100" style key points by extracting value after ":"
        if ":" in clean_desc and len(clean_desc.split(":", 1)[1].strip()) >= 1:
            after_colon = clean_desc.split(":", 1)[1].strip()
            if len(after_colon) < len(clean_desc):
                clean_desc = after_colon
        if not clean_desc:
            clean_desc = raw_desc

        kp_lower = clean_desc.lower().strip(" .,;:()")
        st_lower = student_answer.lower()
        pos_sub = sub_items.get(kp_idx + 1, "").strip()
        pos_lower = pos_sub.lower().strip(" .,;:()")

        # 0. Check explicit option letter mismatch (e.g. student wrote "(b)" when "(c)" was expected)
        opt_exp = re.search(r"(?:^|\s)\(?([a-d])\)?(?:\s|$|[\.\)])", raw_desc, re.IGNORECASE)
        opt_stu = re.search(r"(?:^|\s)\(?([a-d])\)?(?:\s|$|[\.\)])", pos_sub, re.IGNORECASE)
        if opt_exp and opt_stu and len(pos_sub.split()) <= 4:
            if opt_exp.group(1).lower() != opt_stu.group(1).lower():
                return 0.0

        # 0. Symbolic answer check (e.g. ".." for parent directory, "/" for root, "~" for home)
        for sym in ("..", "~", "/"):
            if sym in clean_desc and (sym in pos_sub or sym in student_answer):
                return 0.98

        # 1. True / False positional check
        if kp_lower in ("true", "false"):
            target_scope = pos_lower if (total_kps >= 2 and pos_lower) else st_lower
            has_true = bool(re.search(r"\b(?:true|t|correct|yes)\b", target_scope))
            has_false = bool(re.search(r"\b(?:false|f|incorrect|no|not\s+true)\b", target_scope))
            if kp_lower == "true":
                if has_true and not has_false:
                    return 0.98
                if has_false and not has_true:
                    return 0.0
            else:
                if has_false and not has_true:
                    return 0.98
                if has_true and not has_false:
                    return 0.0

        # 2. Direct substring match in positional sub-item or full student answer
        if kp_lower and kp_lower not in ("true", "false"):
            if pos_lower and (kp_lower == pos_lower or kp_lower in pos_lower or pos_lower in kp_lower):
                if len(pos_lower) >= 2:
                    return 0.98
            if len(kp_lower) >= 2 and re.search(r"\b" + re.escape(kp_lower) + r"\b", st_lower):
                return 0.96

        # 3. Token & Fuzzy Token Matching (handles OCR typos like "Federa" vs "Federal")
        all_kp_tokens = re.findall(r"[a-z0-9]+", kp_lower)
        content_tokens = [t for t in all_kp_tokens if t not in _META_STOPWORDS and len(t) >= 2]
        if not content_tokens:
            content_tokens = [t for t in all_kp_tokens if t]

        # Include explicit kp.keywords if provided
        kw_tokens = [
            w.lower().strip(".,;:()")
            for w in getattr(kp, "keywords", [])
            if len(w.strip(".,;:()")) >= 2
        ]

        scope_text = f"{pos_lower} {st_lower}".strip()
        st_tokens = re.findall(r"[a-z0-9]+", scope_text)
        if not st_tokens:
            return 0.0

        def _token_matches(target_tok: str, token_pool: list[str]) -> bool:
            if target_tok in token_pool:
                return True
            if len(target_tok) >= 4:
                for sw in token_pool:
                    if len(sw) >= 3 and SequenceMatcher(None, target_tok, sw).ratio() >= 0.80:
                        return True
            return False

        if kw_tokens:
            kw_hits = sum(1 for kw in kw_tokens if kw in scope_text or _token_matches(kw, st_tokens))
            if kw_hits >= max(1, (len(kw_tokens) + 1) // 2):
                return 0.94

        if content_tokens:
            hits = sum(1 for tok in content_tokens if _token_matches(tok, st_tokens))
            ratio = hits / len(content_tokens)
            is_open_prompt = any(w in all_kp_tokens for w in ("any", "valid", "name", "example"))
            if ratio >= 0.80:
                return 0.96
            if is_open_prompt and pos_sub and (ratio >= 0.45 or len(pos_sub.split()) >= 1):
                return 0.94
            if ratio >= 0.60:
                return 0.84
            if ratio >= 0.40:
                return 0.65

        return 0.0

    def score_answer(
        self, student_answer: str, question: RubricQuestion
    ) -> tuple[float, dict[str, float]]:
        """Return (overall_model_similarity, {point_id: best_hybrid_similarity})."""
        if not student_answer.strip():
            return 0.0, {kp.point_id: 0.0 for kp in question.key_points}

        model = self._ensure_loaded()
        chunks = self._split_chunks(student_answer)
        sub_items = self._extract_positional_subitems(student_answer)

        student_full_emb = model.encode(student_answer, convert_to_tensor=True, normalize_embeddings=True)
        model_ans_emb = model.encode(question.model_answer, convert_to_tensor=True, normalize_embeddings=True)
        overall_sim = float(util.cos_sim(student_full_emb, model_ans_emb).item())

        chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
        kp_texts = [kp.description for kp in question.key_points]
        kp_embs = model.encode(kp_texts, convert_to_tensor=True, normalize_embeddings=True)

        sim_matrix = util.cos_sim(kp_embs, chunk_embs)
        kp_scores: dict[str, float] = {}
        total_kps = len(question.key_points)
        for idx, kp in enumerate(question.key_points):
            dense_sim = float(sim_matrix[idx].max().item())
            lex_sim = self._compute_lexical_kp_similarity(
                kp, idx, total_kps, student_answer, sub_items
            )
            best_sim = max(dense_sim, lex_sim)
            kp_scores[kp.point_id] = round(max(0.0, min(1.0, best_sim)), 4)

        if kp_scores:
            avg_kp_sim = sum(kp_scores.values()) / len(kp_scores)
            overall_sim = max(overall_sim, round(avg_kp_sim * 0.95, 4))
        overall_sim = round(max(0.0, min(1.0, overall_sim)), 4)

        return overall_sim, kp_scores


# ---------------------------------------------------------------------------
# Diagram VLM Evaluation (Gemini + LocalVLMFallback)
# ---------------------------------------------------------------------------
class LocalVLMFallback:
    """Offline/local fallback for diagram evaluation with Qwen2.5-VL stub."""

    def __init__(self, model_name: str = LOCAL_VLM_MODEL_NAME) -> None:
        self.model_name = model_name
        # -------------------------------------------------------------------
        # Ready-to-uncomment local Qwen2.5-VL-7B-Instruct initialization stub:
        # -------------------------------------------------------------------
        # from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        # import torch
        # self.processor = AutoProcessor.from_pretrained(self.model_name)
        # self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     self.model_name,
        #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        #     device_map="auto",
        # )
        # self.model.eval()

    def evaluate_diagram(
        self,
        diagram_path: str | Path,
        question_text: str,
        diagram_criteria: str,
    ) -> DiagramEvaluation:
        """Evaluate a cropped diagram locally (heuristic fallback when Qwen2.5-VL is disabled)."""
        # -------------------------------------------------------------------
        # Ready-to-uncomment Qwen2.5-VL-7B-Instruct inference stub:
        # -------------------------------------------------------------------
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": str(diagram_path)},
        #             {
        #                 "type": "text",
        #                 "text": (
        #                     f"Question: {question_text}\n"
        #                     f"Diagram Criteria: {diagram_criteria}\n"
        #                     "Evaluate this student diagram and return strict JSON with keys: "
        #                     "is_relevant (bool), labels_accurate (bool), "
        #                     "structure_score (float 0..1), critique (str)."
        #                 ),
        #             },
        #         ],
        #     }
        # ]
        # text_prompt = self.processor.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True
        # )
        # image_inputs = Image.open(diagram_path).convert("RGB")
        # inputs = self.processor(
        #     text=[text_prompt], images=[image_inputs], padding=True, return_tensors="pt"
        # ).to(self.model.device)
        # generated_ids = self.model.generate(**inputs, max_new_tokens=256)
        # output_text = self.processor.batch_decode(
        #     generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        # )[0]
        # return DiagramEvaluation.model_validate_json(output_text)

        img = cv2.imread(str(diagram_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return DiagramEvaluation(
                is_relevant=False,
                labels_accurate=False,
                structure_score=0.0,
                critique="Diagram image could not be loaded for local evaluation.",
            )

        h, w = img.shape[:2]
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
        )
        ink_ratio = float(np.count_nonzero(binary)) / max(h * w, 1)
        edges = cv2.Canny(img, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / max(h * w, 1)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        meaningful_components = sum(
            1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= 25
        )
        # Small/medium isolated components often indicate text labels & symbols inside the diagram
        label_like_components = sum(
            1
            for i in range(1, num_labels)
            if 25 <= stats[i, cv2.CC_STAT_AREA] <= 1200
        )

        has_good_ink = 0.015 <= ink_ratio <= 0.45
        has_structure = meaningful_components >= 5 and edge_density >= 0.015
        has_labels = label_like_components >= 4

        score = 0.25
        if has_good_ink:
            score += 0.25
        if has_structure:
            score += 0.30
        if has_labels:
            score += 0.20
        score = round(min(1.0, score), 2)

        return DiagramEvaluation(
            is_relevant=has_good_ink and has_structure,
            labels_accurate=has_labels,
            structure_score=score,
            critique=(
                f"Offline structural check ({w}x{h}px, {meaningful_components} stroke components, "
                f"{label_like_components} label-sized elements). Enable Gemini API or uncomment "
                f"{self.model_name} for semantic label verification."
            ),
        )


def _evaluate_diagram_gemini(
    diagram_path: str | Path,
    question: RubricQuestion,
    api_key: str,
) -> DiagramEvaluation:
    """Evaluate a cropped hand-drawn diagram using Gemini Flash-Lite structured output."""
    import io
    from google.genai import types

    pil_image = Image.open(diagram_path).convert("RGB")
    if pil_image.width > 960:
        ratio = 960.0 / float(pil_image.width)
        pil_image = pil_image.resize((960, int(round(pil_image.height * ratio))))
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=76, optimize=True)
    img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")
    prompt = (
        "You are an expert exam evaluator grading a student's hand-drawn diagram or labeled geography outline map.\n"
        f"Question ({question.question_id}): {question.question_text}\n"
        f"Expected Diagram / Map Criteria: {question.diagram_criteria or 'Relevant, accurately labeled scientific diagram or geography outline map.'}\n\n"
        "IMPORTANT: If the question contains internal choices ('OR' / 'अथवा', e.g., Set 1 OR Set 2 of map locations), evaluate ONLY the option the student attempted on the diagram/map. For geography maps, verify whether each marked point/location (e.g. Digboi, Goa, Durg, Lakshadweep, Mumbai High, Delhi, Raniganj, Tuticorin) is labeled and placed in approximately its correct geographic region.\n"
        "Inspect the attached image carefully and return a JSON assessment with:\n"
        "- is_relevant: whether the drawing/map actually depicts the requested concept/geography.\n"
        "- labels_accurate: whether key parts or map locations are visibly and accurately labeled.\n"
        "- structure_score: float from 0.0 to 1.0 reflecting completeness, clarity, and geographical/structural accuracy.\n"
        "- critique: concise, constructive 1-2 sentence feedback listing which labeled parts/locations are accurately shown and any missing."
    )

    response = generate_content_fast(
        api_key,
        [GEMINI_MODEL_NAME, *GEMINI_FAST_GRADING_MODELS],
        [prompt, img_part],
        response_schema=DiagramEvaluation,
        timeout_ms=14000,
    )
    if response.parsed is not None:
        return DiagramEvaluation.model_validate(response.parsed)
    return DiagramEvaluation.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Rubric Grader (Online Gemini + Offline Heuristic Fallback)
# ---------------------------------------------------------------------------
def _round_quarter(value: float) -> float:
    return round(value * 4.0) / 4.0


def clean_text_for_quote(text: str) -> str:
    """Normalizes text for quote verification: lowercase, alphanumeric and collapsed whitespace."""
    if not text:
        return ""
    t = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", t).strip().lower()


def verify_evidence_quote(quote: str, full_text: str) -> bool:
    """
    Verifies that the quoted evidence string actually exists in the student's text.
    Uses substring containment, multi-clause validation, and RapidFuzz partial ratio >= 82.
    """
    if not quote or not full_text:
        return False

    norm_quote = clean_text_for_quote(quote)
    norm_text = clean_text_for_quote(full_text)

    if not norm_quote or not norm_text:
        return False

    if norm_quote in norm_text:
        return True

    # Check multi-clause parts
    raw_clauses = [c.strip() for c in re.split(r"[;\n]+", quote) if c.strip()]
    if len(raw_clauses) > 1:
        if all(clean_text_for_quote(c) in norm_text for c in raw_clauses if clean_text_for_quote(c)):
            return True

    try:
        from rapidfuzz import fuzz
        if fuzz.partial_ratio(norm_quote, norm_text) >= 82:
            return True
    except Exception:
        pass

    return SequenceMatcher(None, norm_quote, norm_text).ratio() >= 0.80


def parse_option_parts(text: str) -> tuple[str | None, str, list[str]]:
    """
    Parses option string (key or student answer) into (letter, clean_body, variants):
      e.g. 'B) Operating system / ऑपरेटिंग सिस्टम' -> ('b', 'operating system...', ['operating system', 'ऑपरेटिंग सिस्टम'])
    """
    if not text:
        return (None, "", [])

    raw = text.strip()

    # Match standalone letter or roman numeral: 'A', '(A)', '(i)', 'ii', '1'
    m_full = re.match(r"^\s*\(?([a-zA-Z0-9]|[ivxIVX]{1,4})\)?\s*$", raw)
    if m_full:
        return (m_full.group(1).lower(), "", [])

    # Match leading option label like '(A)', 'A)', 'A.', 'A -', 'A:', '(i)', 'i)'
    m_pref = re.match(
        r"^\s*(?:\(([a-zA-Z0-9]|[ivxIVX]{1,4})\)|\b([a-zA-Z0-9]|[ivxIVX]{1,4})[.)\]:-])\s*(.*)$",
        raw,
        re.DOTALL,
    )
    letter = None
    body = raw
    if m_pref:
        letter = (m_pref.group(1) or m_pref.group(2)).lower()
        body = m_pref.group(3).strip()

    clean_body = re.sub(r"(?<=\w)\.+$", "", body.strip()).strip().lower()

    # Split variants by '/' or '|' for bilingual or alternative answers
    variants: list[str] = []
    if clean_body:
        variants.append(clean_body)
        for part in re.split(r"[/|]+", clean_body):
            p = re.sub(r"(?<=\w)\.+$", "", part.strip()).strip().lower()
            if p and p not in variants:
                variants.append(p)

    return (letter, clean_body, variants)


def check_math_equivalence(student_expr_str: str, key_expr_str: str) -> bool:
    """Checks mathematical equivalence using SymPy (algebraic and numerical)."""
    clean_s = re.sub(r"[^\w\s\+\-\*\/\^\(\)\.\=]", "", student_expr_str).strip()
    clean_k = re.sub(r"[^\w\s\+\-\*\/\^\(\)\.\=]", "", key_expr_str).strip()

    if "=" in clean_s:
        clean_s = clean_s.split("=")[-1].strip()
    if "=" in clean_k:
        clean_k = clean_k.split("=")[-1].strip()

    clean_s = clean_s.replace("^", "**")
    clean_k = clean_k.replace("^", "**")

    if not clean_s or not clean_k:
        return False

    try:
        import sympy as sp
        s_sym = sp.sympify(clean_s)
        k_sym = sp.sympify(clean_k)
        if sp.simplify(s_sym - k_sym) == 0:
            return True
        if abs(float(s_sym.evalf()) - float(k_sym.evalf())) < 1e-4:
            return True
    except Exception:
        pass
    return False


def evaluate_spelling_tolerance(
    student_word: str,
    target_word: str,
    strictness_level: int = 5,
) -> bool:
    """Evaluates word match under strict vs lenient tolerance using RapidFuzz."""
    s_norm = re.sub(r"[^\w]", "", student_word.lower())
    t_norm = re.sub(r"[^\w]", "", target_word.lower())

    if s_norm == t_norm:
        return True
    if strictness_level >= 8:
        return False  # Strict mode requires exact character match

    try:
        from rapidfuzz import fuzz
        ratio = fuzz.ratio(s_norm, t_norm)
        # Lenient/Balanced allows ratio >= 80 (e.g. 'kernal' vs 'kernel', 'operatng' vs 'operating')
        threshold = 78 if strictness_level <= 3 else 80
        return ratio >= threshold
    except Exception:
        return SequenceMatcher(None, s_norm, t_norm).ratio() >= 0.80


def _get_strictness_instructions(strictness_level: int) -> str:
    """Build dynamic grading prompt instructions calibrated across Lenient, Moderate, and Strict tiers."""
    lvl = max(1, min(10, int(strictness_level)))
    if lvl <= 3:
        return (
            f"### GRADING STRICTNESS: LENIENT (LEVEL {lvl}/10 - GENEROUS & ENCOURAGING)\n"
            "- Award partial credit generously for any demonstration of the core concept or relevant attempt.\n"
            "- Give the student the benefit of the doubt for phrasing variations, synonyms, or alternative valid approaches.\n"
            "- Do NOT heavily penalize minor calculation slips, minor spelling/OCR slips, or informal wording if understanding is clear.\n"
            "- Full marks may be given even if secondary minor details are omitted, as long as primary concepts are met."
        )
    if lvl <= 7:
        return (
            f"### GRADING STRICTNESS: MODERATE (LEVEL {lvl}/10 - BALANCED BOARD EVALUATION)\n"
            "- Apply standard university / board examination marking.\n"
            "- Award fair, proportional partial credit strictly matching the rubric and required steps.\n"
            "- Reward correct reasoning, but deduct points proportionally for missing key points or arithmetic mistakes.\n"
            "- Ensure marks accurately reflect both strengths and deficiencies in the student's answer."
        )
    return (
        f"### GRADING STRICTNESS: STRICT (LEVEL {lvl}/10 - HIGH ACADEMIC RIGOR)\n"
        "- Demand complete accuracy, correct scientific/mathematical terminology, and step-by-step reasoning.\n"
        "- Strictly deduct marks for omitted key points, improper units, vague statements, or unjustified steps.\n"
        "- Award full marks ONLY if the answer is completely accurate, thorough, and fully satisfies every point of the model answer.\n"
        "- Explicitly penalize factual errors or misconceptions (e.g., claiming an external command is internal)."
    )


def _grade_objective_answer(
    question: RubricQuestion,
    student_answer: str,
    strictness_level: int = 5,
) -> LLMGradeOutput:
    """Fast, ultra-accurate evaluation for objective questions (MCQ, Fill in blanks, One word) with bilingual & math support."""
    raw_student = student_answer.strip()
    raw_model = question.model_answer.strip()
    q_type = question.question_type
    max_marks = question.max_marks

    norm_lower = raw_student.lower()
    if not raw_student or norm_lower in (
        "not attempted", "unattempted", "nil", "n/a", "none", "not answered", "skipped", "blank", "—"
    ):
        return LLMGradeOutput(
            step_wise_breakdown=[
                StepBreakdownItem(
                    point_id="KP1",
                    description=question.model_answer,
                    max_point_marks=max_marks,
                    awarded_point_marks=0.0,
                    status="Missed",
                    similarity_score=0.0,
                    notes="No answer detected for this question.",
                )
            ],
            missing_key_points=[question.model_answer],
            marks_awarded=0.0,
            student_feedback=f"No answer provided. Expected: {question.model_answer}",
        )

    s_letter, s_body, s_variants = parse_option_parts(raw_student)
    k_letter, k_body, k_variants = parse_option_parts(raw_model)

    matched = False
    match_note = ""

    # 1. Option letter match (e.g. '(b)' vs '(b)', or 'a' vs 'A')
    if s_letter and k_letter and s_letter == k_letter:
        matched = True
        match_note = f"Correct option selected ({raw_student})."

    # 2. Text body / variant match
    if not matched and s_body:
        if s_body == k_body or s_body in k_variants:
            matched = True
            match_note = f"Correct option text identified ({raw_student})."
        elif k_body and (s_body in k_body or any(s in v for v in k_variants for s in s_variants)):
            matched = True
            match_note = f"Correct option identified ({raw_student})."

    # 3. Cross-variant match (bilingual support)
    if not matched:
        for sv in s_variants:
            if sv in k_variants or sv == k_body:
                matched = True
                match_note = f"Correct bilingual/variant match ({raw_student})."
                break

    # 4. Math equivalence via SymPy
    if not matched and (re.search(r"\d", raw_student) or re.search(r"[\+\-\*\/\^]", raw_student)):
        if check_math_equivalence(raw_student, raw_model):
            matched = True
            match_note = f"SymPy verified mathematical equivalence ({raw_student} == {raw_model})."

    # 5. Spelling tolerance via RapidFuzz
    if not matched and k_body and s_body:
        if evaluate_spelling_tolerance(s_body, k_body, strictness_level=strictness_level):
            matched = True
            match_note = f"Correct option identified within OCR spelling tolerance ({raw_student})."

    # 6. Fallback normalized character match
    if not matched:
        st_clean = clean_text_for_quote(raw_student)
        ky_clean = clean_text_for_quote(raw_model)
        if st_clean and ky_clean and (st_clean == ky_clean or st_clean in ky_clean):
            matched = True
            match_note = f"Correct answer matched ({raw_student})."

    if not matched:
        match_note = f"Wrong — wrote '{raw_student}', whereas correct option is '{raw_model}'."

    awarded = max_marks if matched else 0.0
    status: Literal["Hit", "Partial", "Missed"] = "Hit" if matched else "Missed"
    missing = [] if matched else [question.model_answer]
    feedback = (
        f"Correct! Full marks awarded ({awarded:.2f}/{max_marks:.2f})."
        if matched
        else match_note
    )

    return LLMGradeOutput(
        step_wise_breakdown=[
            StepBreakdownItem(
                point_id="KP1",
                description=question.model_answer,
                max_point_marks=max_marks,
                awarded_point_marks=awarded,
                status=status,
                similarity_score=1.0 if matched else 0.0,
                notes=match_note,
            )
        ],
        missing_key_points=missing,
        marks_awarded=awarded,
        student_feedback=feedback,
    )


def load_rubric(rubric_path: str | Path) -> ExamRubric:
    """Load and validate an exam rubric JSON file."""
    path = Path(rubric_path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return ExamRubric.model_validate(data)


class ExamGrader:
    """Grades mapped student answers & diagrams against a structured rubric."""

    def __init__(
        self,
        offline: bool = False,
        strictness_level: int = 5,
        cfg: GradingConfig = GRADING_CONFIG,
    ) -> None:
        self.cfg = cfg
        self.strictness_level = max(1, min(10, int(strictness_level)))
        self.api_key = get_gemini_api_key()
        self.offline = offline or not bool(self.api_key)
        if not offline and not self.api_key:
            logger.warning(
                "GEMINI_API_KEY is not set. Falling back to offline local grading mode."
            )
        self.semantic_scorer = SemanticScorer()
        self.local_vlm = LocalVLMFallback()

    @staticmethod
    def _allocate_keypoint_marks(question: RubricQuestion) -> dict[str, float]:
        """Convert relative key-point weights into absolute marks summing to text_max_marks."""
        total_weight = sum(kp.weight for kp in question.key_points)
        if total_weight <= 0:
            even = question.max_marks / max(len(question.key_points), 1)
            return {kp.point_id: even for kp in question.key_points}
        return {
            kp.point_id: (kp.weight / total_weight) * question.max_marks
            for kp in question.key_points
        }

    def _grade_offline(
        self,
        question: RubricQuestion,
        student_answer: str,
        overall_sim: float,
        kp_sims: dict[str, float],
        diagram_eval: DiagramEvaluation | None,
        strictness_level: int | None = None,
    ) -> LLMGradeOutput:
        """Deterministic local heuristic rubric scorer using hybrid exact/lexical + MiniLM + diagram structure."""
        lvl = max(1, min(10, int(strictness_level if strictness_level is not None else self.strictness_level)))
        if (
            question.question_type in ("mcq", "fill_blank", "one_word")
            and len(question.key_points) <= 1
            and question.max_marks <= 1.5
        ):
            return _grade_objective_answer(question, student_answer, strictness_level=lvl)

        text_budget = (
            question.max_marks * (1.0 - self.cfg.diagram_weight_when_required)
            if question.requires_diagram
            else question.max_marks
        )
        diagram_budget = question.max_marks - text_budget

        total_weight = sum(kp.weight for kp in question.key_points) or 1.0
        breakdown: list[StepBreakdownItem] = []
        missing: list[str] = []
        text_awarded = 0.0

        # Scale thresholds dynamically by strictness_level (1..10)
        hit_threshold = round(0.45 + (lvl * 0.031), 3)
        partial_threshold = round(0.25 + (lvl * 0.030), 3)
        partial_credit_factor = 0.70 if lvl <= 3 else (0.35 if lvl >= 8 else 0.50)
        word_count = len(student_answer.split())
        is_objective_group = question.question_type in ("mcq", "fill_blank", "one_word")

        for kp in question.key_points:
            kp_max = round((kp.weight / total_weight) * text_budget, 2)
            sim = kp_sims.get(kp.point_id, 0.0)
            if is_objective_group or sim >= 0.85:
                blended_sim = sim
            else:
                blended_sim = (1.0 - self.cfg.semantic_blend_weight) * sim + (
                    self.cfg.semantic_blend_weight * overall_sim
                )

            # Penalty if subjective answer is too short for a multi-mark question
            if not is_objective_group and question.max_marks >= 3.0 and word_count < 6 and lvl >= 4 and sim < 0.85:
                blended_sim *= 0.45 if lvl >= 8 else 0.6

            eff_hit_thresh = (0.72 if lvl <= 7 else 0.80) if is_objective_group else hit_threshold
            if is_objective_group:
                if blended_sim >= eff_hit_thresh:
                    status: Literal["Hit", "Partial", "Missed"] = "Hit"
                    awarded = kp_max
                    note = f"Correctly identified ({kp.description})."
                else:
                    status = "Missed"
                    awarded = 0.0
                    note = f"Wrong — expected {kp.description}."
                    missing.append(kp.description)
            elif blended_sim >= eff_hit_thresh:
                status: Literal["Hit", "Partial", "Missed"] = "Hit"
                awarded = kp_max
                note = f"Correctly matched in answer (confidence {sim:.2f}, strictness {lvl}/10)."
            elif blended_sim >= partial_threshold:
                status = "Partial"
                awarded = _round_quarter(kp_max * partial_credit_factor)
                note = f"Partially addressed (similarity {sim:.2f}, strictness {lvl}/10)."
                missing.append(f"Elaborate further on: {kp.description}")
            else:
                status = "Missed"
                awarded = 0.0
                note = f"Wrong — misses expected concept '{kp.description}' (match {sim:.2f})."
                missing.append(kp.description)

            awarded = min(kp_max, max(0.0, awarded))
            text_awarded += awarded
            breakdown.append(
                StepBreakdownItem(
                    point_id=kp.point_id,
                    description=kp.description,
                    max_point_marks=kp_max,
                    awarded_point_marks=round(awarded, 2),
                    status=status,
                    similarity_score=sim,
                    notes=note,
                )
            )

        diagram_awarded = 0.0
        if question.requires_diagram:
            if diagram_eval is not None and diagram_eval.is_relevant:
                diagram_awarded = diagram_budget * diagram_eval.structure_score
            elif diagram_eval is None:
                missing.append(f"Required diagram missing: {question.diagram_criteria}")

        total_awarded = _round_quarter(min(question.max_marks, text_awarded + diagram_awarded))

        if not student_answer.strip() and diagram_eval is None:
            feedback = "No response was found for this question."
        elif total_awarded <= 0.0 and question.max_marks > 0:
            reason = missing[0] if missing else question.model_answer
            feedback = f"Wrong — student's answer does not match key concept ({reason})."
        else:
            hit_count = sum(1 for item in breakdown if item.status == "Hit")
            feedback_parts = [
                f"Covered {hit_count}/{len(breakdown)} rubric key points "
                f"(Strictness {lvl}/10, accuracy: {overall_sim:.2f})."
            ]
            if missing:
                feedback_parts.append(f"Missing/Wrong: {missing[0]}")
            if question.requires_diagram:
                if diagram_eval is not None:
                    feedback_parts.append(f"Diagram score: {diagram_eval.structure_score:.2f}/1.00.")
                else:
                    feedback_parts.append("Include a neat, labeled diagram as requested.")
            feedback = " ".join(feedback_parts)

        return LLMGradeOutput(
            step_wise_breakdown=breakdown,
            missing_key_points=missing,
            marks_awarded=total_awarded,
            student_feedback=feedback,
        )

    @staticmethod
    def _enforce_verified_hits(
        grade_out: LLMGradeOutput,
        question: RubricQuestion,
        kp_sims: dict[str, float],
        text_budget: float,
    ) -> LLMGradeOutput:
        """Ensure any key point with verified exact/sub-item match (sim >= 0.90) receives full Hit credit."""
        total_weight = sum(kp.weight for kp in question.key_points) or 1.0
        kp_max_map = {
            kp.point_id: round((kp.weight / total_weight) * text_budget, 2)
            for kp in question.key_points
        }
        updated_any = False
        hit_floor = 0.85 if question.question_type in ("mcq", "fill_blank", "one_word") else 0.90
        for step in grade_out.step_wise_breakdown:
            sim_val = kp_sims.get(step.point_id, step.similarity_score)
            if sim_val >= hit_floor and step.status != "Hit":
                max_m = kp_max_map.get(step.point_id, step.max_point_marks)
                step.status = "Hit"
                step.awarded_point_marks = max_m
                step.similarity_score = sim_val
                step.notes = f"Verified correct match ({step.description})."
                updated_any = True
        if updated_any:
            new_sum = sum(s.awarded_point_marks for s in grade_out.step_wise_breakdown)
            grade_out.marks_awarded = _round_quarter(
                max(grade_out.marks_awarded, min(question.max_marks, new_sum))
            )
            grade_out.missing_key_points = [
                s.description for s in grade_out.step_wise_breakdown if s.status != "Hit"
            ]
        return grade_out

    def _grade_with_gemini(
        self,
        question: RubricQuestion,
        student_answer: str,
        overall_sim: float,
        kp_sims: dict[str, float],
        diagram_eval: DiagramEvaluation | None,
        strictness_level: int | None = None,
    ) -> LLMGradeOutput:
        """Use Gemini with automatic model fallback and Pydantic structured output for step-wise grading."""
        lvl = max(1, min(10, int(strictness_level if strictness_level is not None else self.strictness_level)))
        if (
            question.question_type in ("mcq", "fill_blank", "one_word")
            and len(question.key_points) <= 1
            and question.max_marks <= 1.5
        ):
            return _grade_objective_answer(question, student_answer, strictness_level=lvl)

        assert self.api_key is not None

        text_budget = (
            question.max_marks * (1.0 - self.cfg.diagram_weight_when_required)
            if question.requires_diagram
            else question.max_marks
        )
        diagram_budget = round(question.max_marks - text_budget, 2)
        total_weight = sum(kp.weight for kp in question.key_points) or 1.0

        kp_lines = []
        for kp in question.key_points:
            kp_max = round((kp.weight / total_weight) * text_budget, 2)
            kp_lines.append(
                f"- {kp.point_id} (max {kp_max} marks, match_score={kp_sims.get(kp.point_id, 0.0):.3f}): "
                f"{kp.description}"
            )

        diagram_summary = (
            diagram_eval.model_dump_json(indent=2)
            if diagram_eval is not None
            else ("MISSING (required by rubric)" if question.requires_diagram else "Not required")
        )

        strictness_block = _get_strictness_instructions(lvl)

        prompt = (
            "You are an expert academic examiner and evaluator grading a student's answer against an assessment rubric.\n"
            f"{strictness_block}\n\n"
            "CRITICAL GRADING RULES:\n"
            "1. FACTUAL ACCURACY & RIGOR:\n"
            "   - Check factual accuracy strictly: explicitly penalize wrong technical claims.\n"
            "   - MULTILINGUAL & HINDI SUPPORT: Students may write in English, Hindi (Devanagari), or mixed bilingual / Hinglish. Evaluate the conceptual correctness of their answers against the model answer/key points regardless of language.\n"
            "   - Everyday analogies earn partial credit, but full credit requires technical correctness.\n"
            "2. OBJECTIVE QUESTIONS & MULTI-PART OBJECTIVE GROUPS (MCQ, Fill in blanks, One word/sentence, True/False, Match pairs):\n"
            "   - Grade each sub-item ((i), (ii), (1), (2), etc.) directly based on factual correctness of the student's term, number, or option.\n"
            "   - If a key point asks an open-ended question (e.g., 'Name of any valid national park') or 'any N features' and the student gives factually valid answers, award FULL MARKS ('Hit')!\n"
            "   - Tolerate minor OCR transcription slips (e.g. 'Federa' for 'Federal', 'kernal' for 'kernel').\n"
            "3. INTERNAL '[OR]' / 'अथवा' / 'ANY N' CHOICES (CRITICAL):\n"
            "   - If the question has an '[OR]' / 'अथवा' choice (e.g., Option A OR Option B) or asks to 'Explain any two/three': identify ONLY the option/items the student chose to answer and grade 100% against THAT chosen option/items.\n"
            "   - Even if some Rubric Key Points refer to the unchosen alternative option, STILL mark ALL key points 'Hit' and award FULL proportional marks if the chosen option is answered accurately! NEVER deduct marks or list missing points for the unchosen alternative!\n"
            "4. DIAGRAMS / OUTLINE MAPS:\n"
            "   - If `Diagram Evaluation` shows accurate labels and `structure_score` for a diagram or India outline map question, award marks proportionally up to Full Max Marks based on the plotted locations/labels and criteria.\n"
            "5. EVIDENCE GROUNDING (NO HALLUCINATIONS):\n"
            "   - Only award credit for content actually present in the student's transcribed text or Diagram Evaluation. Never assume or invent points not written by the student.\n"
            "6. SUBJECTIVE QUESTIONS (Short Answer, Long Answer):\n"
            f"   - Grade according to the Strictness Guidelines ({lvl}/10) above.\n\n"
            f"Question Type: {question.question_type.upper()}\n"
            f"Question ({question.question_id}, Max Marks = {question.max_marks}):\n{question.question_text}\n\n"
            f"Official Model Answer / Answer Key:\n{question.model_answer}\n\n"
            f"Rubric Key Points (Text budget = {round(text_budget, 2)} marks):\n"
            + "\n".join(kp_lines)
            + f"\n\nStudent Transcribed Answer (Overall match score = {overall_sim:.3f}):\n"
            + (student_answer if student_answer.strip() else "[NO TEXT ANSWER DETECTED]")
            + f"\n\nDiagram Evaluation (Diagram budget = {diagram_budget} marks):\n{diagram_summary}\n\n"
            "Return strict JSON matching the schema:\n"
            "- step_wise_breakdown: list covering EVERY key_point id in order, status ('Hit', 'Partial', or 'Missed'), max_point_marks, awarded_point_marks, and concise notes. If a step/sub-item is WRONG (awarded 0 despite an attempt), start notes with 'Wrong — <very short 1-line reason why vs expected answer>'.\n"
            "- missing_key_points: list of missed or incomplete concepts.\n"
            f"- marks_awarded: total marks out of {question.max_marks} (sum of awarded_point_marks + diagram marks if required), rounded to nearest 0.25.\n"
            "- student_feedback: 1-2 short sentences. If the overall answer is wrong (0 marks despite an attempt), start directly with 'Wrong — ' and briefly explain WHY in 1 crisp sentence."
        )

        response = generate_content_fast(
            self.api_key,
            [GEMINI_MODEL_NAME, *GEMINI_FAST_GRADING_MODELS],
            [prompt],
            response_schema=LLMGradeOutput,
            timeout_ms=14000,
        )
        parsed = (
            LLMGradeOutput.model_validate(response.parsed)
            if response.parsed is not None
            else LLMGradeOutput.model_validate_json(response.text)
        )
        parsed = self._enforce_verified_hits(parsed, question, kp_sims, text_budget)
        parsed.marks_awarded = _round_quarter(
            max(0.0, min(question.max_marks, parsed.marks_awarded))
        )
        return parsed

    @staticmethod
    def _select_diagram_for_question(
        question: RubricQuestion,
        q_pages: list[int],
        ocr_result: "OCRPipelineResult",
        used_diagrams: set[str],
    ) -> Path | None:
        """Associate the largest unused detected diagram on the question's page(s) with the question."""
        def _diag_area(d: Any) -> int:
            try:
                return max(1, (d.bbox.x2 - d.bbox.x1) * (d.bbox.y2 - d.bbox.y1))
            except Exception:
                return 0

        for page in ocr_result.pages:
            if q_pages and page.page_number not in q_pages:
                continue
            candidates = sorted(page.diagrams, key=_diag_area, reverse=True)
            for diag in candidates:
                key = str(diag.crop_path)
                if key not in used_diagrams:
                    used_diagrams.add(key)
                    return diag.crop_path

        if question.requires_diagram:
            for page in ocr_result.pages:
                candidates = sorted(page.diagrams, key=_diag_area, reverse=True)
                for diag in candidates:
                    key = str(diag.crop_path)
                    if key not in used_diagrams:
                        used_diagrams.add(key)
                        return diag.crop_path
        return None

    def _grade_batch_with_gemini(
        self,
        batch_items: list[
            tuple[RubricQuestion, str, float, dict[str, float], DiagramEvaluation | None]
        ],
        strictness_level: int | None = None,
    ) -> dict[str, LLMGradeOutput]:
        """Grade a batch of questions in a SINGLE Gemini API call with automatic retry & model fallback."""
        assert self.api_key is not None
        lvl = max(1, min(10, int(strictness_level if strictness_level is not None else self.strictness_level)))

        class BatchQuestionGrade(BaseModel):
            question_id: str
            grade: LLMGradeOutput

        class BatchGradeResponse(BaseModel):
            results: list[BatchQuestionGrade]

        strictness_block = _get_strictness_instructions(lvl)

        blocks: list[str] = [
            "You are an expert academic examiner and evaluator grading a batch of student answers against rubric key points.\n"
            f"{strictness_block}\n\n"
            "CRITICAL RULES:\n"
            "1. FACTUAL ACCURACY & RIGOR: Enforce technical correctness. Penalize wrong claims. MULTILINGUAL SUPPORT: Evaluate conceptual correctness regardless of whether written in English, Hindi (Devanagari), or bilingual.\n"
            "2. OBJECTIVE QUESTIONS / 6-item blocks: Check each sub-item directly. Award full marks ('Hit') for factually correct answers (including open-ended examples and minor OCR slips like 'kernal' for 'kernel').\n"
            "3. INTERNAL '[OR]' / 'अथवा' / 'ANY N' CHOICES (CRITICAL): If a question has 'Option A OR Option B' or 'Explain any N', evaluate 100% against the ONE option/N items chosen by the student. Mark ALL key points 'Hit' and award FULL proportional marks if the chosen option is accurate — NEVER deduct marks or penalize for the unchosen OR option!\n"
            "4. DIAGRAMS / OUTLINE MAPS: When Diagram Evaluation confirms accurate labels and structure, award full proportional diagram/map credit.\n"
            "5. EVIDENCE GROUNDING: Only award credit for concepts demonstrably written in the student's text or diagram evaluation. Never hallucinate or assume unwritten points.\n"
            f"6. SUBJECTIVE QUESTIONS (Short/Long Answer): Evaluate according to the Strictness Guidelines ({lvl}/10).\n"
            "7. Keep `notes` very short (<= 10 words) and `student_feedback` concise (1 sentence <= 18 words). If any answer/sub-item is WRONG (0 marks despite attempt), start notes / student_feedback with 'Wrong — <brief reason vs expected answer>'.\n"
            "Return strict JSON with a result for every question_id below:\n"
        ]

        q_map: dict[str, tuple[RubricQuestion, dict[str, float], float]] = {}
        for question, student_answer, overall_sim, kp_sims, diagram_eval in batch_items:
            text_budget = (
                question.max_marks * (1.0 - self.cfg.diagram_weight_when_required)
                if question.requires_diagram
                else question.max_marks
            )
            diagram_budget = round(question.max_marks - text_budget, 2)
            q_map[question.question_id] = (question, kp_sims, text_budget)
            total_weight = sum(kp.weight for kp in question.key_points) or 1.0
            kp_str = "\n".join(
                f"  * {kp.point_id} (max {round((kp.weight / total_weight) * text_budget, 2)}M, match_score={kp_sims.get(kp.point_id, 0.0):.2f}): {kp.description}"
                for kp in question.key_points
            )
            diag_txt = ""
            if question.requires_diagram or diagram_eval is not None:
                diag_txt = (
                    f"Diagram Evaluation (budget={diagram_budget}M): {diagram_eval.model_dump_json()}\n"
                    if diagram_eval is not None
                    else f"Diagram Evaluation (budget={diagram_budget}M): MISSING\n"
                )
            blocks.append(
                f"=== {question.question_id} ({question.question_type}, Max={question.max_marks}M) ===\n"
                f"Q: {question.question_text}\n"
                f"Official Model Answer: {question.model_answer}\n"
                f"Key Points:\n{kp_str}\n"
                f"{diag_txt}"
                f"Student Answer: {student_answer if student_answer.strip() else '[NO ANSWER]'}\n"
            )

        response = generate_content_fast(
            self.api_key,
            [GEMINI_MODEL_NAME, *GEMINI_FAST_GRADING_MODELS],
            ["\n".join(blocks)],
            response_schema=BatchGradeResponse,
            timeout_ms=16000,
        )
        parsed = (
            BatchGradeResponse.model_validate(response.parsed)
            if response.parsed is not None
            else BatchGradeResponse.model_validate_json(response.text)
        )
        out: dict[str, LLMGradeOutput] = {}
        for item in parsed.results:
            if item.question_id in q_map:
                q_obj, k_sims, t_bud = q_map[item.question_id]
                item.grade = self._enforce_verified_hits(item.grade, q_obj, k_sims, t_bud)
                item.grade.marks_awarded = _round_quarter(
                    max(0.0, min(q_obj.max_marks, item.grade.marks_awarded))
                )
            out[item.question_id] = item.grade
        return out

    def grade_single_question(
        self,
        question: RubricQuestion,
        student_answer: str,
        strictness_level: int | None = None,
        page_numbers: list[int] | None = None,
    ) -> QuestionGradeResult:
        """Re-evaluate a single question after teacher edits its OCR text."""
        lvl = max(1, min(10, int(strictness_level if strictness_level is not None else self.strictness_level)))
        overall_sim, kp_sims = self.semantic_scorer.score_answer(student_answer, question)
        active_key = get_gemini_api_key()
        if active_key:
            self.api_key = active_key
        grade_output: LLMGradeOutput | None = None
        if not self.offline and self.api_key and student_answer.strip():
            try:
                grade_output = self._grade_with_gemini(
                    question, student_answer, overall_sim, kp_sims, None, strictness_level=lvl
                )
            except Exception as exc:
                logger.warning("Single question Gemini regrade fallback: %s", exc)
        if grade_output is None:
            grade_output = self._grade_offline(
                question, student_answer, overall_sim, kp_sims, None, strictness_level=lvl
            )
        return QuestionGradeResult(
            question_id=question.question_id,
            question_type=question.question_type,
            student_answer=student_answer,
            is_attempted=bool(student_answer.strip() or grade_output.marks_awarded > 0),
            page_numbers=page_numbers or [1],
            semantic_similarity=overall_sim,
            diagram_path=None,
            diagram_evaluation=None,
            step_wise_breakdown=grade_output.step_wise_breakdown,
            missing_key_points=grade_output.missing_key_points,
            marks_awarded=grade_output.marks_awarded,
            max_marks=question.max_marks,
            student_feedback=grade_output.student_feedback,
        )

    def grade_exam(
        self,
        ocr_result: "OCRPipelineResult",
        rubric: ExamRubric,
        strictness_level: int | None = None,
    ) -> ExamGradingReport:
        """Map OCR text to rubric questions, evaluate diagrams, and grade in fast batches."""
        from concurrent.futures import ThreadPoolExecutor

        lvl = max(1, min(10, int(strictness_level if strictness_level is not None else self.strictness_level)))
        self.strictness_level = lvl

        mapped_answers, question_pages, question_page_texts = parse_questions_with_page_texts(ocr_result.pages)
        used_diagrams: set[str] = set()
        is_short_submission = len(ocr_result.pages) <= 2

        # Normalize question IDs in rubric so PartII_Q1 / P2_Q1 / Q.1(A) always align with parser output
        for rq in rubric.questions:
            rq.question_id = _normalize_qid(rq.question_id)

        # Smart within-Part alignment for optional choice sections (P2_*, P3_*):
        # If a student (or OCR) numbered chosen optional answers sequentially (e.g. P3_Q1, P3_Q2)
        # when she actually answered P3_Q2 and P3_Q3, align each Part's answers to the best-matching rubric question in that Part.
        for prefix in ("P2_", "P3_"):
            part_rqs = [q for q in rubric.questions if q.question_id.startswith(prefix)]
            part_ans_qids = [
                qid for qid in list(mapped_answers.keys())
                if qid.startswith(prefix) and mapped_answers.get(qid, "").strip()
            ]
            if len(part_rqs) >= 2 and 1 <= len(part_ans_qids) <= len(part_rqs):
                rq_by_id = {q.question_id: q for q in part_rqs}
                # Check if re-mapping improves similarity significantly
                reassigned_ans: dict[str, str] = {}
                reassigned_pages: dict[str, list[int]] = {}
                reassigned_ptexts: dict[str, dict[str, str]] = {}
                used_target_qids: set[str] = set()
                changed_any = False

                for src_qid in part_ans_qids:
                    ans_txt = mapped_answers[src_qid]
                    cur_q = rq_by_id.get(src_qid)
                    cur_sim = self.semantic_scorer.score_answer(ans_txt, cur_q)[0] if cur_q else 0.0
                    best_qid = src_qid
                    best_sim = cur_sim
                    for cand_q in part_rqs:
                        if cand_q.question_id in used_target_qids:
                            continue
                        cand_sim = self.semantic_scorer.score_answer(ans_txt, cand_q)[0]
                        if cand_sim > best_sim + 0.15:
                            best_sim = cand_sim
                            best_qid = cand_q.question_id
                    if best_qid != src_qid:
                        changed_any = True
                    used_target_qids.add(best_qid)
                    reassigned_ans[best_qid] = ans_txt
                    reassigned_pages[best_qid] = question_pages.get(src_qid, [])
                    reassigned_ptexts[best_qid] = question_page_texts.get(src_qid, {})

                if changed_any and len(reassigned_ans) == len(part_ans_qids):
                    for old_k in part_ans_qids:
                        mapped_answers.pop(old_k, None)
                        question_pages.pop(old_k, None)
                        question_page_texts.pop(old_k, None)
                    mapped_answers.update(reassigned_ans)
                    question_pages.update(reassigned_pages)
                    question_page_texts.update(reassigned_ptexts)

        # Cross-part healing: if an answer is tagged as P3_Q4 / P3_Q5, generic Q1..Q10, or student-numbered Q2/Q3,
        # heal it to the correct rubric question (e.g. student wrote Q2 for Absolute Path, which corresponds to P2_Q4):
        rubric_qids = {q.question_id for q in rubric.questions}
        for k in list(mapped_answers.keys()):
            if k not in rubric_qids and mapped_answers.get(k, "").strip():
                ans_txt = mapped_answers[k]
                m_num = re.search(r"\d+", k)
                healed = False
                
                # First try exact number prefix match if available
                if m_num:
                    num_str = m_num.group(0)
                    for target_prefix in ("P2_Q", "P3_Q", "Q"):
                        cand = f"{target_prefix}{num_str}"
                        if cand in rubric_qids and not mapped_answers.get(cand, "").strip():
                            cand_q = next((q for q in rubric.questions if q.question_id == cand), None)
                            cand_sim = self.semantic_scorer.score_answer(ans_txt, cand_q)[0] if cand_q else 0.0
                            if cand_sim >= 0.25:
                                mapped_answers[cand] = mapped_answers.pop(k)
                                if k in question_pages:
                                    question_pages[cand] = question_pages.pop(k)
                                if k in question_page_texts:
                                    question_page_texts[cand] = question_page_texts.pop(k)
                                healed = True
                                break

                # If not healed by same number, check ALL unmapped rubric questions by semantic similarity
                if not healed:
                    best_cand_qid = None
                    best_cand_sim = 0.28
                    for rq in rubric.questions:
                        if rq.question_id not in mapped_answers or not mapped_answers.get(rq.question_id, "").strip():
                            sim = self.semantic_scorer.score_answer(ans_txt, rq)[0]
                            if sim > best_cand_sim:
                                best_cand_sim = sim
                                best_cand_qid = rq.question_id

                    if best_cand_qid:
                        logger.info(f"Cross-healed unmapped answer '{k}' -> '{best_cand_qid}' (semantic sim: {best_cand_sim:.3f})")
                        mapped_answers[best_cand_qid] = mapped_answers.pop(k)
                        if k in question_pages:
                            question_pages[best_cand_qid] = question_pages.pop(k)
                        if k in question_page_texts:
                            question_page_texts[best_cand_qid] = question_page_texts.pop(k)

        # In multi-part university papers with optional choice sections (e.g. P2_Q1..P2_Q6 "Attempt Any 4/5",
        # P3_Q1..P3_Q3 "Attempt Any 2"), omit unattempted optional questions when other questions in that Part are attempted.
        part2_attempted = any(
            q.question_id.startswith("P2_") and mapped_answers.get(q.question_id, "").strip()
            for q in rubric.questions
        )
        part3_attempted = any(
            q.question_id.startswith("P3_") and mapped_answers.get(q.question_id, "").strip()
            for q in rubric.questions
        )
        effective_questions: list[RubricQuestion] = []
        for rq in rubric.questions:
            if rq.question_id.startswith("P2_") and part2_attempted and not mapped_answers.get(rq.question_id, "").strip():
                continue
            if rq.question_id.startswith("P3_") and part3_attempted and not mapped_answers.get(rq.question_id, "").strip():
                continue
            effective_questions.append(rq)

        existing_qids = {q.question_id for q in effective_questions}

        # Auto-include any questions answered by the student that weren't in the initial rubric
        # so NO student answers are ever dropped or missing from review!
        def _qid_sort_key(q_str: str) -> int:
            m = re.search(r"\d+", q_str)
            return int(m.group(0)) if m else 999

        for extra_qid in sorted(mapped_answers.keys(), key=_qid_sort_key):
            norm_extra = _normalize_qid(extra_qid)
            norm_existing = {_normalize_qid(q) for q in existing_qids}
            if extra_qid in existing_qids or norm_extra in norm_existing:
                continue
            extra_ans = mapped_answers.get(extra_qid, "").strip()
            if not extra_ans:
                continue
            words = extra_ans.split()
            q_type: Literal["mcq", "fill_blank", "one_word", "short_answer", "long_answer"] = (
                "one_word" if len(words) <= 12 else ("long_answer" if len(words) >= 65 else "short_answer")
            )
            default_max = 1.0 if q_type == "one_word" else (4.0 if q_type == "long_answer" else 2.0)
            effective_questions.append(
                RubricQuestion(
                    question_id=extra_qid,
                    question_type=q_type,
                    question_text=f"Answer evaluation for {extra_qid} (Detected on Page {', '.join(str(p) for p in question_pages.get(extra_qid, [1]))})",
                    max_marks=default_max,
                    model_answer=extra_ans,
                    key_points=[
                        KeyPoint(
                            point_id=f"{extra_qid}_P1",
                            description=f"Core concepts, accuracy, and completeness for {extra_qid}",
                            weight=1.0,
                            keywords=[w.lower().strip(".,;:()") for w in words[:8] if len(w) > 3],
                        )
                    ],
                )
            )
            existing_qids.add(extra_qid)

        prepared_items: list[list[Any]] = []
        diag_futures: dict[str, tuple[Any, bool, Path, RubricQuestion]] = {}
        diag_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=4)
            if (not self.offline and self.api_key)
            else None
        )

        batch_grades: dict[str, LLMGradeOutput] = {}
        try:
            for question in effective_questions:
                qid = question.question_id
                student_answer = mapped_answers.get(qid, "")
                q_pages = list(question_pages.get(qid, []))

                diagram_path: Path | None = None
                if question.requires_diagram or (
                    is_short_submission
                    and any(p.page_number in q_pages and p.diagrams for p in ocr_result.pages)
                ):
                    diagram_path = self._select_diagram_for_question(
                        question, q_pages, ocr_result, used_diagrams
                    )
                    if diagram_path is not None:
                        for p_obj in ocr_result.pages:
                            if any(str(d.crop_path) == str(diagram_path) for d in p_obj.diagrams):
                                if p_obj.page_number not in q_pages:
                                    q_pages.append(p_obj.page_number)

                if diagram_path is not None and is_short_submission:
                    try:
                        fast_ocr = globals().get("_SHARED_RAPID_OCR")
                        if fast_ocr is None:
                            from rapidocr_onnxruntime import RapidOCR

                            fast_ocr = RapidOCR()
                            globals()["_SHARED_RAPID_OCR"] = fast_ocr
                        d_res, _ = fast_ocr(str(diagram_path))
                        if d_res:
                            diag_text = " ".join(r[1].strip() for r in d_res if r[1].strip())
                            if diag_text:
                                diag_clean = _clean_ocr_handwriting_text(diag_text)
                                if diag_clean and diag_clean not in student_answer:
                                    student_answer = f"{student_answer} {diag_clean}".strip()
                                    mapped_answers[qid] = student_answer
                    except Exception as exc:
                        logger.debug("RapidOCR diagram text extraction skipped for %s: %s", qid, exc)

                diagram_eval: DiagramEvaluation | None = None
                if diagram_path is not None:
                    if diag_pool is not None and question.requires_diagram:
                        fut = diag_pool.submit(_evaluate_diagram_gemini, diagram_path, question, self.api_key)
                        diag_futures[qid] = (fut, False, diagram_path, question)
                    else:
                        diagram_eval = self.local_vlm.evaluate_diagram(
                            diagram_path, question.question_text, question.diagram_criteria
                        )
                elif (
                    question.requires_diagram
                    and diag_pool is not None
                    and ocr_result.pages
                    and student_answer.strip()
                ):
                    target_pnum = q_pages[0] if q_pages else ocr_result.pages[0].page_number
                    target_page_obj = next(
                        (p for p in ocr_result.pages if p.page_number == target_pnum),
                        ocr_result.pages[0],
                    )
                    fut = diag_pool.submit(_evaluate_diagram_gemini, target_page_obj.source_path, question, self.api_key)
                    diag_futures[qid] = (fut, True, target_page_obj.source_path, question)

                overall_sim, kp_sims = self.semantic_scorer.score_answer(student_answer, question)
                prepared_items.append(
                    [question, student_answer, overall_sim, kp_sims, diagram_path, diagram_eval, q_pages]
                )

            # Split non-diagram items vs diagram items so text grading overlaps completely with diagram vision calls
            non_diag_items: list[tuple[RubricQuestion, str, float, dict[str, float], DiagramEvaluation | None]] = []
            diag_dep_qids: set[str] = set(diag_futures.keys())
            if not self.offline and self.api_key:
                for q, ans, osim, ksims, _, deval, _ in prepared_items:
                    if q.question_id not in diag_dep_qids and (ans.strip() or deval is not None):
                        non_diag_items.append((q, ans, osim, ksims, deval))

            grade_pool: ThreadPoolExecutor | None = (
                ThreadPoolExecutor(max_workers=4) if non_diag_items else None
            )
            non_diag_futures: list[Any] = []
            try:
                if grade_pool is not None and non_diag_items:
                    if len(non_diag_items) == 1:
                        q1, ans1, osim1, ksims1, deval1 = non_diag_items[0]
                        non_diag_futures.append(
                            ("single", q1.question_id, grade_pool.submit(self._grade_with_gemini, q1, ans1, osim1, ksims1, deval1, lvl))
                        )
                    else:
                        chunk_size = 5
                        for i in range(0, len(non_diag_items), chunk_size):
                            chunk = non_diag_items[i : i + chunk_size]
                            non_diag_futures.append(
                                ("batch", "", grade_pool.submit(self._grade_batch_with_gemini, chunk, lvl))
                            )

                if diag_futures:
                    for entry in prepared_items:
                        q = entry[0]
                        if q.question_id in diag_futures:
                            fut, is_fullpage_fallback, dpath, qobj = diag_futures[q.question_id]
                            try:
                                deval = fut.result()
                                if not is_fullpage_fallback or (deval.is_relevant and deval.structure_score >= 0.25):
                                    entry[5] = deval
                            except Exception as exc:
                                if not is_fullpage_fallback:
                                    entry[5] = self.local_vlm.evaluate_diagram(
                                        dpath, qobj.question_text, qobj.diagram_criteria
                                    )
                                else:
                                    logger.debug("Full-page Gemini diagram check skipped for %s: %s", q.question_id, exc)

                # Grade remaining diagram-dependent questions concurrently while collecting non-diagram futures
                if not self.offline and self.api_key and diag_dep_qids:
                    diag_grade_items = [
                        (q, ans, osim, ksims, deval)
                        for q, ans, osim, ksims, _, deval, _ in prepared_items
                        if q.question_id in diag_dep_qids and (ans.strip() or deval is not None)
                    ]
                    if len(diag_grade_items) == 1:
                        qd, ansd, osimd, ksimsd, devald = diag_grade_items[0]
                        try:
                            batch_grades[qd.question_id] = self._grade_with_gemini(
                                qd, ansd, osimd, ksimsd, devald, strictness_level=lvl
                            )
                        except Exception as exc:
                            logger.warning("Gemini single diagram grading fallback for %s: %s", qd.question_id, exc)
                    elif diag_grade_items:
                        try:
                            batch_grades.update(self._grade_batch_with_gemini(diag_grade_items, lvl))
                        except Exception as exc:
                            logger.warning("Gemini diagram batch grading fallback: %s", exc)

                for kind, qid_tag, gf in non_diag_futures:
                    try:
                        res_obj = gf.result()
                        if kind == "single":
                            batch_grades[qid_tag] = res_obj
                        else:
                            batch_grades.update(res_obj)
                    except Exception as exc:
                        logger.warning("Gemini parallel grading fallback triggered: %s", exc)
            finally:
                if grade_pool is not None:
                    grade_pool.shutdown(wait=False)
        finally:
            if diag_pool is not None:
                diag_pool.shutdown(wait=False)

        results: list[QuestionGradeResult] = []
        for question, student_answer, overall_sim, kp_sims, diagram_path, diagram_eval, q_pages in prepared_items:
            qid = question.question_id
            grade_output = batch_grades.get(qid)
            if grade_output is None:
                grade_output = self._grade_offline(
                    question, student_answer, overall_sim, kp_sims, diagram_eval, strictness_level=lvl
                )
            results.append(
                QuestionGradeResult(
                    question_id=qid,
                    question_type=question.question_type,
                    student_answer=student_answer,
                    is_attempted=bool(student_answer.strip() or diagram_path or grade_output.marks_awarded > 0),
                    page_numbers=q_pages if q_pages else [1],
                    page_texts=question_page_texts.get(qid, {}),
                    semantic_similarity=overall_sim,
                    diagram_path=str(diagram_path) if diagram_path else None,
                    diagram_evaluation=diagram_eval,
                    step_wise_breakdown=grade_output.step_wise_breakdown,
                    missing_key_points=grade_output.missing_key_points,
                    marks_awarded=grade_output.marks_awarded,
                    max_marks=question.max_marks,
                    student_feedback=grade_output.student_feedback,
                )
            )

        total_awarded = round(sum(r.marks_awarded for r in results), 2)
        printed_mm = float(getattr(rubric, "printed_max_marks", 0.0) or 0.0)
        total_max = round(printed_mm if printed_mm > 0 else sum(r.max_marks for r in results), 2)

        return ExamGradingReport(
            exam_title=rubric.exam_title,
            mode="offline_local" if self.offline else "online_gemini",
            strictness_level=lvl,
            total_marks_awarded=total_awarded,
            total_max_marks=total_max,
            printed_max_marks=printed_mm,
            mapped_answers=mapped_answers,
            question_results=results,
        )
