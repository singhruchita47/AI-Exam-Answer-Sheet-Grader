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
    GEMINI_MODEL_NAME,
    GRADING_CONFIG,
    LOCAL_VLM_MODEL_NAME,
    GradingConfig,
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
    mapped_answers: dict[str, str]
    question_results: list[QuestionGradeResult]


# ---------------------------------------------------------------------------
# Regex Question Boundary Segmentation
# ---------------------------------------------------------------------------
_EXPLICIT_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Matches: Q1, Q.1, Q. No. 1, Question no. - 01, Q1), Question 1:
    re.compile(
        r"^\s*Q(?:uestion)?\s*(?:no\.?|number)?\s*[\.\-:#]?\s*0*([0-9]{1,2}|[!lI|])\s*(?:\[?\s*or\s*\]?|\(\s*or\s*\))?\s*[\)\.\-:#]*(?:\s+(.*))?$",
        re.IGNORECASE,
    ),
    # Matches: Answer no. - 01, Answer no.-06 [or], Ans. No. 2, Ans 1, ns 1, ins 2
    re.compile(
        r"^\s*(?:ans(?:wer)?|an[a-z]{2,5}|[a-z]{0,2}[ns]|[a-z]?s|[Aa])\s*(?:no\.?|m?no\.?|number)?\s*[\.\-:#\s]+\s*0*([0-9]{1,2}|[!lI|])\s*(?:\[?\s*or\s*\]?|\(\s*or\s*\))?\s*(?:[\.\-:#]\s*(?:[0-9]{1,2}\b)?)?\s*[\)\.\-:#]*(?:\s*(.*))?$",
        re.IGNORECASE,
    ),
    # Also matches Ans1, Ans2 (no space) with digits
    re.compile(
        r"^\s*(?:ans(?:wer)?|[a-z]{0,2}[ns]|[a-z]?s|[Aa])0*([0-9]{1,2})\s*(?:\[?\s*or\s*\]?)?\s*(?:[\.\-:#]\s*(?:[0-9]{1,2}\b)?)?\s*[\)\.\-:#]*(?:\s*(.*))?$",
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
)


def _clean_ocr_handwriting_text(text: str) -> str:
    """Fix common OCR misrecognitions, strip Devanagari/Hindi characters, and filter TrOCR hallucinations."""
    # Strip any Hindi / Devanagari characters
    text = re.sub(r"[\u0900-\u097F]+", "", text)
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
                remainder = (match.group(2) or "").strip()
                remainder = re.sub(
                    r"^(?:\[?\s*or\s*\]?|\(\s*or\s*\)|[\.\(\)\-–!|:\s])+(?:\d{1,2}\s*[\.\)]\s*)?",
                    "",
                    remainder,
                    flags=re.IGNORECASE,
                ).strip()
                remainder = _clean_ocr_handwriting_text(remainder)
                return f"Q{q_num}", remainder
    return None


def parse_questions(full_text: str) -> dict[str, str]:
    """Split continuous reconstructed OCR text into a {question_id: answer_text} map."""
    mapped: dict[str, list[str]] = {}
    current_qid: str = "Q1"  # Default start to Q1 so top text is never discarded
    lines = [raw.strip() for raw in full_text.splitlines() if raw.strip()]
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


def parse_questions_with_page_texts(
    pages: list["PageOCRResult"],
) -> tuple[dict[str, str], dict[str, list[int]], dict[str, dict[str, str]]]:
    """Parse questions across ordered pages while tracking page numbers and per-page text per question."""
    mapped: dict[str, list[str]] = {}
    question_pages: dict[str, list[int]] = {}
    page_texts_list: dict[str, dict[str, list[str]]] = {}
    current_qid: str = "Q1"  # Default start to Q1 so opening text is NEVER lost
    explicit_opened: set[str] = set()

    all_lines = [
        raw.strip()
        for page in pages
        for raw in page.transcript.splitlines()
        if raw.strip()
    ]
    has_explicit = any(
        _match_question_marker(l, explicit_only=True) is not None for l in all_lines
    )

    for page in pages:
        p_str = str(page.page_number)
        for raw_line in page.transcript.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            marker = _match_question_marker(line, explicit_only=has_explicit)
            if marker is not None:
                marker_qid, remainder = marker
                # If Q1 was only opened implicitly on an earlier cover page and now
                # explicitly starts on a later page, clear pre-marker cover noise.
                if (
                    has_explicit
                    and marker_qid == "Q1"
                    and "Q1" not in explicit_opened
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
                mapped.setdefault(current_qid, [])
                question_pages.setdefault(current_qid, [])
                page_texts_list.setdefault(current_qid, {})
                if page.page_number not in question_pages[current_qid]:
                    question_pages[current_qid].append(page.page_number)
                mapped[current_qid].append(cleaned_line)
                page_texts_list[current_qid].setdefault(p_str, []).append(cleaned_line)

    clean_answers = {
        qid: " ".join(parts).strip()
        for qid, parts in mapped.items()
        if " ".join(parts).strip()
    }
    clean_page_texts = {
        qid: {
            p_str: " ".join(p_lines).strip()
            for p_str, p_lines in p_map.items()
            if " ".join(p_lines).strip()
        }
        for qid, p_map in page_texts_list.items()
    }
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
            seg = text[start_pos:end_pos].strip(" ,;.-:\n")
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
        kp: RubricKeyPoint,
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
        if not clean_desc:
            clean_desc = raw_desc

        kp_lower = clean_desc.lower().strip(" .,;:()")
        st_lower = student_answer.lower()
        pos_sub = sub_items.get(kp_idx + 1, "").strip()
        pos_lower = pos_sub.lower().strip(" .,;:()")

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
    """Evaluate a cropped hand-drawn diagram using Gemini 2.5 Flash structured output."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    pil_image = Image.open(diagram_path).convert("RGB")
    prompt = (
        "You are an expert STEM exam evaluator grading a student's hand-drawn diagram.\n"
        f"Question ({question.question_id}): {question.question_text}\n"
        f"Expected Diagram Criteria: {question.diagram_criteria or 'Relevant, accurately labeled scientific diagram.'}\n\n"
        "Inspect the attached cropped diagram image carefully and return a JSON assessment with:\n"
        "- is_relevant: whether the drawing actually depicts the requested concept/system.\n"
        "- labels_accurate: whether key parts are visibly and accurately labeled.\n"
        "- structure_score: float from 0.0 to 1.0 reflecting completeness, clarity, and accuracy.\n"
        "- critique: concise, constructive 1-2 sentence feedback on what is drawn well and what is missing."
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[prompt, pil_image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DiagramEvaluation,
            temperature=0.1,
        ),
    )
    if response.parsed is not None:
        return DiagramEvaluation.model_validate(response.parsed)
    return DiagramEvaluation.model_validate_json(response.text)


# ---------------------------------------------------------------------------
# Rubric Grader (Online Gemini + Offline Heuristic Fallback)
# ---------------------------------------------------------------------------
def _round_quarter(value: float) -> float:
    return round(value * 4.0) / 4.0


def _get_strictness_instructions(strictness_level: int) -> str:
    """Build dynamic grading prompt instructions for Strictness Level 1..10."""
    lvl = max(1, min(10, int(strictness_level)))
    if lvl <= 3:
        return (
            f"GLOBAL STRICTNESS LEVEL: {lvl}/10 (LENIENT MODE).\n"
            "- Focus on basic concept understanding.\n"
            "- Ignore minor spelling, grammar, or OCR errors and missing sub-steps.\n"
            "- Award generous partial marks whenever the student conveys the core idea."
        )
    if lvl <= 7:
        return (
            f"GLOBAL STRICTNESS LEVEL: {lvl}/10 (BALANCED / STANDARD BOARD EVALUATION).\n"
            "- Apply standard board/university evaluation.\n"
            "- Check key terms, steps, and diagram labels fairly.\n"
            "- Award proportional partial marks for partially complete answers."
        )
    return (
        f"GLOBAL STRICTNESS LEVEL: {lvl}/10 (STRICT / RIGOROUS MODE).\n"
        "- Enforce strict keyword matching and exact formula/step verification.\n"
        "- Apply strict penalties for missing labels, vague explanations, or incomplete sub-steps.\n"
        "- Only award full marks when every technical term and step is accurately stated."
    )


def _grade_objective_answer(
    question: RubricQuestion,
    student_answer: str,
    strictness_level: int = 5,
) -> LLMGradeOutput:
    """Fast, accurate direct evaluation for objective questions (MCQ, Fill in blanks, One word)."""
    st_clean = student_answer.strip().lower()
    mod_clean = question.model_answer.strip().lower()
    q_type = question.question_type
    max_marks = question.max_marks

    if not st_clean:
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

    matched = False
    match_note = ""
    typo_threshold = 0.68 if strictness_level <= 3 else (0.86 if strictness_level >= 8 else 0.78)

    if q_type == "mcq":
        # Extract option letter e.g. (b) or b from model answer
        opt_match = re.search(r"\(?([a-d])\)?", mod_clean)
        expected_opt = opt_match.group(1) if opt_match else None
        expected_text = re.sub(r"^\(?[a-d]\)?[\.\s:\-]*", "", mod_clean).strip()

        # Check student option letter e.g. (b), b, b), [b]
        st_opt_match = re.search(r"\(?([a-d])\)?", st_clean)
        st_opt = st_opt_match.group(1) if st_opt_match else None

        opt_matched = bool(expected_opt and st_opt and expected_opt == st_opt)
        text_matched = False
        if expected_text and len(expected_text) > 2:
            if expected_text in st_clean:
                text_matched = True
            else:
                sim = SequenceMatcher(None, expected_text, st_clean).ratio()
                if sim >= typo_threshold:
                    text_matched = True

        if opt_matched or text_matched:
            matched = True
            match_note = f"Correct option selected ({student_answer.strip()})."
        else:
            match_note = f"Incorrect option. Expected: {question.model_answer.strip()}."

    else:
        # fill_blank or one_word
        target_words = re.findall(r"\w+", mod_clean)
        st_words = re.findall(r"\w+", st_clean)

        if mod_clean in st_clean:
            matched = True
        elif any(w in st_words for w in target_words if len(w) > 3):
            matched = True
        else:
            for tw in target_words:
                if len(tw) > 3:
                    for sw in st_words:
                        if SequenceMatcher(None, tw, sw).ratio() >= typo_threshold:
                            matched = True
                            break
        if matched:
            match_note = f"Correct target word/term identified ({student_answer.strip()})."
        else:
            match_note = f"Incorrect term. Expected: {question.model_answer.strip()}."

    awarded = max_marks if matched else 0.0
    status: Literal["Hit", "Partial", "Missed"] = "Hit" if matched else "Missed"
    missing = [] if matched else [question.model_answer]
    feedback = (
        f"Correct! Full marks awarded ({awarded:.2f}/{max_marks:.2f})."
        if matched
        else f"Incorrect. Correct answer is: {question.model_answer}"
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
            if sim >= 0.85:
                blended_sim = sim
            else:
                blended_sim = (1.0 - self.cfg.semantic_blend_weight) * sim + (
                    self.cfg.semantic_blend_weight * overall_sim
                )

            # Penalty if subjective answer is too short for a multi-mark question
            if not is_objective_group and question.max_marks >= 3.0 and word_count < 6 and lvl >= 4 and sim < 0.85:
                blended_sim *= 0.45 if lvl >= 8 else 0.6

            if blended_sim >= hit_threshold:
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
                note = f"Not clearly detected in answer (similarity {sim:.2f}, strictness {lvl}/10)."
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
        else:
            hit_count = sum(1 for item in breakdown if item.status == "Hit")
            feedback_parts = [
                f"Covered {hit_count}/{len(breakdown)} rubric key points "
                f"(Strictness {lvl}/10, accuracy: {overall_sim:.2f})."
            ]
            if missing:
                feedback_parts.append(f"Focus on: {missing[0]}")
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
        for step in grade_out.step_wise_breakdown:
            sim_val = kp_sims.get(step.point_id, step.similarity_score)
            if sim_val >= 0.90 and step.status != "Hit":
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
        import time
        from google import genai
        from google.genai import types

        lvl = max(1, min(10, int(strictness_level if strictness_level is not None else self.strictness_level)))
        if (
            question.question_type in ("mcq", "fill_blank", "one_word")
            and len(question.key_points) <= 1
            and question.max_marks <= 1.5
        ):
            return _grade_objective_answer(question, student_answer, strictness_level=lvl)

        assert self.api_key is not None
        client = genai.Client(api_key=self.api_key)

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
            "You are an expert board exam evaluator grading a student's OCR-transcribed handwritten answer.\n"
            f"{strictness_block}\n\n"
            "CRITICAL GRADING RULES:\n"
            "1. OBJECTIVE QUESTIONS & MULTI-PART OBJECTIVE GROUPS (MCQ, Fill in blanks, One word/sentence, True/False, Match pairs):\n"
            "   - Grade each sub-item ((i), (ii), (1), (2), etc.) directly based on factual correctness of the student's term, number, or option.\n"
            "   - If a key point asks an open-ended question (e.g., 'Name of any valid national park in MP') and the student gives a factually valid answer (e.g., 'Kanha National Park Kisli'), award FULL MARKS ('Hit')!\n"
            "   - Ignore minor OCR spelling variations (e.g. 'Federa' for 'Federal').\n"
            "2. INTERNAL '[OR]' CHOICES:\n"
            "   - If the question has an '[OR]' option, the student only needs to answer ONE of the options. Award full credit if their chosen option is answered accurately.\n"
            "3. SUBJECTIVE QUESTIONS (Short Answer, Long Answer):\n"
            f"   - Grade according to the Global Strictness Level ({lvl}/10) above.\n\n"
            f"Question Type: {question.question_type.upper()}\n"
            f"Question ({question.question_id}, Max Marks = {question.max_marks}):\n{question.question_text}\n\n"
            f"Official Model Answer / Answer Key:\n{question.model_answer}\n\n"
            f"Rubric Key Points (Text budget = {round(text_budget, 2)} marks):\n"
            + "\n".join(kp_lines)
            + f"\n\nStudent Transcribed Answer (Overall match score = {overall_sim:.3f}):\n"
            + (student_answer if student_answer.strip() else "[NO TEXT ANSWER DETECTED]")
            + f"\n\nDiagram Evaluation (Diagram budget = {diagram_budget} marks):\n{diagram_summary}\n\n"
            "Return strict JSON matching the schema:\n"
            "- step_wise_breakdown: list covering EVERY key_point id in order, status ('Hit', 'Partial', or 'Missed'), max_point_marks, awarded_point_marks, and concise notes.\n"
            "- missing_key_points: list of missed or incomplete concepts.\n"
            f"- marks_awarded: total marks out of {question.max_marks} (sum of awarded_point_marks + diagram marks if required), rounded to nearest 0.25.\n"
            "- student_feedback: 1-2 short, actionable, constructive sentences."
        )

        last_err: Exception | None = None
        for m_name in dict.fromkeys([GEMINI_MODEL_NAME, "gemini-2.5-flash", "gemini-2.0-flash"]):
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LLMGradeOutput,
                        temperature=0.1,
                    ),
                )
                if response.parsed is not None:
                    parsed = LLMGradeOutput.model_validate(response.parsed)
                else:
                    parsed = LLMGradeOutput.model_validate_json(response.text)
                parsed = self._enforce_verified_hits(parsed, question, kp_sims, text_budget)
                parsed.marks_awarded = _round_quarter(
                    max(0.0, min(question.max_marks, parsed.marks_awarded))
                )
                return parsed
            except Exception as exc:
                last_err = exc
                time.sleep(0.8)
        raise RuntimeError(f"Gemini grading failed across fallback models: {last_err}")

    @staticmethod
    def _select_diagram_for_question(
        question: RubricQuestion,
        q_pages: list[int],
        ocr_result: "OCRPipelineResult",
        used_diagrams: set[str],
    ) -> Path | None:
        """Associate an unused detected diagram on the question's page(s) with the question."""
        for page in ocr_result.pages:
            if q_pages and page.page_number not in q_pages:
                continue
            for diag in page.diagrams:
                key = str(diag.crop_path)
                if key not in used_diagrams:
                    used_diagrams.add(key)
                    return diag.crop_path

        if question.requires_diagram:
            for page in ocr_result.pages:
                for diag in page.diagrams:
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
        import time
        from google import genai
        from google.genai import types

        assert self.api_key is not None
        client = genai.Client(api_key=self.api_key)
        lvl = max(1, min(10, int(strictness_level if strictness_level is not None else self.strictness_level)))

        class BatchQuestionGrade(BaseModel):
            question_id: str
            grade: LLMGradeOutput

        class BatchGradeResponse(BaseModel):
            results: list[BatchQuestionGrade]

        strictness_block = _get_strictness_instructions(lvl)

        blocks: list[str] = [
            "You are an expert board exam evaluator grading a batch of student answers.\n"
            f"{strictness_block}\n\n"
            "CRITICAL RULES:\n"
            "1. For OBJECTIVE questions / 6-item objective blocks ((i)..(vi), (1)..(6)): check each sub-item directly against the student's sub-answer. Award full marks ('Hit') for each factually correct sub-item (including valid examples like 'Kanha National Park Kisli' for 'Name of any valid national park in MP', and minor OCR typos like 'Federa' for 'Federal').\n"
            "2. For '[OR]' internal choices: award full credit if the student accurately answered EITHER option.\n"
            f"3. For SUBJECTIVE questions (Short/Long Answer): evaluate according to Global Strictness Level ({lvl}/10).\n"
            "4. Keep notes and student_feedback concise (1 sentence).\n"
            "Return strict JSON with a result for every question_id below:\n"
        ]

        q_map: dict[str, tuple[RubricQuestion, dict[str, float], float]] = {}
        for question, student_answer, overall_sim, kp_sims, diagram_eval in batch_items:
            text_budget = (
                question.max_marks * (1.0 - self.cfg.diagram_weight_when_required)
                if question.requires_diagram
                else question.max_marks
            )
            q_map[question.question_id] = (question, kp_sims, text_budget)
            total_weight = sum(kp.weight for kp in question.key_points) or 1.0
            kp_str = "\n".join(
                f"  * {kp.point_id} (max {round((kp.weight / total_weight) * text_budget, 2)}M, match_score={kp_sims.get(kp.point_id, 0.0):.2f}): {kp.description}"
                for kp in question.key_points
            )
            blocks.append(
                f"=== {question.question_id} ({question.question_type}, Max={question.max_marks}M) ===\n"
                f"Q: {question.question_text}\n"
                f"Official Model Answer: {question.model_answer}\n"
                f"Key Points:\n{kp_str}\n"
                f"Student Answer: {student_answer if student_answer.strip() else '[NO ANSWER]'}\n"
            )

        last_err: Exception | None = None
        for m_name in dict.fromkeys([GEMINI_MODEL_NAME, "gemini-2.5-flash", "gemini-2.0-flash"]):
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=["\n".join(blocks)],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BatchGradeResponse,
                        temperature=0.1,
                    ),
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
            except Exception as exc:
                last_err = exc
                time.sleep(0.9)
        raise RuntimeError(f"Batch Gemini grading failed across fallback models: {last_err}")

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

        effective_questions: list[RubricQuestion] = list(rubric.questions)
        existing_qids = {q.question_id for q in effective_questions}

        # Ensure any additional questions detected on multi-page student sheets are also graded & shown on their page
        def _qid_sort_key(q_str: str) -> int:
            m = re.search(r"\d+", q_str)
            return int(m.group(0)) if m else 999

        for extra_qid in sorted(mapped_answers.keys(), key=_qid_sort_key):
            if extra_qid in existing_qids:
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
                        RubricKeyPoint(
                            point_id=f"{extra_qid}_P1",
                            description=f"Core concepts, accuracy, and completeness for {extra_qid}",
                            weight=1.0,
                            keywords=[w.lower().strip(".,;:()") for w in words[:8] if len(w) > 3],
                        )
                    ],
                )
            )
            existing_qids.add(extra_qid)

        prepared_items: list[
            tuple[RubricQuestion, str, float, dict[str, float], Path | None, DiagramEvaluation | None, list[int]]
        ] = []
        fast_ocr = None

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
                    if fast_ocr is None:
                        from rapidocr_onnxruntime import RapidOCR

                        fast_ocr = RapidOCR()
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
                if not self.offline and self.api_key and question.requires_diagram and is_short_submission:
                    try:
                        diagram_eval = _evaluate_diagram_gemini(
                            diagram_path, question, self.api_key
                        )
                    except Exception:
                        diagram_eval = self.local_vlm.evaluate_diagram(
                            diagram_path, question.question_text, question.diagram_criteria
                        )
                else:
                    diagram_eval = self.local_vlm.evaluate_diagram(
                        diagram_path, question.question_text, question.diagram_criteria
                    )

            overall_sim, kp_sims = self.semantic_scorer.score_answer(student_answer, question)
            prepared_items.append(
                (question, student_answer, overall_sim, kp_sims, diagram_path, diagram_eval, q_pages)
            )

        batch_grades: dict[str, LLMGradeOutput] = {}
        if not self.offline and self.api_key:
            to_grade_online = [
                (q, ans, osim, ksims, deval)
                for q, ans, osim, ksims, _, deval, _ in prepared_items
                if ans.strip() or deval is not None
            ]
            if len(to_grade_online) <= 3:
                for q, ans, osim, ksims, deval in to_grade_online:
                    try:
                        batch_grades[q.question_id] = self._grade_with_gemini(
                            q, ans, osim, ksims, deval, strictness_level=lvl
                        )
                    except Exception as exc:
                        logger.warning("Gemini single grading fallback for %s: %s", q.question_id, exc)
            else:
                chunk_size = 12
                chunks = [
                    to_grade_online[i : i + chunk_size]
                    for i in range(0, len(to_grade_online), chunk_size)
                ]
                with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as pool:
                    futures = [
                        pool.submit(self._grade_batch_with_gemini, chunk, lvl)
                        for chunk in chunks
                    ]
                    for fut in futures:
                        try:
                            batch_grades.update(fut.result())
                        except Exception as exc:
                            logger.warning("Gemini batch grading fallback triggered: %s", exc)

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
        total_max = round(sum(r.max_marks for r in results), 2)

        return ExamGradingReport(
            exam_title=rubric.exam_title,
            mode="offline_local" if self.offline else "online_gemini",
            strictness_level=lvl,
            total_marks_awarded=total_awarded,
            total_max_marks=total_max,
            mapped_answers=mapped_answers,
            question_results=results,
        )
