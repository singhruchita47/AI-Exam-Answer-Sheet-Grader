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
    total_marks_awarded: float
    total_max_marks: float
    mapped_answers: dict[str, str]
    question_results: list[QuestionGradeResult]


# ---------------------------------------------------------------------------
# Regex Question Boundary Segmentation
# ---------------------------------------------------------------------------
_QUESTION_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Matches: Q1, Q.1, Q 1:, Q1), Q. 2 -, Question 1:, Q!, Question 1.
    re.compile(
        r"^\s*Q(?:uestion)?\s*[\.\-:#]?\s*([0-9]{1,2}|[!lI|])\s*[\)\.\-:#]*(?:\s+(.*))?$",
        re.IGNORECASE,
    ),
    # Matches: Ans 1, Ans. 1, Answer 2, Ans1., ns 1, ins 2, as 1, ms 1, s 1, A1, A.1
    re.compile(
        r"^\s*(?:ans(?:wer)?|[a-z]{0,2}[ns]|[a-z]?s|[Aa])\s*[\.\-:#\s]\s*([0-9]{1,2}|[!lI|])\s*(?:[\.\-:#]\s*(?:[0-9]{1,2}\b)?)?\s*[\)\.\-:#]*(?:\s*(.*))?$",
        re.IGNORECASE,
    ),
    # Also matches Ans1, Ans2 (no space) with digits
    re.compile(
        r"^\s*(?:ans(?:wer)?|[a-z]{0,2}[ns]|[a-z]?s|[Aa])([0-9]{1,2})\s*(?:[\.\-:#]\s*(?:[0-9]{1,2}\b)?)?\s*[\)\.\-:#]*(?:\s*(.*))?$",
        re.IGNORECASE,
    ),
    # Matches: (1), (2), (10)
    re.compile(r"^\s*\(([0-9]{1,2}|[!lI|])\)\s*(.*)$"),
    # Matches: 1), 2:, 1 ) or 1 - Text
    re.compile(r"^\s*([0-9]{1,2})\s*[\)\:\-–]\s*(.*)$"),
    # Matches: 1. Text... (requires dot + space to avoid decimals like 1.5)
    re.compile(r"^\s*([0-9]{1,2})\.\s+(.*)$"),
)


def _clean_ocr_handwriting_text(text: str) -> str:
    """Fix common OCR misrecognitions and strip Devanagari/Hindi characters."""
    # Strip any Hindi / Devanagari characters
    text = re.sub(r"[\u0900-\u097F]+", "", text)
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


def _match_question_marker(line: str) -> tuple[str, str] | None:
    """Return normalized (question_id, remainder_text) if line starts with a marker."""
    stripped = line.strip()
    if not stripped:
        return None
    for pattern in _QUESTION_MARKER_PATTERNS:
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
                remainder = re.sub(r"^[\.\(\)\-–!|:\s]+(?:\d{1,2}\s*[\.\)]\s*)?", "", remainder).strip()
                remainder = _clean_ocr_handwriting_text(remainder)
                return f"Q{q_num}", remainder
    return None


def parse_questions(full_text: str) -> dict[str, str]:
    """Split continuous reconstructed OCR text into a {question_id: answer_text} map."""
    mapped: dict[str, list[str]] = {}
    current_qid: str = "Q1"  # Default start to Q1 so top text is never discarded

    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        marker = _match_question_marker(line)
        if marker is not None:
            current_qid, remainder = marker
            mapped.setdefault(current_qid, [])
            if remainder:
                mapped[current_qid].append(remainder)
        else:
            cleaned_line = _clean_ocr_handwriting_text(line)
            mapped.setdefault(current_qid, []).append(cleaned_line)

    if not mapped and full_text.strip():
        return {"Q1": _clean_ocr_handwriting_text(" ".join([l.strip() for l in full_text.splitlines() if l.strip()]))}

    return {qid: " ".join(parts).strip() for qid, parts in mapped.items()}


def parse_questions_with_pages(
    pages: list["PageOCRResult"],
) -> tuple[dict[str, str], dict[str, list[int]]]:
    """Parse questions across ordered pages while tracking page numbers per question."""
    mapped: dict[str, list[str]] = {}
    question_pages: dict[str, list[int]] = {}
    current_qid: str = "Q1"  # Default start to Q1 so opening text is NEVER lost

    for page in pages:
        for raw_line in page.transcript.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            marker = _match_question_marker(line)
            if marker is not None:
                current_qid, remainder = marker
                mapped.setdefault(current_qid, [])
                question_pages.setdefault(current_qid, [])
                if page.page_number not in question_pages[current_qid]:
                    question_pages[current_qid].append(page.page_number)
                if remainder:
                    mapped[current_qid].append(remainder)
            else:
                cleaned_line = _clean_ocr_handwriting_text(line)
                mapped.setdefault(current_qid, [])
                question_pages.setdefault(current_qid, [])
                if page.page_number not in question_pages[current_qid]:
                    question_pages[current_qid].append(page.page_number)
                mapped[current_qid].append(cleaned_line)

    # Clean empty strings
    clean_answers = {qid: " ".join(parts).strip() for qid, parts in mapped.items() if " ".join(parts).strip()}
    return clean_answers, question_pages


# ---------------------------------------------------------------------------
# Semantic Similarity (Sentence Transformers)
# ---------------------------------------------------------------------------
class SemanticScorer:
    """Computes dense cosine similarity using all-MiniLM-L6-v2."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading sentence-transformer '%s'...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _split_chunks(text: str) -> list[str]:
        parts = [p.strip() for p in re.split(r"[.!?\n;]+", text) if p.strip()]
        chunks = [p for p in parts if len(p.split()) >= 3]
        if text.strip() and text.strip() not in chunks:
            chunks.append(text.strip())
        return chunks

    def score_answer(
        self, student_answer: str, question: RubricQuestion
    ) -> tuple[float, dict[str, float]]:
        """Return (overall_model_similarity, {point_id: best_similarity})."""
        if not student_answer.strip():
            return 0.0, {kp.point_id: 0.0 for kp in question.key_points}

        model = self._ensure_loaded()
        chunks = self._split_chunks(student_answer)

        student_full_emb = model.encode(student_answer, convert_to_tensor=True, normalize_embeddings=True)
        model_ans_emb = model.encode(question.model_answer, convert_to_tensor=True, normalize_embeddings=True)
        overall_sim = float(util.cos_sim(student_full_emb, model_ans_emb).item())
        overall_sim = round(max(0.0, min(1.0, overall_sim)), 4)

        chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
        kp_texts = [kp.description for kp in question.key_points]
        kp_embs = model.encode(kp_texts, convert_to_tensor=True, normalize_embeddings=True)

        sim_matrix = util.cos_sim(kp_embs, chunk_embs)
        kp_scores: dict[str, float] = {}
        for idx, kp in enumerate(question.key_points):
            best_sim = float(sim_matrix[idx].max().item())
            kp_scores[kp.point_id] = round(max(0.0, min(1.0, best_sim)), 4)

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


def _grade_objective_answer(
    question: RubricQuestion,
    student_answer: str,
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
                if sim >= 0.75:
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
                        if SequenceMatcher(None, tw, sw).ratio() >= 0.78:
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
        cfg: GradingConfig = GRADING_CONFIG,
    ) -> None:
        self.cfg = cfg
        self.api_key = GEMINI_API_KEY
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
    ) -> LLMGradeOutput:
        """Deterministic local heuristic rubric scorer using MiniLM + diagram structure."""
        if question.question_type in ("mcq", "fill_blank", "one_word"):
            return _grade_objective_answer(question, student_answer)

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

        # Strict thresholds for subjective Short & Long answers
        hit_threshold = max(0.65, self.cfg.keypoint_hit_threshold)
        partial_threshold = max(0.42, self.cfg.keypoint_partial_threshold)
        word_count = len(student_answer.split())

        for kp in question.key_points:
            kp_max = round((kp.weight / total_weight) * text_budget, 2)
            sim = kp_sims.get(kp.point_id, 0.0)
            blended_sim = (1.0 - self.cfg.semantic_blend_weight) * sim + (
                self.cfg.semantic_blend_weight * overall_sim
            )

            # Strict penalty if answer is too short for a multi-mark question
            if question.max_marks >= 3.0 and word_count < 6:
                blended_sim *= 0.5

            if blended_sim >= hit_threshold:
                status: Literal["Hit", "Partial", "Missed"] = "Hit"
                awarded = kp_max
                note = f"Key concept matched (similarity {sim:.2f})."
            elif blended_sim >= partial_threshold:
                status = "Partial"
                awarded = _round_quarter(kp_max * 0.50)
                note = f"Partially addressed (similarity {sim:.2f})."
                missing.append(f"Elaborate further on: {kp.description}")
            else:
                status = "Missed"
                awarded = 0.0
                note = f"Not clearly detected in answer (similarity {sim:.2f})."
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
                f"(semantic similarity: {overall_sim:.2f})."
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

    def _grade_with_gemini(
        self,
        question: RubricQuestion,
        student_answer: str,
        overall_sim: float,
        kp_sims: dict[str, float],
        diagram_eval: DiagramEvaluation | None,
    ) -> LLMGradeOutput:
        """Use Gemini 2.5 Flash with Pydantic structured output for step-wise grading."""
        if question.question_type in ("mcq", "fill_blank", "one_word"):
            return _grade_objective_answer(question, student_answer)

        from google import genai
        from google.genai import types

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
                f"- {kp.point_id} (max {kp_max} marks, MiniLM similarity={kp_sims.get(kp.point_id, 0.0):.3f}): "
                f"{kp.description}"
            )

        diagram_summary = (
            diagram_eval.model_dump_json(indent=2)
            if diagram_eval is not None
            else ("MISSING (required by rubric)" if question.requires_diagram else "Not required")
        )

        prompt = (
            "You are a strict and rigorous STEM examiner grading a student's OCR-transcribed handwritten answer.\n"
            "CRITICAL GRADING RULES BY QUESTION TYPE:\n"
            "1. OBJECTIVE QUESTIONS (MCQ, Fill in blanks, One word):\n"
            "   - Grade directly based on correctness of the selected option letter or target keyword.\n"
            "   - If correct, award FULL marks. If incorrect or missing, award 0 marks.\n"
            "2. SUBJECTIVE QUESTIONS (Short Answer, Long Answer):\n"
            "   - GRADE WITH HIGH ACADEMIC RIGOR AND STRICTNESS.\n"
            "   - Award marks ONLY when the student explicitly demonstrates clear conceptual understanding and mentions required technical terms/principles.\n"
            "   - Do NOT give generous marks for vague, circular, or superficial answers.\n"
            "   - For Long Answers, check that all necessary sub-points, mechanisms, formulas, or diagrams are present. Deduct marks for omitted steps.\n\n"
            f"Question Type: {question.question_type.upper()}\n"
            f"Question ({question.question_id}, Max Marks = {question.max_marks}):\n{question.question_text}\n\n"
            f"Model Answer:\n{question.model_answer}\n\n"
            f"Rubric Key Points (Text budget = {round(text_budget, 2)} marks):\n"
            + "\n".join(kp_lines)
            + f"\n\nStudent Transcribed Answer (Overall MiniLM cosine similarity = {overall_sim:.3f}):\n"
            + (student_answer if student_answer.strip() else "[NO TEXT ANSWER DETECTED]")
            + f"\n\nDiagram Evaluation (Diagram budget = {diagram_budget} marks):\n{diagram_summary}\n\n"
            "Return strict JSON matching the schema:\n"
            "- step_wise_breakdown: list covering EVERY key_point id in order, using the provided similarity_score, "
            "status ('Hit', 'Partial', or 'Missed'), max_point_marks, awarded_point_marks, and concise notes.\n"
            "- missing_key_points: list of missed or incomplete concepts.\n"
            f"- marks_awarded: total marks out of {question.max_marks} (sum of awarded_point_marks + diagram marks if required), rounded to nearest 0.25.\n"
            "- student_feedback: 1-3 short, actionable, constructive sentences."
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
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
        """Associate an unused detected diagram on the question's page(s) with the question."""
        # Prefer diagrams on the exact pages where this question's text appeared
        for page in ocr_result.pages:
            if q_pages and page.page_number not in q_pages:
                continue
            for diag in page.diagrams:
                key = str(diag.crop_path)
                if key not in used_diagrams:
                    used_diagrams.add(key)
                    return diag.crop_path

        # Fallback: if question requires a diagram and none matched its exact page, pick next unused diagram
        if question.requires_diagram:
            for page in ocr_result.pages:
                for diag in page.diagrams:
                    key = str(diag.crop_path)
                    if key not in used_diagrams:
                        used_diagrams.add(key)
                        return diag.crop_path
        return None

    def grade_exam(
        self, ocr_result: "OCRPipelineResult", rubric: ExamRubric
    ) -> ExamGradingReport:
        """Map OCR text to rubric questions, evaluate diagrams, and grade step-by-step."""
        mapped_answers, question_pages = parse_questions_with_pages(ocr_result.pages)
        used_diagrams: set[str] = set()
        results: list[QuestionGradeResult] = []

        for question in rubric.questions:
            qid = question.question_id
            student_answer = mapped_answers.get(qid, "")
            q_pages = question_pages.get(qid, [])

            overall_sim, kp_sims = self.semantic_scorer.score_answer(student_answer, question)

            diagram_path: Path | None = None
            diagram_eval: DiagramEvaluation | None = None
            if question.requires_diagram or any(
                p.page_number in q_pages and p.diagrams for p in ocr_result.pages
            ):
                diagram_path = self._select_diagram_for_question(
                    question, q_pages, ocr_result, used_diagrams
                )

            if diagram_path is not None:
                # Extract text/labels/formulas inside diagram crop using RapidOCR
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    fast_ocr = RapidOCR()
                    d_res, _ = fast_ocr(str(diagram_path))
                    if d_res:
                        diag_text = " ".join(r[1].strip() for r in d_res if r[1].strip())
                        if diag_text:
                            diag_clean = _clean_ocr_handwriting_text(diag_text)
                            student_answer = f"{student_answer} {diag_clean}".strip()
                            mapped_answers[qid] = student_answer
                except Exception as exc:
                    logger.debug("RapidOCR diagram text extraction skipped for %s: %s", qid, exc)

                if not self.offline and self.api_key:
                    try:
                        diagram_eval = _evaluate_diagram_gemini(
                            diagram_path, question, self.api_key
                        )
                    except Exception as exc:
                        logger.warning(
                            "Gemini diagram evaluation failed for %s (%s); using LocalVLMFallback.",
                            qid,
                            exc,
                        )
                        diagram_eval = self.local_vlm.evaluate_diagram(
                            diagram_path, question.question_text, question.diagram_criteria
                        )
                else:
                    diagram_eval = self.local_vlm.evaluate_diagram(
                        diagram_path, question.question_text, question.diagram_criteria
                    )

            if not student_answer.strip() and diagram_eval is None:
                grade_output = self._grade_offline(
                    question, student_answer, overall_sim, kp_sims, diagram_eval
                )
            elif not self.offline and self.api_key:
                try:
                    grade_output = self._grade_with_gemini(
                        question, student_answer, overall_sim, kp_sims, diagram_eval
                    )
                except Exception as exc:
                    logger.warning(
                        "Gemini grading failed for %s (%s); falling back to offline scorer.",
                        qid,
                        exc,
                    )
                    grade_output = self._grade_offline(
                        question, student_answer, overall_sim, kp_sims, diagram_eval
                    )
            else:
                grade_output = self._grade_offline(
                    question, student_answer, overall_sim, kp_sims, diagram_eval
                )

            results.append(
                QuestionGradeResult(
                    question_id=qid,
                    question_type=question.question_type,
                    student_answer=student_answer,
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
            total_marks_awarded=total_awarded,
            total_max_marks=total_max,
            mapped_answers=mapped_answers,
            question_results=results,
        )
