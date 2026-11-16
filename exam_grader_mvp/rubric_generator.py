"""Automated Rubric Generator from Teacher Question Photo / Text and Subject/Marks input."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MODEL_NAME, get_gemini_api_key
from grader import ExamRubric, KeyPoint, RubricQuestion

logger = logging.getLogger(__name__)

DIAGRAM_KEYWORDS: tuple[str, ...] = (
    "diagram",
    "circuit",
    "sketch",
    "draw",
    "schematic",
    "graph",
    "plot",
    "figure",
    "label",
    "labeled",
    "chart",
    "flowchart",
)


def filter_hindi_keep_english(text: str) -> str:
    """Filter out Hindi / Devanagari lines from bilingual exam papers, keeping only English with numbering."""
    marker_pattern = re.compile(
        r"^\s*(\(?\s*(?:\d{1,2}|[ivxIVX]{1,4})\s*[\)\.\-:]|\(\s*(?:\d{1,2}|[ivxIVX]{1,4})\s*\))\s+"
    )
    cleaned_lines: list[str] = []
    pending_marker = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        dev_count = len(re.findall(r"[\u0900-\u097F]", line))
        latin_count = len(re.findall(r"[A-Za-z]", line))

        m = marker_pattern.match(line)
        marker = m.group(1).strip() if m else ""

        # Pure or predominantly Hindi line
        if dev_count > 0 and latin_count < 4:
            if marker:
                pending_marker = marker
            continue

        # English or mixed line: strip out any Devanagari characters
        clean_en = re.sub(r"[\u0900-\u097F]+", "", line).strip()
        clean_en = re.sub(r"\s{2,}", " ", clean_en)

        if pending_marker and not marker_pattern.match(clean_en):
            clean_en = f"{pending_marker} {clean_en}"
            pending_marker = ""
        elif marker_pattern.match(clean_en):
            pending_marker = ""

        if clean_en:
            cleaned_lines.append(clean_en)

    return "\n".join(cleaned_lines)


class RubricGenerator:
    """Generates structured exam grading rubrics from question images or text."""

    def __init__(self, api_key: str | None = GEMINI_API_KEY) -> None:
        self.api_key = api_key

    def generate(
        self,
        subject: str = "Science",
        total_marks: float = 5.0,
        question_image_path: str | Path | None = None,
        question_images: list[Image.Image] | None = None,
        question_text: str | None = None,
        teacher_hints: str = "",
        answer_key_image_path: str | Path | None = None,
        answer_key_images: list[Image.Image] | None = None,
        answer_key_text: str = "",
        force_offline: bool = False,
    ) -> ExamRubric:
        """Generate an ExamRubric using Gemini VLM or local heuristic fallback.

        Supports Dual Reference Input:
        - If Official Answer Key (file/images/text) is provided alongside the Marking Scheme / Question Paper,
          student answers are evaluated against the Official Answer Key using the Marking Scheme's weightage.
        - If only the Marking Scheme / Question Paper is provided, expected key points and model answers
          are automatically inferred from the Marking Scheme.
        """
        has_answer_key = bool(
            (answer_key_text and answer_key_text.strip())
            or answer_key_image_path
            or answer_key_images
            or (teacher_hints and teacher_hints.strip())
        )
        ref_mode = "dual_reference" if has_answer_key else "auto_inferred"

        active_key = get_gemini_api_key() or self.api_key
        if not force_offline and active_key:
            self.api_key = active_key
            try:
                rubric = self._generate_with_gemini(
                    subject=subject,
                    total_marks=total_marks,
                    question_image_path=question_image_path,
                    question_images=question_images,
                    question_text=question_text,
                    teacher_hints=teacher_hints,
                    answer_key_image_path=answer_key_image_path,
                    answer_key_images=answer_key_images,
                    answer_key_text=answer_key_text,
                )
                rubric.reference_mode = ref_mode
                return rubric
            except Exception as exc:
                logger.warning(
                    "Gemini auto-rubric generation failed (%s); falling back to local generator.",
                    exc,
                )

        rubric = self._generate_local(
            subject=subject,
            total_marks=total_marks,
            question_image_path=question_image_path,
            question_text=question_text,
            teacher_hints=teacher_hints,
            answer_key_image_path=answer_key_image_path,
            answer_key_text=answer_key_text,
        )
        rubric.reference_mode = ref_mode
        return rubric

    def _generate_with_gemini(
        self,
        subject: str,
        total_marks: float,
        question_image_path: str | Path | None,
        question_images: list[Image.Image] | None,
        question_text: str | None,
        teacher_hints: str,
        answer_key_image_path: str | Path | None = None,
        answer_key_images: list[Image.Image] | None = None,
        answer_key_text: str = "",
    ) -> ExamRubric:
        """Call Gemini Flash with structured Pydantic output to create the rubric."""
        from google import genai
        from google.genai import types

        assert self.api_key is not None
        client = genai.Client(api_key=self.api_key)

        has_answer_key = bool(
            (answer_key_text and answer_key_text.strip())
            or answer_key_image_path
            or answer_key_images
            or (teacher_hints and teacher_hints.strip())
        )

        prompt_parts: list[str] = [
            f"You are a Chief Academic Examiner and Curriculum Specialist in '{subject}'.",
            f"Subject: {subject}",
            f"Target Total Marks: {total_marks}",
        ]

        if has_answer_key:
            prompt_parts.append(
                "\n=== DUAL REFERENCE MODE (MARKING SCHEME + OFFICIAL ANSWER KEY) ===\n"
                "The teacher has uploaded BOTH the Question Paper / Marking Scheme AND an Official Answer Key.\n"
                "You MUST:\n"
                "1. Extract each question and its exact mark weightage from the Question Paper / Marking Scheme.\n"
                "2. Extract the authoritative model_answer and weighted key_points directly from the Official Answer Key "
                "so student answers will be evaluated strictly against the Official Answer Key using the Marking Scheme's weightage."
            )
        else:
            prompt_parts.append(
                "\n=== SINGLE REFERENCE MODE (AUTO-INFERRED FROM MARKING SCHEME) ===\n"
                "The teacher has uploaded ONLY the Question Paper / Marking Scheme (without a separate Official Answer Key).\n"
                "You MUST automatically generate/infer the expected model_answer and weighted key_points from the Question Paper / Marking Scheme."
            )

        cleaned_q_text = filter_hindi_keep_english(question_text.strip()) if question_text else ""
        if cleaned_q_text:
            prompt_parts.append(f"\nQuestion Paper / Marking Scheme Text (English Only):\n\"\"\"{cleaned_q_text}\"\"\"")

        combined_key_text = "\n".join(
            part.strip()
            for part in [answer_key_text, teacher_hints]
            if part and part.strip()
        )
        if combined_key_text:
            cleaned_key_text = filter_hindi_keep_english(combined_key_text)
            prompt_parts.append(
                f"\nOfficial Answer Key / Teacher Marking Guidelines:\n\"\"\"{cleaned_key_text}\"\"\""
            )

        prompt_parts.extend(
            [
                "\nINSTRUCTIONS FOR RUBRIC GENERATION:",
                "CRITICAL BILINGUAL RULE: The exam paper or answer key may contain both Hindi and English. COMPLETELY IGNORE all Hindi (Devanagari) text and cover-page instructions. Extract and formulate all questions and model answers EXCLUSIVELY in English.",
                "1. QUESTION NUMBERING & MULTI-PART PAPERS:",
                "   - If the paper has a single continuous numbering sequence (Q1, Q2, Q3 ... Q23), name question_id as 'Q1', 'Q2', 'Q3', etc.",
                "   - If the paper is divided into Parts (e.g. Part I, Part II, Part III) where question numbering restarts at Q.1 inside each Part:",
                "     * For Part I with Q.1(A) and Q.1(B), use question_id 'Q1A' and 'Q1B'.",
                "     * For Part II questions (Q.1, Q.2, Q.3, Q.4, Q.5, Q.6...), use question_id 'P2_Q1', 'P2_Q2', 'P2_Q3', 'P2_Q4', 'P2_Q5', 'P2_Q6', etc.",
                "     * For Part III questions (Q.1, Q.2, Q.3...), use question_id 'P3_Q1', 'P3_Q2', 'P3_Q3', etc.",
                "     * Extract EVERY SINGLE numbered question printed in Part I, Part II, and Part III as its own separate RubricQuestion item (never skip or merge the last question of a section, such as Part III Q.3).",
                "2. MULTI-ITEM OBJECTIVE / SMALL ANSWER BLOCKS:",
                "   - If a main question (such as Q1..Q5 in board papers, or Q1A / Q1B in university papers) contains multiple sub-items ((i)..(v) or (i)..(vi) worth 1 mark each), keep it as ONE RubricQuestion (e.g. Q1A with max_marks=5, Q1B with max_marks=5) and create 1 key_point per sub-item in exact order with its expected answer.",
                "3. INTERNAL '[OR]' CHOICES:",
                "   - If a question has an internal '[OR]' choice, include both the main question and the '[OR]' option in question_text and model_answer so a student answering EITHER option gets full credit.",
                "4. Classify question_type for EACH question:",
                "   - 'mcq' : For Multiple Choice Questions.",
                "   - 'fill_blank' : For Fill in the Blanks questions.",
                "   - 'one_word' : For One Word / One Sentence / True-False / Match-column objective questions.",
                "   - 'short_answer' : For Short conceptual questions or blocks of short definitions (2-5 marks).",
                "   - 'long_answer' : For Long descriptive/analytical questions (6+ marks). Provide 3-5 concise key points.",
                "5. Keep question_text, model_answer, and key_point descriptions concise, accurate, and clear.",
                "6. Set requires_diagram = true ONLY if the question explicitly asks the student to draw/sketch a diagram or map.",
                "7. EXACT PRINTED MARKS: Use the exact per-question marks printed on the question paper (e.g. 5 marks for Q1A, 5 marks for Q1B, 6 marks for each Part II question, 10 marks for each Part III question). Only distribute Target Total Marks if no marks are printed on the paper.",
                f"8. Set exam_title to '{subject} Examination Rubric'.",
                "\nReturn strict JSON matching the ExamRubric schema.",
            ]
        )

        import io

        def _to_jpeg_part(pil_img: Image.Image, max_w: int = 1400, quality: int = 84) -> Any:
            rgb = pil_img.convert("RGB")
            if rgb.width > max_w:
                ratio = max_w / float(rgb.width)
                rgb = rgb.resize((max_w, int(round(rgb.height * ratio))))
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

        full_prompt = "\n".join(prompt_parts)
        contents: list[Any] = [full_prompt]

        if question_images:
            contents.append("--- QUESTION PAPER / MARKING SCHEME PAGES ---")
            contents.extend(_to_jpeg_part(img) for img in question_images)
        elif question_image_path:
            img_path = Path(question_image_path)
            if img_path.exists():
                contents.append("--- QUESTION PAPER / MARKING SCHEME IMAGE ---")
                contents.append(_to_jpeg_part(Image.open(img_path)))

        if answer_key_images:
            contents.append("--- OFFICIAL ANSWER KEY PAGES ---")
            contents.extend(_to_jpeg_part(img) for img in answer_key_images)
        elif answer_key_image_path:
            ak_path = Path(answer_key_image_path)
            if ak_path.exists():
                contents.append("--- OFFICIAL ANSWER KEY IMAGE ---")
                contents.append(_to_jpeg_part(Image.open(ak_path)))

        import time

        last_err: Exception | None = None
        response = None
        for m_name in dict.fromkeys([GEMINI_MODEL_NAME, *GEMINI_FALLBACK_MODELS]):
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=ExamRubric,
                        temperature=0.0,
                    ),
                )
                break
            except Exception as exc:
                last_err = exc
                time.sleep(0.25)
        if response is None:
            raise RuntimeError(f"Gemini rubric generation failed across fallback models: {last_err}")

        if response.parsed is not None:
            rubric = ExamRubric.model_validate(response.parsed)
        else:
            rubric = ExamRubric.model_validate_json(response.text)

        # Normalize question IDs (e.g. PartII_Q1 -> P2_Q1, PartIII_Q2 -> P3_Q2, PartI_Q1A -> Q1A)
        for q in rubric.questions:
            qid_clean = q.question_id.strip()
            qid_clean = re.sub(r"^(?:Part|Section)[\s_\-]*I(?:I|1)?[\s_\-]*Q\.?\s*1\s*\(?\s*([AB])\s*\)?$", r"Q1\1", qid_clean, flags=re.IGNORECASE)
            qid_clean = re.sub(r"^Q\.?\s*1\s*\(?\s*([AB])\s*\)?$", r"Q1\1", qid_clean, flags=re.IGNORECASE)
            qid_clean = re.sub(r"^(?:Part|Section|P)[\s_\-]*(?:II|2)[\s_\-]*Q(?:uestion)?[\s_\-\.]*(\d+)$", r"P2_Q\1", qid_clean, flags=re.IGNORECASE)
            qid_clean = re.sub(r"^(?:Part|Section|P)[\s_\-]*(?:III|3)[\s_\-]*Q(?:uestion)?[\s_\-\.]*(\d+)$", r"P3_Q\1", qid_clean, flags=re.IGNORECASE)
            q.question_id = qid_clean.upper()

        # Preserve printed per-question marks on multi-question exam papers; only rescale single/custom rubrics
        current_sum = sum(q.max_marks for q in rubric.questions)
        if (
            current_sum > 0
            and abs(current_sum - total_marks) > 0.01
            and len(rubric.questions) < 3
            and current_sum < 15.0
        ):
            ratio = total_marks / current_sum
            for q in rubric.questions:
                q.max_marks = round(q.max_marks * ratio, 2)
        elif not rubric.questions:
            rubric.questions.append(
                RubricQuestion(
                    question_id="Q1",
                    question_text=question_text or "Question",
                    model_answer=teacher_hints or "Model answer",
                    key_points=[
                        KeyPoint(
                            point_id="KP1",
                            description="Core concept and explanation.",
                            weight=1.0,
                        )
                    ],
                    requires_diagram=False,
                    max_marks=total_marks,
                )
            )

        return rubric

    def _generate_local(
        self,
        subject: str,
        total_marks: float,
        question_image_path: str | Path | None,
        question_text: str | None,
        teacher_hints: str,
        answer_key_image_path: str | Path | None = None,
        answer_key_text: str = "",
    ) -> ExamRubric:
        """Deterministic offline fallback rubric generator using heuristics and OCR text."""
        raw_text = (question_text or "").strip()

        # If question text wasn't provided directly, attempt fast printed OCR using RapidOCR
        if not raw_text and question_image_path:
            p = Path(question_image_path)
            if p.exists():
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    engine = RapidOCR()
                    res, _ = engine(str(p))
                    if res:
                        raw_text = "\n".join([line[1].strip() for line in res if line[1].strip()])
                except Exception as exc:
                    logger.warning("RapidOCR failed on question image: %s", exc)

        raw_key_text = (answer_key_text or "").strip()
        if not raw_key_text and answer_key_image_path:
            ak_p = Path(answer_key_image_path)
            if ak_p.exists():
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    engine = RapidOCR()
                    res, _ = engine(str(ak_p))
                    if res:
                        raw_key_text = "\n".join([line[1].strip() for line in res if line[1].strip()])
                except Exception as exc:
                    logger.warning("RapidOCR failed on answer key image: %s", exc)

        combined_hints = "\n".join(
            p for p in [filter_hindi_keep_english(raw_key_text), teacher_hints.strip()] if p
        ).strip()
        from grader import parse_questions
        parsed_key_map = parse_questions(combined_hints) if combined_hints else {}

        raw_text = filter_hindi_keep_english(raw_text)
        if not raw_text:
            raw_text = f"Explain key principles of {subject}."

        # Detect questions formatted like Q1, Q.1, 1), 1., (1), (i), etc.
        q_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        detected_questions: list[str] = []
        curr: list[str] = []

        q_marker_re = re.compile(
            r"^\s*(?:Q(?:uestion)?\s*[\.\-:#]?\s*\d{1,2}|Ans(?:wer)?|\d{1,2}[\)\.]|\(\s*(?:\d{1,2}|[ivxIVX]{1,4})\s*\))\s*",
            re.IGNORECASE,
        )
        has_any_marker = any(q_marker_re.match(ln) for ln in q_lines)

        # Common exam cover page terms to skip
        skip_terms = (
            "roll no", "serial number", "total printed pages", "maximum marks",
            "time :", "high school", "instructions", "निर्देश", "http", ".com",
            "p.t.o", "set /"
        )

        for line in q_lines:
            lower_line = line.lower()
            if any(term in lower_line for term in skip_terms):
                continue

            if q_marker_re.match(line):
                if curr:
                    detected_questions.append(" ".join(curr))
                curr = [line]
            elif curr:
                curr.append(line)
            elif not has_any_marker:
                curr.append(line)

        if curr:
            detected_questions.append(" ".join(curr))

        if not detected_questions:
            detected_questions = [raw_text]

        num_q = len(detected_questions)
        marks_per_q = round(total_marks / num_q, 2)

        # Regex to detect inline marks e.g. [5 Marks], (5 marks), 5M
        inline_marks_re = re.compile(
            r"[\(\[\{]\s*(\d+(?:\.\d+)?)\s*(?:marks?|pts?|m)\s*[\)\]\}]", re.IGNORECASE
        )

        questions: list[RubricQuestion] = []
        for idx, q_str in enumerate(detected_questions, start=1):
            qid = f"Q{idx}"
            clean_q = q_marker_re.sub("", q_str).strip()

            detected_q_marks = marks_per_q
            marks_match = inline_marks_re.search(clean_q)
            if marks_match:
                try:
                    detected_q_marks = float(marks_match.group(1))
                except ValueError:
                    pass
                clean_q = inline_marks_re.sub("", clean_q).strip()

            clean_q = re.sub(r"^[\.\-:\s]+", "", clean_q)
            if not clean_q:
                clean_q = q_str

            # Check if question requires a diagram
            lower_q = clean_q.lower()
            needs_diagram = any(kw in lower_q for kw in DIAGRAM_KEYWORDS)
            diag_criteria = (
                f"Clearly drawn and accurately labeled {subject} diagram with standard scientific symbols."
                if needs_diagram
                else ""
            )

            specific_key = parsed_key_map.get(qid, "") or combined_hints

            # Detect question type
            if any(opt in lower_q for opt in ["(a)", "(b)", "choose the correct", "multiple choice", "mcq"]):
                q_type = "mcq"
                model_ans = specific_key if specific_key else "Correct option selected from given alternatives."
                key_points = [KeyPoint(point_id="KP1", description=model_ans[:120], weight=1.0)]
            elif "____" in clean_q or "fill in" in lower_q or "blank" in lower_q:
                q_type = "fill_blank"
                model_ans = specific_key if specific_key else "Accurate missing scientific term or value."
                key_points = [KeyPoint(point_id="KP1", description=model_ans[:120], weight=1.0)]
            elif "one word" in lower_q or "name the" in lower_q or detected_q_marks <= 1.0:
                q_type = "one_word"
                model_ans = specific_key if specific_key else "Precise technical term or name."
                key_points = [KeyPoint(point_id="KP1", description=model_ans[:120], weight=1.0)]
            elif detected_q_marks >= 4.0:
                q_type = "long_answer"
            else:
                q_type = "short_answer"

            if q_type in ("short_answer", "long_answer"):
                source_for_kp = specific_key if specific_key else clean_q
                sentences = [
                    s.strip()
                    for s in re.split(r"[.!?;\n]+", source_for_kp)
                    if len(s.strip()) > 6
                ]
                if len(sentences) < 2:
                    key_points = [
                        KeyPoint(
                            point_id="KP1",
                            description=specific_key if specific_key else f"Definition and core concept of {clean_q[:60]}",
                            weight=0.4,
                        ),
                        KeyPoint(
                            point_id="KP2",
                            description="Detailed explanation of the process, key inputs, mechanism, and outcomes.",
                            weight=0.4,
                        ),
                        KeyPoint(
                            point_id="KP3",
                            description="Accurate terminology, facts, and conclusions.",
                            weight=0.2,
                        ),
                    ]
                else:
                    total_s = len(sentences)
                    key_points = [
                        KeyPoint(
                            point_id=f"KP{s_idx}",
                            description=sent if specific_key else f"Address concept: {sent}",
                            weight=round(1.0 / total_s, 2),
                        )
                        for s_idx, sent in enumerate(sentences[:6], start=1)
                    ]

                model_ans = (
                    specific_key
                    if specific_key
                    else f"Comprehensive model answer explaining {clean_q} with standard {subject} conventions."
                )

            questions.append(
                RubricQuestion(
                    question_id=qid,
                    question_text=clean_q,
                    question_type=q_type,
                    model_answer=model_ans,
                    key_points=key_points,
                    requires_diagram=needs_diagram,
                    diagram_criteria=diag_criteria,
                    max_marks=detected_q_marks,
                )
            )

        return ExamRubric(
            exam_title=f"{subject} Examination Rubric",
            questions=questions,
        )
