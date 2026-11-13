"""Automated Rubric Generator from Teacher Question Photo / Text and Subject/Marks input."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field

from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
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


class RubricGenerator:
    """Generates structured exam grading rubrics from question images or text."""

    def __init__(self, api_key: str | None = GEMINI_API_KEY) -> None:
        self.api_key = api_key

    def generate(
        self,
        subject: str = "Science",
        total_marks: float = 5.0,
        question_image_path: str | Path | None = None,
        question_text: str | None = None,
        teacher_hints: str = "",
        force_offline: bool = False,
    ) -> ExamRubric:
        """Generate an ExamRubric using Gemini VLM or local heuristic fallback."""
        if not force_offline and self.api_key:
            try:
                return self._generate_with_gemini(
                    subject=subject,
                    total_marks=total_marks,
                    question_image_path=question_image_path,
                    question_text=question_text,
                    teacher_hints=teacher_hints,
                )
            except Exception as exc:
                logger.warning(
                    "Gemini auto-rubric generation failed (%s); falling back to local generator.",
                    exc,
                )

        return self._generate_local(
            subject=subject,
            total_marks=total_marks,
            question_image_path=question_image_path,
            question_text=question_text,
            teacher_hints=teacher_hints,
        )

    def _generate_with_gemini(
        self,
        subject: str,
        total_marks: float,
        question_image_path: str | Path | None,
        question_text: str | None,
        teacher_hints: str,
    ) -> ExamRubric:
        """Call Gemini 2.5 Flash with structured Pydantic output to create the rubric."""
        from google import genai
        from google.genai import types

        assert self.api_key is not None
        client = genai.Client(api_key=self.api_key)

        prompt_parts: list[str] = [
            f"You are a Chief Academic Examiner and Curriculum Specialist in '{subject}'.",
            "The teacher has provided an exam question (via attached image and/or text), the subject, and total marks.",
            f"Subject: {subject}",
            f"Target Total Marks: {total_marks}",
        ]

        if question_text and question_text.strip():
            prompt_parts.append(f"\nTeacher Provided Question Text:\n\"\"\"{question_text.strip()}\"\"\"")

        if teacher_hints and teacher_hints.strip():
            prompt_parts.append(f"\nTeacher Solution / Grading Hints:\n\"\"\"{teacher_hints.strip()}\"\"\"")

        prompt_parts.extend(
            [
                "\nINSTRUCTIONS FOR RUBRIC GENERATION:",
                "1. Accurately transcribe or extract each question from the image or text.",
                "2. Assign question_id sequentially: Q1, Q2, etc.",
                "3. Write a comprehensive, gold-standard model_answer for each question.",
                "4. Break down the model answer into 2 to 5 clear, distinct, objective key_points with IDs (KP1, KP2, etc.).",
                "   - Assign relative weights to key points reflecting their academic importance.",
                "5. Detect whether the question asks the student to draw/sketch/construct any visual element (diagram, circuit, graph, chart, labeled anatomy, etc.):",
                "   - Set requires_diagram = true if drawing is requested or essential.",
                "   - If true, specify precise diagram_criteria detailing required components, standard symbols, and expected labels.",
                "   - If false, set requires_diagram = false and diagram_criteria = ''.",
                "6. Distribute max_marks across questions so their sum equals exactly the Target Total Marks.",
                f"7. Set exam_title to '{subject} Examination Rubric'.",
                "\nReturn strict JSON matching the ExamRubric schema.",
            ]
        )

        full_prompt = "\n".join(prompt_parts)
        contents: list[Any] = [full_prompt]

        if question_image_path:
            img_path = Path(question_image_path)
            if img_path.exists():
                pil_img = Image.open(img_path).convert("RGB")
                contents.append(pil_img)

        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExamRubric,
                temperature=0.1,
            ),
        )

        if response.parsed is not None:
            rubric = ExamRubric.model_validate(response.parsed)
        else:
            rubric = ExamRubric.model_validate_json(response.text)

        # Ensure total marks match requested target marks
        current_sum = sum(q.max_marks for q in rubric.questions)
        if current_sum > 0 and abs(current_sum - total_marks) > 0.01:
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
    ) -> ExamRubric:
        """Deterministic offline fallback rubric generator using heuristics and OCR text."""
        raw_text = (question_text or "").strip()

        # If question text wasn't provided directly, attempt OCR on the image
        if not raw_text and question_image_path:
            p = Path(question_image_path)
            if p.exists():
                try:
                    from ocr_engine import OCREngine

                    engine = OCREngine()
                    ocr_res = engine.process_pages([p], debug_dir=p.parent / "debug_question_crops")
                    raw_text = ocr_res.full_transcript.strip()
                except Exception as exc:
                    logger.warning("Local OCR failed on question image: %s", exc)

        if not raw_text:
            raw_text = f"Explain key principles of {subject}."

        # Detect questions formatted like Q1, Q.1, 1), 1., etc.
        q_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
        detected_questions: list[str] = []
        curr: list[str] = []

        q_marker_re = re.compile(
            r"^\s*(?:Q(?:uestion)?\s*[\.\-:#]?\s*\d{1,2}|Ans(?:wer)?|\d{1,2}[\)\.])\s*",
            re.IGNORECASE,
        )
        has_any_marker = any(q_marker_re.match(ln) for ln in q_lines)

        for line in q_lines:
            if q_marker_re.match(line):
                if curr:
                    detected_questions.append(" ".join(curr))
                curr = [line]
            elif curr:
                # Discard non-question instructions like "Instructions to Candidates:"
                if "instruction" in line.lower() or "candidate" in line.lower():
                    continue
                curr.append(line)
            elif not has_any_marker:
                # If no Q1/Q2 markers on the page, keep all text as single question
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

            # Heuristic key-point extraction
            sentences = [
                s.strip()
                for s in re.split(r"[.!?;\n]+", clean_q)
                if len(s.strip()) > 8
            ]
            if len(sentences) < 2:
                key_points = [
                    KeyPoint(
                        point_id="KP1",
                        description=f"State definition and core principles related to: {clean_q[:65]}...",
                        weight=0.4 if needs_diagram else 0.5,
                    ),
                    KeyPoint(
                        point_id="KP2",
                        description="Accurate scientific explanation, mathematical formula, or step derivation.",
                        weight=0.35,
                    ),
                    KeyPoint(
                        point_id="KP3",
                        description="Correct technical terminology, units, and clear conclusions.",
                        weight=0.25 if needs_diagram else 0.15,
                    ),
                ]
            else:
                total_s = len(sentences)
                key_points = [
                    KeyPoint(
                        point_id=f"KP{s_idx}",
                        description=f"Address concept: {sent}",
                        weight=round(1.0 / total_s, 2),
                    )
                    for s_idx, sent in enumerate(sentences, start=1)
                ]

            model_ans = (
                teacher_hints.strip()
                if teacher_hints.strip()
                else f"Comprehensive model answer explaining {clean_q} with standard {subject} conventions."
            )

            questions.append(
                RubricQuestion(
                    question_id=qid,
                    question_text=clean_q,
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
