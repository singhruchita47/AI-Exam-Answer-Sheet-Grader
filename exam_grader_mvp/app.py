"""FastAPI web UI for testing the Automated Exam Answer Sheet Checking Engine."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import GEMINI_API_KEY, TROCR_MODEL_NAME, get_gemini_api_key
from grader import ExamGrader, ExamRubric, RubricQuestion, SemanticScorer
from ocr_engine import OCREngine
from rubric_generator import RubricGenerator, filter_hindi_keep_english

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exam_grader_ui")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
UPLOADS_DIR = BASE_DIR / "ui_uploads"
DEBUG_CROPS_DIR = BASE_DIR / "debug_crops"

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Exam Grader Testing UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/debug_crops", StaticFiles(directory=str(DEBUG_CROPS_DIR)), name="debug_crops"
)
app.mount("/ui_uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="ui_uploads")

_ocr_engine: OCREngine | None = None
_semantic_scorer: SemanticScorer | None = None


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine(trocr_model_name=TROCR_MODEL_NAME)
    return _ocr_engine


def get_semantic_scorer() -> SemanticScorer:
    global _semantic_scorer
    if _semantic_scorer is None:
        _semantic_scorer = SemanticScorer()
    return _semantic_scorer


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the testing web UI HTML file."""
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"), status_code=200
    )


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    """Return backend configuration, default rubrics, and API key status."""
    rubric_path = BASE_DIR / "sample_rubric.json"
    default_rubric = ""
    if rubric_path.exists():
        default_rubric = rubric_path.read_text(encoding="utf-8")

    photo_rubric_path = BASE_DIR / "sample_rubric_photosynthesis.json"
    photo_rubric = ""
    if photo_rubric_path.exists():
        photo_rubric = photo_rubric_path.read_text(encoding="utf-8")

    return {
        "has_gemini_key": bool(get_gemini_api_key()),
        "trocr_model": TROCR_MODEL_NAME,
        "default_rubric": default_rubric,
        "photo_rubric": photo_rubric,
    }


def _extract_reference_upload(
    upload: UploadFile,
    target_dir: Path,
    prefix: str,
    offline: bool,
) -> tuple[Path | None, list[Any] | None, str]:
    """Extract image path, multi-page PIL images, and/or English text from an uploaded PDF or image."""
    image_path: Path | None = None
    pil_images: list[Any] | None = None
    extracted_text: str = ""

    file_ext = Path(upload.filename or f"{prefix}.jpg").suffix.lower() or ".jpg"
    if file_ext == ".pdf":
        import numpy as np
        import pypdfium2 as pdfium

        pdf_path = target_dir / f"{prefix}{file_ext}"
        with pdf_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        doc = pdfium.PdfDocument(pdf_path)

        extracted_chunks: list[str] = []
        max_p = min(len(doc), 25)
        for p_idx in range(max_p):
            tp = doc[p_idx].get_textpage()
            p_text = tp.get_text_range().strip()
            if len(p_text) > 40:
                extracted_chunks.append(p_text)

        if len(doc) > 0:
            first_page = doc[0].render(scale=1.2).to_pil()
            image_path = target_dir / f"{prefix}_page_1.jpg"
            first_page.save(image_path, "JPEG", quality=88)

        if not extracted_chunks:
            if not offline and get_gemini_api_key():
                pil_images = [doc[p_idx].render(scale=1.0).to_pil() for p_idx in range(max_p)]
            else:
                from rapidocr_onnxruntime import RapidOCR

                rapid_ocr = RapidOCR()
                for p_idx in range(min(max_p, 8)):
                    pil_img = doc[p_idx].render(scale=1.0).to_pil()
                    res, _ = rapid_ocr(np.array(pil_img))
                    if res:
                        lines = [r[1].strip() for r in res if r[1].strip()]
                        extracted_chunks.append("\n".join(lines))

        doc.close()
        if extracted_chunks:
            extracted_text = filter_hindi_keep_english("\n\n".join(extracted_chunks))
    else:
        image_path = target_dir / f"{prefix}{file_ext}"
        with image_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

    return image_path, pil_images, extracted_text


@app.post("/api/generate_rubric")
async def api_generate_rubric(
    subject: str = Form("Physics"),
    total_marks: float = Form(10.0),
    question_text: str = Form(""),
    teacher_hints: str = Form(""),
    answer_key_text: str = Form(""),
    use_sample_question: bool = Form(False),
    offline: bool = Form(False),
    question_image: UploadFile | None = File(None),
    answer_key_image: UploadFile | None = File(None),
) -> dict[str, Any]:
    """Generate structured grading rubric from Marking Scheme + Optional Official Answer Key."""
    image_path: Path | None = None
    question_images: list[Any] | None = None
    ak_image_path: Path | None = None
    ak_images: list[Any] | None = None

    session_id = uuid.uuid4().hex[:8]
    q_dir = UPLOADS_DIR / f"q_{session_id}"

    if use_sample_question:
        image_path = BASE_DIR / "sample_question_paper.jpg"
        if not image_path.exists():
            from create_dummy_question import generate_sample_question_paper

            generate_sample_question_paper(image_path)
    elif question_image is not None and getattr(question_image, "filename", None):
        q_dir.mkdir(parents=True, exist_ok=True)
        image_path, question_images, pdf_q_text = _extract_reference_upload(
            question_image, q_dir, "question", offline
        )
        if not question_text.strip() and pdf_q_text:
            question_text = pdf_q_text

    if answer_key_image is not None and getattr(answer_key_image, "filename", None):
        q_dir.mkdir(parents=True, exist_ok=True)
        ak_image_path, ak_images, pdf_ak_text = _extract_reference_upload(
            answer_key_image, q_dir, "answer_key", offline
        )
        if pdf_ak_text:
            answer_key_text = (
                f"{answer_key_text.strip()}\n\n{pdf_ak_text}".strip()
                if answer_key_text.strip()
                else pdf_ak_text
            )

    try:
        generator = RubricGenerator(api_key=get_gemini_api_key())
        rubric = generator.generate(
            subject=subject,
            total_marks=total_marks,
            question_image_path=image_path,
            question_images=question_images,
            question_text=question_text,
            teacher_hints=teacher_hints,
            answer_key_image_path=ak_image_path,
            answer_key_images=ak_images,
            answer_key_text=answer_key_text,
            force_offline=offline,
        )
        return rubric.model_dump()
    except Exception as exc:
        logger.exception("Error generating rubric automatically")
        raise HTTPException(status_code=500, detail=f"Failed to generate rubric: {exc}")


def _prepare_student_pages(
    session_upload_dir: Path,
    use_photosynthesis_sample: bool,
    use_sample_pages: bool,
    pages: list[UploadFile],
) -> list[Path]:
    """Save uploaded student images/PDFs (or copy samples) into session_upload_dir and return page paths."""
    page_paths: list[Path] = []
    if use_photosynthesis_sample:
        p_photo = BASE_DIR / "sample_photosynthesis_student.jpg"
        if not p_photo.exists():
            raise HTTPException(
                status_code=404, detail="sample_photosynthesis_student.jpg not found."
            )
        dest = session_upload_dir / "page_1.jpg"
        shutil.copy2(p_photo, dest)
        page_paths = [dest]
    elif use_sample_pages:
        p1 = BASE_DIR / "sample_page1.jpg"
        p2 = BASE_DIR / "sample_page2.jpg"
        if not p1.exists() or not p2.exists():
            from create_dummy_sheet import generate_sample_pages

            generate_sample_pages(BASE_DIR)
        d1 = session_upload_dir / "page_1.jpg"
        d2 = session_upload_dir / "page_2.jpg"
        shutil.copy2(p1, d1)
        shutil.copy2(p2, d2)
        page_paths = [d1, d2]
    else:
        if not pages:
            raise HTTPException(
                status_code=400, detail="No exam sheet images or PDF provided."
            )
        import pypdfium2 as pdfium

        page_counter = 1
        for upload in pages:
            file_ext = Path(upload.filename or "page.jpg").suffix.lower()
            if file_ext == ".pdf":
                pdf_dest = session_upload_dir / f"uploaded_{upload.filename}"
                with pdf_dest.open("wb") as f:
                    shutil.copyfileobj(upload.file, f)
                try:
                    doc = pdfium.PdfDocument(pdf_dest)
                    for page_idx in range(len(doc)):
                        page_img = doc[page_idx].render(scale=1.3).to_pil()
                        img_path = session_upload_dir / f"page_{page_counter}.jpg"
                        page_img.save(img_path, "JPEG", quality=88)
                        page_paths.append(img_path)
                        page_counter += 1
                    doc.close()
                except Exception as pdf_err:
                    logger.exception("Failed to render PDF: %s", pdf_err)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process PDF file: {pdf_err}",
                    )
            else:
                dest = session_upload_dir / f"page_{page_counter}{file_ext or '.jpg'}"
                with dest.open("wb") as f:
                    shutil.copyfileobj(upload.file, f)
                page_paths.append(dest)
                page_counter += 1
    return page_paths


def _build_final_payload(
    session_id: str,
    ocr_result: Any,
    report: Any,
) -> dict[str, Any]:
    """Format segmentation and grading report with client-friendly relative URLs."""
    from grader import _clean_ocr_handwriting_text

    report_dump = report.model_dump()
    for q_res in report_dump.get("question_results", []):
        if q_res.get("diagram_path"):
            diag_p = Path(q_res["diagram_path"])
            if diag_p.is_absolute() and BASE_DIR in diag_p.parents:
                q_res["diagram_path"] = str(diag_p.relative_to(BASE_DIR))

    seg_data = []
    for p in ocr_result.pages:
        rel_layout = p.layout_image_path.relative_to(BASE_DIR)
        rel_orig = (
            p.source_path.relative_to(BASE_DIR)
            if p.source_path.is_absolute() and BASE_DIR in p.source_path.parents
            else rel_layout
        )
        page_qids = [
            q_res["question_id"]
            for q_res in report_dump.get("question_results", [])
            if p.page_number in (q_res.get("page_numbers") or [])
        ]
        cleaned_page_lines = [
            _clean_ocr_handwriting_text(raw)
            for raw in (p.transcript or "").splitlines()
            if _clean_ocr_handwriting_text(raw)
        ]
        seg_data.append(
            {
                "page_number": p.page_number,
                "source_path": str(p.source_path),
                "original_image_path": str(rel_orig),
                "skew_angle_deg": p.skew_angle_deg,
                "text_lines_detected": len(p.text_lines),
                "diagrams_detected": len(p.diagrams),
                "layout_image_path": str(rel_layout),
                "page_transcript": "\n".join(cleaned_page_lines).strip(),
                "page_questions": page_qids,
            }
        )

    return {
        "session_id": session_id,
        "segmentation": seg_data,
        "grading_report": report_dump,
    }


@app.post("/api/grade")
async def api_grade(
    offline: bool = Form(False),
    strictness_level: int = Form(5),
    use_sample_pages: bool = Form(False),
    use_photosynthesis_sample: bool = Form(False),
    rubric_json: str = Form(...),
    pages: list[UploadFile] = File([]),
) -> dict[str, Any]:
    """Process uploaded pages or built-in samples, run OCR + Grading, and return JSON."""
    try:
        rubric_dict = json.loads(rubric_json)
        rubric = ExamRubric.model_validate(rubric_dict)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid rubric JSON: {exc}")

    session_id = uuid.uuid4().hex[:8]
    session_upload_dir = UPLOADS_DIR / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_crops_dir = DEBUG_CROPS_DIR / f"session_{session_id}"
    session_crops_dir.mkdir(parents=True, exist_ok=True)

    page_paths = _prepare_student_pages(
        session_upload_dir, use_photosynthesis_sample, use_sample_pages, pages
    )

    try:
        engine = get_ocr_engine()
        online_fast = not offline and bool(get_gemini_api_key())
        ocr_result = engine.process_pages(
            page_paths, debug_dir=session_crops_dir, online_fast=online_fast
        )

        grader = ExamGrader(offline=offline, strictness_level=strictness_level)
        grader.semantic_scorer = get_semantic_scorer()
        report = grader.grade_exam(ocr_result, rubric, strictness_level=strictness_level)

        return _build_final_payload(session_id, ocr_result, report)
    except Exception as exc:
        logger.exception("Error evaluating exam sheets")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/grade_stream")
async def api_grade_stream(
    offline: bool = Form(False),
    strictness_level: int = Form(5),
    use_sample_pages: bool = Form(False),
    use_photosynthesis_sample: bool = Form(False),
    rubric_json: str = Form(...),
    pages: list[UploadFile] = File([]),
) -> StreamingResponse:
    """Stream live page-by-page evaluation progress (NDJSON) followed by the final grading report."""
    try:
        rubric_dict = json.loads(rubric_json)
        rubric = ExamRubric.model_validate(rubric_dict)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid rubric JSON: {exc}")

    session_id = uuid.uuid4().hex[:8]
    session_upload_dir = UPLOADS_DIR / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_crops_dir = DEBUG_CROPS_DIR / f"session_{session_id}"
    session_crops_dir.mkdir(parents=True, exist_ok=True)

    page_paths = _prepare_student_pages(
        session_upload_dir, use_photosynthesis_sample, use_sample_pages, pages
    )
    total_pages = len(page_paths)
    lvl = max(1, min(10, int(strictness_level)))

    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def push_event(evt: dict[str, Any]) -> None:
        event_queue.put(evt)

    def worker() -> None:
        try:
            push_event(
                {
                    "type": "progress",
                    "step": "init",
                    "page": 1,
                    "total_pages": total_pages,
                    "percent": 4,
                    "message": f"Prepared {total_pages} page(s) → Starting Page 1 of {total_pages}...",
                }
            )
            engine = get_ocr_engine()
            online_fast = not offline and bool(GEMINI_API_KEY)
            ocr_result = engine.process_pages(
                page_paths,
                debug_dir=session_crops_dir,
                online_fast=online_fast,
                progress_callback=push_event,
            )

            push_event(
                {
                    "type": "progress",
                    "step": "grading",
                    "page": total_pages,
                    "total_pages": total_pages,
                    "percent": 82,
                    "message": (
                        f"All {total_pages} Page(s) Done ✔ → Grading {len(rubric.questions)} Question(s) "
                        f"at Strictness Level {lvl}/10..."
                    ),
                }
            )

            grader = ExamGrader(offline=offline, strictness_level=lvl)
            grader.semantic_scorer = get_semantic_scorer()
            report = grader.grade_exam(ocr_result, rubric, strictness_level=lvl)

            payload = _build_final_payload(session_id, ocr_result, report)
            push_event(
                {
                    "type": "complete",
                    "percent": 100,
                    "total_pages": total_pages,
                    "message": f"Evaluation Complete ✔ All {total_pages} Page(s) Graded!",
                    "data": payload,
                }
            )
        except Exception as exc:
            logger.exception("Error in streaming evaluation")
            push_event({"type": "error", "message": str(exc)})
        finally:
            event_queue.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def stream_generator():
        while True:
            item = event_queue.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")


@app.post("/api/regrade_question")
async def api_regrade_question(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Re-evaluate a single question when the teacher edits its extracted OCR digital text."""
    try:
        question = RubricQuestion.model_validate(payload["question"])
        student_answer = str(payload.get("student_answer", ""))
        strictness_level = int(payload.get("strictness_level", 5))
        offline = bool(payload.get("offline", False))
        page_numbers = payload.get("page_numbers", [1])

        grader = ExamGrader(offline=offline, strictness_level=strictness_level)
        grader.semantic_scorer = get_semantic_scorer()
        result = grader.grade_single_question(
            question=question,
            student_answer=student_answer,
            strictness_level=strictness_level,
            page_numbers=page_numbers,
        )
        return result.model_dump()
    except Exception as exc:
        logger.exception("Failed to regrade question")
        raise HTTPException(status_code=500, detail=f"Failed to re-grade question: {exc}")


@app.post("/api/finalize_marks")
async def api_finalize_marks(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Save and finalize teacher-verified OCR text and marks for the session."""
    session_id = str(payload.get("session_id") or "default")
    grading_report = payload.get("grading_report") or {}

    target_dir = UPLOADS_DIR / session_id
    target_dir.mkdir(parents=True, exist_ok=True)
    verified_path = target_dir / "verified_report.json"

    record = {
        "session_id": session_id,
        "finalized_at": datetime.now().isoformat(),
        "grading_report": grading_report,
    }
    verified_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    return {
        "status": "finalized",
        "session_id": session_id,
        "finalized_at": record["finalized_at"],
        "total_marks_awarded": grading_report.get("total_marks_awarded", 0.0),
        "total_max_marks": grading_report.get("total_max_marks", 0.0),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
