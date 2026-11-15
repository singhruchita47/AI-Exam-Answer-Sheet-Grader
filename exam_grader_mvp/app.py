"""FastAPI web UI for testing the Automated Exam Answer Sheet Checking Engine."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import GEMINI_API_KEY, TROCR_MODEL_NAME
from grader import ExamGrader, ExamRubric, SemanticScorer
from ocr_engine import OCREngine
from rubric_generator import RubricGenerator

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
        "has_gemini_key": bool(GEMINI_API_KEY),
        "trocr_model": TROCR_MODEL_NAME,
        "default_rubric": default_rubric,
        "photo_rubric": photo_rubric,
    }


@app.post("/api/generate_rubric")
async def api_generate_rubric(
    subject: str = Form("Physics"),
    total_marks: float = Form(10.0),
    question_text: str = Form(""),
    teacher_hints: str = Form(""),
    use_sample_question: bool = Form(False),
    offline: bool = Form(False),
    question_image: UploadFile | None = File(None),
) -> dict[str, Any]:
    """Generate structured grading rubric automatically from teacher's question photo or text."""
    image_path: Path | None = None
    question_images: list[Any] | None = None
    if use_sample_question:
        image_path = BASE_DIR / "sample_question_paper.jpg"
        if not image_path.exists():
            from create_dummy_question import generate_sample_question_paper

            generate_sample_question_paper(image_path)
    elif question_image is not None and getattr(question_image, "filename", None):
        session_id = uuid.uuid4().hex[:8]
        q_dir = UPLOADS_DIR / f"q_{session_id}"
        q_dir.mkdir(parents=True, exist_ok=True)
        file_ext = Path(question_image.filename).suffix.lower() or ".jpg"
        if file_ext == ".pdf":
            import numpy as np
            import pypdfium2 as pdfium

            pdf_path = q_dir / f"question{file_ext}"
            with pdf_path.open("wb") as f:
                shutil.copyfileobj(question_image.file, f)
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
                image_path = q_dir / "question_page_1.jpg"
                first_page.save(image_path, "JPEG", quality=88)

            # If scanned PDF (no embedded text): pass page images directly to Gemini VLM (saves 60s of CPU OCR)
            if not extracted_chunks:
                if not offline and GEMINI_API_KEY:
                    question_images = [doc[p_idx].render(scale=1.0).to_pil() for p_idx in range(max_p)]
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

            if not question_text and extracted_chunks:
                from rubric_generator import filter_hindi_keep_english

                question_text = filter_hindi_keep_english("\n\n".join(extracted_chunks))
        else:
            image_path = q_dir / f"question{file_ext}"
            with image_path.open("wb") as f:
                shutil.copyfileobj(question_image.file, f)

    try:
        generator = RubricGenerator(api_key=GEMINI_API_KEY)
        rubric = generator.generate(
            subject=subject,
            total_marks=total_marks,
            question_image_path=image_path,
            question_images=question_images,
            question_text=question_text,
            teacher_hints=teacher_hints,
            force_offline=offline,
        )
        return rubric.model_dump()
    except Exception as exc:
        logger.exception("Error generating rubric automatically")
        raise HTTPException(status_code=500, detail=f"Failed to generate rubric: {exc}")


@app.post("/api/grade")
async def api_grade(
    offline: bool = Form(False),
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

    page_paths: list[Path] = []

    if use_photosynthesis_sample:
        p_photo = BASE_DIR / "sample_photosynthesis_student.jpg"
        if not p_photo.exists():
            raise HTTPException(
                status_code=404, detail="sample_photosynthesis_student.jpg not found."
            )
        page_paths = [p_photo]
    elif use_sample_pages:
        p1 = BASE_DIR / "sample_page1.jpg"
        p2 = BASE_DIR / "sample_page2.jpg"
        if not p1.exists() or not p2.exists():
            from create_dummy_sheet import generate_sample_pages

            generate_sample_pages(BASE_DIR)
        page_paths = [p1, p2]
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

    try:
        engine = get_ocr_engine()
        online_fast = not offline and bool(GEMINI_API_KEY)
        ocr_result = engine.process_pages(
            page_paths, debug_dir=session_crops_dir, online_fast=online_fast
        )

        grader = ExamGrader(offline=offline)
        grader.semantic_scorer = get_semantic_scorer()
        report = grader.grade_exam(ocr_result, rubric)

        # Build client-friendly relative URLs
        seg_data = []
        for p in ocr_result.pages:
            rel_layout = p.layout_image_path.relative_to(BASE_DIR)
            seg_data.append(
                {
                    "page_number": p.page_number,
                    "source_path": str(p.source_path),
                    "skew_angle_deg": p.skew_angle_deg,
                    "text_lines_detected": len(p.text_lines),
                    "diagrams_detected": len(p.diagrams),
                    "layout_image_path": str(rel_layout),
                }
            )

        report_dump = report.model_dump()
        for q_res in report_dump.get("question_results", []):
            if q_res.get("diagram_path"):
                diag_p = Path(q_res["diagram_path"])
                if diag_p.is_absolute() and BASE_DIR in diag_p.parents:
                    q_res["diagram_path"] = str(diag_p.relative_to(BASE_DIR))

        return {
            "session_id": session_id,
            "segmentation": seg_data,
            "grading_report": report_dump,
        }
    except Exception as exc:
        logger.exception("Error evaluating exam sheets")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
