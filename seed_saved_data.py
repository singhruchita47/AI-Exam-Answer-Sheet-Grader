import json
import shutil
from pathlib import Path

BASE_DIR = Path("exam_grader_mvp").resolve()
SAVED_PAPERS = BASE_DIR / "saved_papers"
SAVED_SHEETS = BASE_DIR / "saved_sheets"

SAVED_PAPERS.mkdir(parents=True, exist_ok=True)
SAVED_SHEETS.mkdir(parents=True, exist_ok=True)

# 1. Seed Linux Paper (60m)
linux_p = SAVED_PAPERS / "paper_linux_60m"
linux_p.mkdir(parents=True, exist_ok=True)

src_q = BASE_DIR / "ui_uploads" / "q_a1fda99f" / "question.jpeg"
if src_q.exists():
    shutil.copy2(src_q, linux_p / "paper_file.jpeg")

# Generate or fetch rubric JSON
import sys
sys.path.insert(0, str(BASE_DIR))
from rubric_generator import RubricGenerator
gen = RubricGenerator()
if (linux_p / "paper_file.jpeg").exists():
    rubric = gen.generate(question_image_path=linux_p / "paper_file.jpeg")
    (linux_p / "rubric.json").write_text(json.dumps(rubric.model_dump(), indent=2), encoding="utf-8")
    meta_paper = {
        "id": "paper_linux_60m",
        "title": "LINUX Examination Paper (60 Marks)",
        "filename": "question.jpeg",
        "subject": rubric.detected_subject or "LINUX",
        "printed_max_marks": rubric.printed_max_marks or 60.0,
        "total_questions": len(rubric.questions),
        "created_at": "2026-11-17T23:14:00"
    }
    (linux_p / "meta.json").write_text(json.dumps(meta_paper, indent=2), encoding="utf-8")
    print("Seeded Linux paper:", meta_paper["title"])

# 2. Seed Photosynthesis Paper
photo_p = SAVED_PAPERS / "paper_photosynthesis"
photo_p.mkdir(parents=True, exist_ok=True)
sample_rubric_photo = BASE_DIR / "sample_rubric_photosynthesis.json"
if sample_rubric_photo.exists():
    shutil.copy2(sample_rubric_photo, photo_p / "rubric.json")
    meta_photo = {
        "id": "paper_photosynthesis",
        "title": "Biology - Photosynthesis Test (5 Marks)",
        "filename": "sample_question_paper.jpg",
        "subject": "Biology",
        "printed_max_marks": 5.0,
        "total_questions": 1,
        "created_at": "2026-11-17T23:14:00"
    }
    (photo_p / "meta.json").write_text(json.dumps(meta_photo, indent=2), encoding="utf-8")
    print("Seeded Photosynthesis paper")

# 3. Seed Student Copy: Kanak Patel
kanak_dir = SAVED_SHEETS / "sheet_kanak_patel"
kanak_dir.mkdir(parents=True, exist_ok=True)

src_student_dir = BASE_DIR / "ui_uploads" / "8ba68234"
if src_student_dir.exists():
    for f in src_student_dir.glob("page_*.jpg"):
        shutil.copy2(f, kanak_dir / f.name)
    pdf_files = list(src_student_dir.glob("*.pdf"))
    if pdf_files:
        shutil.copy2(pdf_files[0], kanak_dir / pdf_files[0].name)
    
    meta_kanak = {
        "id": "sheet_kanak_patel",
        "filename": pdf_files[0].name if pdf_files else "Stuti Gulati_23btm053_CSE E.pdf",
        "student_name": "Kanak Patel",
        "roll_no": "SGSU250489",
        "subject": "Linux / UNIX",
        "page_count": len(list(kanak_dir.glob("page_*.jpg"))),
        "created_at": "2026-11-17T23:14:00"
    }
    (kanak_dir / "meta.json").write_text(json.dumps(meta_kanak, indent=2), encoding="utf-8")
    print("Seeded Kanak Patel answer sheet (9 pages)")

# 4. Seed Photosynthesis Student Copy
photo_student_dir = SAVED_SHEETS / "sheet_photosynthesis_sample"
photo_student_dir.mkdir(parents=True, exist_ok=True)
photo_student_src = BASE_DIR / "sample_photosynthesis_student.jpg"
if photo_student_src.exists():
    shutil.copy2(photo_student_src, photo_student_dir / "page_1.jpg")
    meta_photo_student = {
        "id": "sheet_photosynthesis_sample",
        "filename": "sample_photosynthesis_student.jpg",
        "student_name": "Sample Student",
        "roll_no": "BIO-101",
        "subject": "Biology",
        "page_count": 1,
        "created_at": "2026-11-17T23:14:00"
    }
    (photo_student_dir / "meta.json").write_text(json.dumps(meta_photo_student, indent=2), encoding="utf-8")
    print("Seeded Photosynthesis student sheet")

print("All seeds complete.")
