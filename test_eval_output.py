import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "exam_grader_mvp"))

from rubric_generator import RubricGenerator
from ocr_engine import OCREngine
from grader import ExamGrader

gen = RubricGenerator()
rubric = gen.generate(question_image_path="exam_grader_mvp/ui_uploads/q_a1fda99f/question.jpeg")

print(f"Rubric: {rubric.exam_title} | Printed Max Marks: {rubric.printed_max_marks}")
for q in rubric.questions:
    print(f"  {q.question_id} ({q.max_marks}m) [{q.question_type}]: {q.question_text[:70]}...")

ocr = OCREngine()
pages = [f"exam_grader_mvp/ui_uploads/8ba68234/page_{i}.jpg" for i in range(1, 10)]
ocr_res = ocr.process_pages(pages, online_fast=True)

print("\n--- OCR TRANSCRIPT SAMPLE ---")
for p in ocr_res.pages:
    print(f"Page {p.page_number} ({len(p.transcript)} chars): {p.transcript[:100]}...")

grader = ExamGrader()
rep = grader.grade_exam(ocr_res, rubric)

pct = round((rep.total_marks_awarded / max(rep.total_max_marks, 1)) * 100, 1)
print(f"\n==========================================")
print(f"TOTAL: {rep.total_marks_awarded} / {rep.total_max_marks} ({pct}%)")
print(f"==========================================")
for qr in rep.question_results:
    ans_snippet = (qr.student_answer or "").replace("\n", " ")[:60]
    print(f"\n[{qr.question_id}] {qr.marks_awarded} / {qr.max_marks} marks")
    print(f"  Attempted: {bool(qr.student_answer)}")
    print(f"  Student Ans: {ans_snippet}")
    print(f"  Feedback: {qr.student_feedback[:120]}")
    if qr.step_wise_breakdown:
        print("  Breakdown:")
        for step in qr.step_wise_breakdown:
            desc = getattr(step, "description", getattr(step, "step_description", ""))
            print(f"    - {desc[:45]}: {step.awarded_point_marks}/{step.max_point_marks} ({step.notes[:40]})")
