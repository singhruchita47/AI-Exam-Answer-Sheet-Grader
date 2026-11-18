"""Generate and persist multi-format Exam Evaluation Reports (PDF, Printable HTML, CSV, JSON)."""

from __future__ import annotations

import csv
import html
import io
import json
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def _sanitize_pdf_text(text: Any) -> str:
    """Clean text for safe rendering inside Matplotlib vector PDFs (DejaVu Sans)."""
    s = str(text or "")
    # Replace common UI emojis/symbols with crisp PDF-safe text
    replacements = {
        "✔": "[OK]",
        "✓": "[OK]",
        "✅": "[OK]",
        "❌": "[X]",
        "⚠️": "[!]",
        "⚠": "[!]",
        "→": "->",
        "—": "-",
        "–": "-",
        "•": "*",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Strip Devanagari and non-BMP emojis that matplotlib fonts lack
    s = re.sub(r"[\u0900-\u097F\U00010000-\U0010FFFF]+", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _compute_grade_and_pct(awarded: float, max_marks: float) -> tuple[float, str]:
    pct = (awarded / max_marks * 100.0) if max_marks > 0 else 0.0
    if pct >= 90.0:
        grade = "A+"
    elif pct >= 80.0:
        grade = "A"
    elif pct >= 65.0:
        grade = "B"
    elif pct >= 50.0:
        grade = "C"
    elif pct >= 35.0:
        grade = "D"
    else:
        grade = "F"
    return round(pct, 1), grade


def _safe_slug(text: str, fallback: str = "report") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")
    return cleaned[:40] if cleaned else fallback


def parse_student_meta_from_filename(filename: str) -> tuple[str, str]:
    """Infer Student Name and Roll No from filename like 'Stuti Gulati_23btm053_CSE E.pdf'."""
    stem = Path(filename or "").stem
    if not stem:
        return "", ""
    # Look for roll-number-like alphanumeric token containing digits
    tokens = re.split(r"[_\-]+", stem)
    roll = ""
    name_parts: list[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if not roll and re.search(r"\d{2,}", t) and len(t) <= 20:
            roll = t.upper()
        elif not roll and re.match(r"^(?:CSE|IT|BCA|MCA|ECE|ME|CE)\b", t, re.IGNORECASE):
            continue
        elif roll and re.match(r"^(?:CSE|IT|BCA|MCA|ECE|ME|CE|SEC|[A-Z])$", t, re.IGNORECASE):
            continue
        else:
            name_parts.append(t)
    student_name = " ".join(name_parts).strip()
    return student_name, roll


def build_report_pdf_bytes(report_payload: dict[str, Any], report_mode: str = "detailed") -> bytes:
    """Render a crisp, vector PDF scorecard from a verified grading report (detailed or short 1-page)."""
    grading = report_payload.get("grading_report") or {}
    q_results: list[dict[str, Any]] = grading.get("question_results") or []

    exam_title = _sanitize_pdf_text(grading.get("exam_title") or "Exam Assessment Report")
    student_name = _sanitize_pdf_text(report_payload.get("student_name") or "Not Specified")
    roll_no = _sanitize_pdf_text(report_payload.get("roll_no") or "-")
    source_file = _sanitize_pdf_text(report_payload.get("source_filename") or "-")
    teacher_remarks = _sanitize_pdf_text(report_payload.get("overall_remarks") or "")
    saved_at = _sanitize_pdf_text(report_payload.get("saved_at") or datetime.now().strftime("%Y-%m-%d %H:%M"))
    strictness_level = int(grading.get("strictness_level") or report_payload.get("strictness_level") or 5)
    mode_raw = grading.get("mode") or "online_gemini"
    mode_label = "Multimodal Vision + Rubric Grader" if mode_raw == "online_gemini" else "Local OCR + Semantic Grader"

    total_awarded = float(grading.get("total_marks_awarded") or 0.0)
    total_max = float(grading.get("total_max_marks") or 0.0)
    pct, grade = _compute_grade_and_pct(total_awarded, total_max)
    strict_tier = "Lenient" if strictness_level <= 3 else ("Balanced" if strictness_level <= 7 else "Strict")
    is_short = (report_mode == "short")

    # Save into in-memory buffer
    buf = io.BytesIO()
    plt.rcParams["font.family"] = "DejaVu Sans"

    def _new_a4_page() -> tuple[plt.Figure, Any]:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 141.4)
        ax.axis("off")
        return fig, ax

    def _add_footer(ax: Any, page_idx: int) -> None:
        ax.plot([5, 95], [5.8, 5.8], color="#cbd5e1", linewidth=0.7)
        ax.text(
            5,
            3.6,
            f"Automated Exam Answer Sheet Grader  |  Verified Report  |  {saved_at}",
            fontsize=7.5,
            color="#64748b",
        )
        ax.text(95, 3.6, f"Page {page_idx}", fontsize=7.5, color="#64748b", ha="right")

    with PdfPages(buf) as pdf:
        page_num = 1
        fig, ax = _new_a4_page()

        # Header Card
        header_box = patches.FancyBboxPatch(
            (5, 122.5), 90, 14.2,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            facecolor="#eef2ff",
            edgecolor="#c7d2fe",
            linewidth=1.2,
        )
        ax.add_patch(header_box)
        ax.text(7.2, 133.2, "OFFICIAL EXAM EVALUATION REPORT", fontsize=8.5, fontweight="bold", color="#4338ca")
        ax.text(7.2, 128.4, exam_title[:65], fontsize=15, fontweight="bold", color="#0f172a")
        ax.text(
            7.2,
            124.5,
            f"Pipeline: {mode_label}   |   Strictness: {strictness_level}/10 ({strict_tier})",
            fontsize=8.2,
            color="#475569",
        )

        # Score Badge on Right of Header
        score_box = patches.FancyBboxPatch(
            (72.5, 124.2), 20.5, 10.6,
            boxstyle="round,pad=0.4,rounding_size=1.0",
            facecolor="#059669",
            edgecolor="#047857",
            linewidth=1.0,
        )
        ax.add_patch(score_box)
        ax.text(82.75, 132.2, "VERIFIED SCORE", fontsize=7.2, fontweight="bold", color="#d1fae5", ha="center")
        ax.text(
            82.75,
            128.3,
            f"{total_awarded:.2f} / {total_max:.2f}",
            fontsize=11.5,
            fontweight="bold",
            color="#ffffff",
            ha="center",
        )
        ax.text(82.75, 125.1, f"{pct:.1f}%  |  Grade: {grade}", fontsize=8.5, fontweight="bold", color="#ecfdf5", ha="center")

        # Student Metadata Strip
        meta_box = patches.FancyBboxPatch(
            (5, 111.5), 90, 8.8,
            boxstyle="round,pad=0.4,rounding_size=0.8",
            facecolor="#f8fafc",
            edgecolor="#e2e8f0",
            linewidth=0.9,
        )
        ax.add_patch(meta_box)
        ax.text(7.2, 117.6, f"Student Name:  {student_name}", fontsize=9, fontweight="bold", color="#1e293b")
        ax.text(55.0, 117.6, f"Roll No / ID:  {roll_no}", fontsize=9, fontweight="bold", color="#1e293b")
        ax.text(7.2, 113.5, f"Answer Sheet File:  {source_file[:48]}", fontsize=8.2, color="#475569")
        ax.text(55.0, 113.5, f"Evaluated On:  {saved_at[:19]}", fontsize=8.2, color="#475569")

        y = 107.2
        if teacher_remarks:
            rem_lines = textwrap.wrap(f"Teacher / Examiner Remarks: {teacher_remarks}", width=92)[:3]
            rem_h = 3.0 + len(rem_lines) * 2.7
            rem_box = patches.FancyBboxPatch(
                (5, y - rem_h + 1.5), 90, rem_h,
                boxstyle="round,pad=0.3,rounding_size=0.7",
                facecolor="#fffbeb",
                edgecolor="#fde68a",
                linewidth=0.8,
            )
            ax.add_patch(rem_box)
            ry = y - 1.2
            for rl in rem_lines:
                ax.text(7.0, ry, rl, fontsize=8.2, color="#78350f")
                ry -= 2.6
            y -= rem_h + 3.0

        # Marks Summary Table
        ax.text(5, y, "Question-Wise Score Summary", fontsize=11, fontweight="bold", color="#0f172a")
        y -= 2.2

        # Table Header
        hdr_rect = patches.Rectangle((5, y - 4.2), 90, 4.2, facecolor="#1e293b", edgecolor="#1e293b")
        ax.add_patch(hdr_rect)
        if is_short:
            ax.text(7.0, y - 2.8, "Q. ID", fontsize=8.2, fontweight="bold", color="#ffffff")
            ax.text(18.0, y - 2.8, "Question Statement", fontsize=8.2, fontweight="bold", color="#ffffff")
            ax.text(70.0, y - 2.8, "Max", fontsize=8.2, fontweight="bold", color="#ffffff", ha="right")
            ax.text(82.0, y - 2.8, "Awarded", fontsize=8.2, fontweight="bold", color="#ffffff", ha="right")
            ax.text(93.5, y - 2.8, "% Score", fontsize=8.2, fontweight="bold", color="#ffffff", ha="right")
        else:
            ax.text(7.0, y - 2.8, "Q. ID", fontsize=8.2, fontweight="bold", color="#ffffff")
            ax.text(21.0, y - 2.8, "Question Type", fontsize=8.2, fontweight="bold", color="#ffffff")
            ax.text(42.0, y - 2.8, "Page(s)", fontsize=8.2, fontweight="bold", color="#ffffff")
            ax.text(56.0, y - 2.8, "Key Points Hit", fontsize=8.2, fontweight="bold", color="#ffffff")
            ax.text(75.0, y - 2.8, "Awarded", fontsize=8.2, fontweight="bold", color="#ffffff", ha="right")
            ax.text(85.0, y - 2.8, "Max", fontsize=8.2, fontweight="bold", color="#ffffff", ha="right")
            ax.text(93.5, y - 2.8, "%", fontsize=8.2, fontweight="bold", color="#ffffff", ha="right")
        y -= 4.2

        for idx_q, q in enumerate(q_results):
            if y < 16.0:
                _add_footer(ax, page_num)
                pdf.savefig(fig)
                plt.close(fig)
                page_num += 1
                fig, ax = _new_a4_page()
                y = 133.0
                ax.text(5, y, "Question-Wise Score Summary (Continued)", fontsize=10.5, fontweight="bold", color="#0f172a")
                y -= 4.0

            q_id = _sanitize_pdf_text(q.get("question_id") or f"Q{idx_q+1}")
            q_type = _sanitize_pdf_text((q.get("question_type") or "short_answer").replace("_", " ").title())
            is_att = q.get("is_attempted", bool(str(q.get("student_answer") or "").strip()))
            raw_pages = q.get("page_numbers") or []
            p_nums = ", ".join(str(p) for p in raw_pages) if (is_att and raw_pages) else ("Not Attempted" if not is_att else "1")
            steps = q.get("step_wise_breakdown") or []
            hits = sum(1 for s in steps if s.get("status") == "Hit")
            kp_stat = f"{hits} / {len(steps)}" if steps else "-"
            aw = float(q.get("marks_awarded") or 0.0)
            mx = float(q.get("max_marks") or 0.0)
            q_pct = (aw / mx * 100.0) if mx > 0 else 0.0

            row_bg = "#ffffff" if idx_q % 2 == 0 else "#f8fafc"
            row_rect = patches.Rectangle((5, y - 3.9), 90, 3.9, facecolor=row_bg, edgecolor="#e2e8f0", linewidth=0.5)
            ax.add_patch(row_rect)

            if is_short:
                q_text = _sanitize_pdf_text(q.get("question_text") or f"Question {q_id}").replace("Question:", "").strip()
                if len(q_text) > 52:
                    q_text = q_text[:49] + "..."
                ax.text(7.0, y - 2.6, q_id, fontsize=8.2, fontweight="bold", color="#312e81")
                ax.text(18.0, y - 2.6, q_text, fontsize=7.8, color="#1e293b")
                ax.text(70.0, y - 2.6, f"{mx:.2f}", fontsize=8.0, color="#475569", ha="right")
                ax.text(82.0, y - 2.6, f"{aw:.2f}", fontsize=8.2, fontweight="bold", color="#047857", ha="right")
                ax.text(93.5, y - 2.6, f"{q_pct:.0f}%", fontsize=8.0, color="#334155", ha="right")
            else:
                ax.text(7.0, y - 2.6, q_id, fontsize=8.2, fontweight="bold", color="#312e81")
                ax.text(21.0, y - 2.6, q_type[:18], fontsize=8.0, color="#334155")
                ax.text(42.0, y - 2.6, (f"P {p_nums}" if p_nums != "Not Attempted" else "-")[:12], fontsize=8.0, color="#475569")
                ax.text(56.0, y - 2.6, kp_stat, fontsize=8.0, color="#475569")
                ax.text(75.0, y - 2.6, f"{aw:.2f}", fontsize=8.2, fontweight="bold", color="#047857", ha="right")
                ax.text(85.0, y - 2.6, f"{mx:.2f}", fontsize=8.0, color="#475569", ha="right")
                ax.text(93.5, y - 2.6, f"{q_pct:.0f}%", fontsize=8.0, color="#334155", ha="right")
            y -= 3.9

        # Total Summary Row
        tot_rect = patches.Rectangle((5, y - 4.4), 90, 4.4, facecolor="#ecfdf5", edgecolor="#059669", linewidth=0.9)
        ax.add_patch(tot_rect)
        ax.text(7.0, y - 2.9, "TOTAL VERIFIED SCORE", fontsize=8.6, fontweight="bold", color="#065f46")
        if is_short:
            ax.text(70.0, y - 2.9, f"{total_max:.2f}", fontsize=8.6, fontweight="bold", color="#065f46", ha="right")
            ax.text(82.0, y - 2.9, f"{total_awarded:.2f}", fontsize=8.8, fontweight="bold", color="#065f46", ha="right")
            ax.text(93.5, y - 2.9, f"{pct:.1f}%", fontsize=8.6, fontweight="bold", color="#065f46", ha="right")
        else:
            ax.text(75.0, y - 2.9, f"{total_awarded:.2f}", fontsize=8.8, fontweight="bold", color="#065f46", ha="right")
            ax.text(85.0, y - 2.9, f"{total_max:.2f}", fontsize=8.6, fontweight="bold", color="#065f46", ha="right")
            ax.text(93.5, y - 2.9, f"{pct:.1f}%", fontsize=8.6, fontweight="bold", color="#065f46", ha="right")
        y -= 8.0

        if is_short:
            # Short format single-page completion with verification sign-off
            y = max(y, 22.0)
            cert_box = patches.FancyBboxPatch(
                (5, y - 11.5), 90, 11.5,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                facecolor="#f0fdf4",
                edgecolor="#86efac",
                linewidth=0.8,
            )
            ax.add_patch(cert_box)
            ax.text(7.2, y - 3.0, "EXAMINER / EVALUATOR VERIFICATION CERTIFICATE", fontsize=8.2, fontweight="bold", color="#166534")
            ax.text(7.2, y - 6.2, f"Candidate {student_name} ({roll_no}) scored {total_awarded:.2f} / {total_max:.2f} Marks ({pct:.1f}%, Grade {grade}).", fontsize=8.0, color="#14532d")
            ax.text(7.2, y - 9.2, "Marks evaluated accurately with Multimodal Vision OCR and Rubric Grader.", fontsize=7.5, color="#15803d")
            ax.plot([74, 93], [y - 6.2, y - 6.2], color="#94a3b8", linewidth=0.8)
            ax.text(83.5, y - 8.5, "Authorized Signature", fontsize=7.2, color="#64748b", ha="center")

            _add_footer(ax, 1)
            pdf.savefig(fig)
            plt.close(fig)
            return buf.getvalue()

        # Detailed Question-by-Question Breakdown
        if y < 40.0:
            _add_footer(ax, page_num)
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1
            fig, ax = _new_a4_page()
            y = 133.0

        ax.text(5, y, "Detailed Question-Wise Evaluation & Extracted Answers", fontsize=11, fontweight="bold", color="#0f172a")
        y -= 4.0

        for idx_q, q in enumerate(q_results):
            q_id = _sanitize_pdf_text(q.get("question_id") or f"Q{idx_q+1}")
            q_type = _sanitize_pdf_text((q.get("question_type") or "short_answer").replace("_", " ").title())
            is_att = q.get("is_attempted", bool(str(q.get("student_answer") or "").strip()))
            raw_pages = q.get("page_numbers") or []
            p_nums = ", ".join(str(p) for p in raw_pages) if (is_att and raw_pages) else ("Not Attempted" if not is_att else "1")
            aw = float(q.get("marks_awarded") or 0.0)
            mx = float(q.get("max_marks") or 0.0)
            ans_txt = _sanitize_pdf_text(q.get("student_answer") or ("Student did not attempt this question." if not is_att else "[No answer text recorded]"))
            fb_txt = _sanitize_pdf_text(q.get("student_feedback") or "")
            steps = q.get("step_wise_breakdown") or []

            q_raw_text = _sanitize_pdf_text(q.get("question_text") or "").replace("Question:", "").strip()
            q_lines = textwrap.wrap(f"Question: {q_raw_text}", width=92)[:2] if q_raw_text else []

            ans_lines = textwrap.wrap(ans_txt, width=92)[:7]
            if len(textwrap.wrap(ans_txt, width=92)) > 7:
                ans_lines[-1] = ans_lines[-1][:84] + "..."

            step_lines: list[tuple[str, str]] = []
            for s in steps:
                st = s.get("status") or "Partial"
                tag = "[Hit]" if st == "Hit" else ("[Partial]" if st == "Partial" else "[Missed]")
                desc = _sanitize_pdf_text(s.get("description") or "")
                s_aw = float(s.get("awarded_point_marks") or 0.0)
                s_mx = float(s.get("max_point_marks") or 0.0)
                line_s = f"{tag} ({s_aw:.2f}/{s_mx:.2f}): {desc}"
                wrapped_s = textwrap.wrap(line_s, width=90)[:2]
                for ws in wrapped_s:
                    step_lines.append((st, ws))

            if not fb_txt:
                if not is_att:
                    fb_txt = "Student did not attempt this question (0 Marks)."
                elif aw >= mx:
                    fb_txt = "Full marks awarded! Comprehensive, accurate answer addressing all required concepts."
                elif aw > 0:
                    fb_txt = f"Partial marks awarded ({aw:.2f} / {mx:.2f}). Core concepts covered."
                else:
                    fb_txt = "Answer did not match key concepts or answer key criteria."

            fb_lines = textwrap.wrap(f"Feedback: {fb_txt}", width=92)[:3] if fb_txt else []

            block_h = (
                7.2
                + (len(q_lines) * 2.5 + (1.6 if q_lines else 0.0))
                + (len(ans_lines) * 2.5 + 3.2)
                + (len(step_lines) * 2.4 + (2.4 if step_lines else 0.0))
                + (len(fb_lines) * 2.5 + (2.0 if fb_lines else 0.0))
                + 2.0
            )

            if y - block_h < 10.5:
                _add_footer(ax, page_num)
                pdf.savefig(fig)
                plt.close(fig)
                page_num += 1
                fig, ax = _new_a4_page()
                y = 133.0

            card = patches.FancyBboxPatch(
                (5, y - block_h + 1.2), 90, block_h - 1.2,
                boxstyle="round,pad=0.35,rounding_size=0.8",
                facecolor="#ffffff",
                edgecolor="#cbd5e1",
                linewidth=0.85,
            )
            ax.add_patch(card)

            # Card Title Bar
            title_bar = patches.Rectangle((5.2, y - 4.5), 89.6, 4.4, facecolor="#f1f5f9", edgecolor="#e2e8f0", linewidth=0.5)
            ax.add_patch(title_bar)
            page_tag = "Not Attempted" if p_nums == "Not Attempted" else f"Page {p_nums}"
            ax.text(7.0, y - 3.0, f"{q_id}  ({q_type})  -  {page_tag}", fontsize=9.0, fontweight="bold", color="#1e1b4b")
            score_color = "#047857" if (mx > 0 and aw / mx >= 0.6) else ("#b45309" if aw > 0 else "#b91c1c")
            ax.text(93.2, y - 3.0, f"Marks: {aw:.2f} / {mx:.2f}", fontsize=9.0, fontweight="bold", color=score_color, ha="right")

            cy = y - 7.3
            if q_lines:
                for ql in q_lines:
                    ax.text(7.0, cy, ql, fontsize=7.8, fontweight="bold", color="#312e81")
                    cy -= 2.4
                cy -= 0.6

            ax.text(7.0, cy, "Extracted Handwritten Answer:", fontsize=7.6, fontweight="bold", color="#475569")
            cy -= 2.5
            for al in ans_lines:
                ax.text(7.5, cy, al, fontsize=7.8, color="#0f172a")
                cy -= 2.4

            if step_lines:
                cy -= 0.8
                ax.text(7.0, cy, "Rubric Breakdown:", fontsize=7.6, fontweight="bold", color="#475569")
                cy -= 2.3
                for st, sl in step_lines:
                    col = "#047857" if st == "Hit" else ("#a16207" if st == "Partial" else "#b91c1c")
                    ax.text(7.5, cy, sl, fontsize=7.5, color=col)
                    cy -= 2.35

            if fb_lines:
                cy -= 0.7
                for fl in fb_lines:
                    ax.text(7.0, cy, fl, fontsize=7.6, fontweight="bold", color="#3730a3")
                    cy -= 2.4

            y -= block_h + 2.8

        # Sign-off block on final page
        if y < 18.0:
            _add_footer(ax, page_num)
            pdf.savefig(fig)
            plt.close(fig)
            page_num += 1
            fig, ax = _new_a4_page()
            y = 128.0

        ax.plot([5, 38], [y - 6.0, y - 6.0], color="#475569", linewidth=0.8)
        ax.text(5, y - 8.6, "Examiner / Teacher Signature", fontsize=8.0, fontweight="bold", color="#334155")
        ax.plot([62, 95], [y - 6.0, y - 6.0], color="#475569", linewidth=0.8)
        ax.text(95, y - 8.6, f"Date: {saved_at[:10]}", fontsize=8.0, color="#334155", ha="right")

        _add_footer(ax, page_num)
        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.read()


def build_report_csv_text(report_payload: dict[str, Any]) -> str:
    """Build an Excel-friendly CSV scorecard string."""
    grading = report_payload.get("grading_report") or {}
    q_results = grading.get("question_results") or []
    student_name = report_payload.get("student_name") or "Not Specified"
    roll_no = report_payload.get("roll_no") or "-"
    exam_title = grading.get("exam_title") or "Exam Assessment"
    total_awarded = float(grading.get("total_marks_awarded") or 0.0)
    total_max = float(grading.get("total_max_marks") or 0.0)
    pct, grade = _compute_grade_and_pct(total_awarded, total_max)

    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Exam Title", exam_title])
    writer.writerow(["Student Name", student_name])
    writer.writerow(["Roll No / ID", roll_no])
    writer.writerow(["Saved At", report_payload.get("saved_at") or datetime.now().isoformat()])
    writer.writerow(["Strictness Level", f"{grading.get('strictness_level', 5)}/10"])
    writer.writerow(["Total Marks", f"{total_awarded:.2f} / {total_max:.2f}", f"{pct:.1f}%", f"Grade {grade}"])
    writer.writerow([])
    writer.writerow(
        ["Question ID", "Type", "Pages", "Marks Awarded", "Max Marks", "Extracted Student Answer", "Feedback"]
    )
    for q in q_results:
        is_att = q.get("is_attempted", bool(str(q.get("student_answer") or "").strip()))
        raw_pages = q.get("page_numbers") or []
        page_str = ", ".join(str(p) for p in raw_pages) if (is_att and raw_pages) else ("Not Attempted" if not is_att else "1")
        writer.writerow(
            [
                q.get("question_id", ""),
                q.get("question_type", ""),
                page_str,
                f"{float(q.get('marks_awarded') or 0.0):.2f}",
                f"{float(q.get('max_marks') or 0.0):.2f}",
                q.get("student_answer", "") or ("Not Attempted" if not is_att else ""),
                q.get("student_feedback", ""),
            ]
        )
    return out.getvalue()


def build_report_html(report_payload: dict[str, Any]) -> str:
    """Generate a self-contained printable light-theme HTML scorecard."""
    grading = report_payload.get("grading_report") or {}
    q_results = grading.get("question_results") or []
    exam_title = html.escape(str(grading.get("exam_title") or "Exam Evaluation Report"))
    student_name = html.escape(str(report_payload.get("student_name") or "Not Specified"))
    roll_no = html.escape(str(report_payload.get("roll_no") or "-"))
    source_file = html.escape(str(report_payload.get("source_filename") or "-"))
    remarks = html.escape(str(report_payload.get("overall_remarks") or ""))
    saved_at = html.escape(str(report_payload.get("saved_at") or datetime.now().strftime("%Y-%m-%d %H:%M")))
    strictness_level = int(grading.get("strictness_level") or report_payload.get("strictness_level") or 5)

    total_awarded = float(grading.get("total_marks_awarded") or 0.0)
    total_max = float(grading.get("total_max_marks") or 0.0)
    pct, grade = _compute_grade_and_pct(total_awarded, total_max)

    rows_html = []
    details_html = []
    for q in q_results:
        qid = html.escape(str(q.get("question_id") or ""))
        qtype = html.escape(str((q.get("question_type") or "").replace("_", " ").title()))
        is_att = q.get("is_attempted", bool(str(q.get("student_answer") or "").strip()))
        raw_pages = q.get("page_numbers") or []
        pages = html.escape(", ".join(str(p) for p in raw_pages)) if (is_att and raw_pages) else ("Not Attempted" if not is_att else "1")
        aw = float(q.get("marks_awarded") or 0.0)
        mx = float(q.get("max_marks") or 0.0)
        q_pct = (aw / mx * 100.0) if mx > 0 else 0.0
        ans = html.escape(str(q.get("student_answer") or ("Student did not attempt this question." if not is_att else "[No answer recorded]")))
        fb = html.escape(str(q.get("student_feedback") or ""))
        q_text = html.escape(str(q.get("question_text") or f"Question {qid}").replace("Question:", "").strip())
        if not fb:
            if not is_att:
                fb = "Student did not attempt this question (0 Marks awarded)."
            elif aw >= mx:
                fb = "Full marks awarded! Comprehensive, accurate answer addressing all required technical concepts."
            elif aw > 0:
                fb = f"Partial marks awarded ({aw:.2f} / {mx:.2f}). Found matching concepts; review minor secondary points."
            else:
                fb = "Answer does not match expected model answer key criteria."

        page_cell = "-" if pages == "Not Attempted" else f"P {pages}"
        page_head = "Not Attempted" if pages == "Not Attempted" else f"Page {pages}"
        rows_html.append(
            f"<tr><td><b>{qid}</b></td><td>{qtype}</td><td>{page_cell}</td>"
            f"<td class='num win'>{aw:.2f}</td><td class='num'>{mx:.2f}</td><td class='num'>{q_pct:.0f}%</td></tr>"
        )
        steps_li = "".join(
            f"<li><span class='st {html.escape(str(s.get('status',''))).lower()}'>{html.escape(str(s.get('status','')))} "
            f"({float(s.get('awarded_point_marks') or 0):.2f}/{float(s.get('max_point_marks') or 0):.2f})</span> "
            f"{html.escape(str(s.get('description') or ''))}</li>"
            for s in (q.get("step_wise_breakdown") or [])
        )
        details_html.append(
            f"<div class='qcard'><div class='qhead'><span><b>{qid}</b> ({qtype}) &bull; {page_head}</span>"
            f"<span class='qmarks'>Marks: {aw:.2f} / {mx:.2f} ({q_pct:.0f}%)</span></div>"
            f"<div class='qbody'>"
            f"<div class='lbl'>Question:</div><p class='qprompt'>{q_text}</p>"
            f"<div class='lbl'>Extracted Handwritten Answer:</div><p class='ans'>{ans}</p>"
            f"{('<ul class=\"steps\">' + steps_li + '</ul>') if steps_li else ''}"
            f"<div class='fb'><b>Examiner AI Feedback:</b> {fb}</div>"
            f"</div></div>"
        )

    remarks_html = f"<div class='remarks'><b>Examiner Remarks:</b> {remarks}</div>" if remarks else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>{exam_title} - {student_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }}
  .sheet {{ max-width: 900px; margin: 0 auto; background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 28px; box-shadow: 0 4px 16px rgba(15,23,42,.05); }}
  .topbar {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; border-bottom: 2px solid #e0e7ff; padding-bottom: 18px; margin-bottom: 18px; }}
  .badge {{ font-size: 11px; font-weight: 700; color: #4338ca; letter-spacing: .06em; text-transform: uppercase; }}
  h1 {{ margin: 4px 0; font-size: 22px; }}
  .sub {{ font-size: 12px; color: #475569; }}
  .scorecard {{ background: #059669; color: #fff; border-radius: 12px; padding: 12px 18px; text-align: center; min-width: 170px; }}
  .scorecard .big {{ font-size: 20px; font-weight: 800; }}
  .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 16px; font-size: 13px; margin-bottom: 16px; }}
  .remarks {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 10px 14px; font-size: 13px; color: #78350f; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 22px; font-size: 13px; }}
  th {{ background: #1e293b; color: #fff; padding: 9px 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
  .num {{ text-align: right; }}
  .win {{ color: #047857; font-weight: 700; }}
  tr.tot td {{ background: #ecfdf5; font-weight: 800; color: #065f46; border-top: 2px solid #059669; }}
  .qcard {{ border: 1px solid #cbd5e1; border-radius: 10px; margin-bottom: 14px; overflow: hidden; break-inside: avoid; }}
  .qhead {{ background: #f1f5f9; padding: 9px 14px; display: flex; justify-content: space-between; font-size: 13px; border-bottom: 1px solid #e2e8f0; }}
  .qmarks {{ font-weight: 800; color: #047857; }}
  .qbody {{ padding: 12px 14px; font-size: 12.5px; }}
  .lbl {{ font-weight: 700; color: #475569; margin-bottom: 4px; font-size: 11.5px; text-transform: uppercase; }}
  .qprompt {{ background: #eef2ff; border-left: 3px solid #6366f1; padding: 8px 12px; border-radius: 6px; font-weight: 600; margin: 0 0 10px; font-size: 13px; line-height: 1.5; color: #1e1b4b; }}
  .ans {{ margin: 0 0 10px; white-space: pre-wrap; line-height: 1.5; }}
  .steps {{ margin: 0 0 8px; padding-left: 18px; }}
  .steps li {{ margin-bottom: 4px; }}
  .st {{ font-weight: 700; font-size: 11px; padding: 1px 6px; border-radius: 6px; margin-right: 4px; }}
  .st.hit {{ background: #ecfdf5; color: #047857; }}
  .st.partial {{ background: #fefce8; color: #a16207; }}
  .st.missed {{ background: #fef2f2; color: #b91c1c; }}
  .fb {{ background: #eef2ff; color: #312e81; padding: 8px 10px; border-radius: 8px; font-size: 12px; }}
  .actions {{ display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 14px; }}
  .btn {{ background: #4f46e5; color: #fff; border: none; padding: 9px 16px; border-radius: 8px; font-weight: 700; cursor: pointer; font-size: 13px; }}
  @media print {{ .actions {{ display: none; }} body {{ background: #fff; padding: 0; }} .sheet {{ border: none; box-shadow: none; }} }}
</style>
</head>
<body>
<div class="sheet">
  <div class="actions"><button class="btn" onclick="window.print()">Print / Save as PDF</button></div>
  <div class="topbar">
    <div>
      <div class="badge">Official Exam Evaluation Report</div>
      <h1>{exam_title}</h1>
      <div class="sub">Strictness: {strictness_level}/10 &bull; Saved: {saved_at}</div>
    </div>
    <div class="scorecard">
      <div style="font-size:11px;opacity:.9;font-weight:700;">VERIFIED SCORE</div>
      <div class="big">{total_awarded:.2f} / {total_max:.2f}</div>
      <div style="font-size:12px;font-weight:700;">{pct:.1f}% &bull; Grade {grade}</div>
    </div>
  </div>
  <div class="meta">
    <div><b>Student Name:</b> {student_name}</div>
    <div><b>Roll No / ID:</b> {roll_no}</div>
    <div><b>Answer Sheet File:</b> {source_file}</div>
    <div><b>Questions Graded:</b> {len(q_results)}</div>
  </div>
  {remarks_html}
  <table>
    <thead><tr><th>Question</th><th>Type</th><th>Page(s)</th><th class="num">Awarded</th><th class="num">Max Marks</th><th class="num">%</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
      <tr class="tot"><td colspan="3">TOTAL VERIFIED SCORE</td><td class="num">{total_awarded:.2f}</td><td class="num">{total_max:.2f}</td><td class="num">{pct:.1f}%</td></tr>
    </tbody>
  </table>
  {''.join(details_html)}
</div>
</body>
</html>"""


def build_short_report_html(report_payload: dict[str, Any]) -> str:
    """Generate an ultra-clean 1-page printable summary scorecard with candidate details, questions, and marks."""
    grading = report_payload.get("grading_report") or {}
    q_results = grading.get("question_results") or []
    exam_title = html.escape(str(grading.get("exam_title") or "Exam Evaluation Report"))
    student_name = html.escape(str(report_payload.get("student_name") or "Not Specified"))
    roll_no = html.escape(str(report_payload.get("roll_no") or "-"))
    remarks = html.escape(str(report_payload.get("overall_remarks") or ""))
    saved_at = html.escape(str(report_payload.get("saved_at") or datetime.now().strftime("%Y-%m-%d %H:%M")))
    strictness_level = int(grading.get("strictness_level") or report_payload.get("strictness_level") or 5)

    total_awarded = float(grading.get("total_marks_awarded") or 0.0)
    total_max = float(grading.get("total_max_marks") or 0.0)
    pct, grade = _compute_grade_and_pct(total_awarded, total_max)

    rows_html = []
    for q in q_results:
        qid = html.escape(str(q.get("question_id") or ""))
        is_att = q.get("is_attempted", bool(str(q.get("student_answer") or "").strip()))
        aw = float(q.get("marks_awarded") or 0.0)
        mx = float(q.get("max_marks") or 0.0)
        q_pct = (aw / mx * 100.0) if mx > 0 else 0.0
        q_text = html.escape(str(q.get("question_text") or f"Question {qid}").replace("Question:", "").strip())

        status_label = "Full Marks" if (aw >= mx and is_att) else ("Partial" if (aw > 0 and is_att) else ("Incorrect" if is_att else "Not Attempted"))
        status_class = "st-full" if (aw >= mx and is_att) else ("st-part" if (aw > 0 and is_att) else "st-none")

        rows_html.append(
            f"<tr>"
            f"<td class='qid-cell'><b>{qid}</b></td>"
            f"<td class='qprompt-cell'>{q_text}</td>"
            f"<td class='num'>{mx:.1f}</td>"
            f"<td class='num win'><b>{aw:.1f}</b></td>"
            f"<td class='num-status'><span class='{status_class}'>{status_label} ({q_pct:.0f}%)</span></td>"
            f"</tr>"
        )

    remarks_html = f"<div class='remarks'><b>Examiner Remarks:</b> {remarks}</div>" if remarks else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>{exam_title} - Short Scorecard - {student_name}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 24px; }}
  .sheet {{ max-width: 860px; margin: 0 auto; background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 28px; box-shadow: 0 4px 16px rgba(15,23,42,.05); }}
  .topbar {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; border-bottom: 2px solid #e0e7ff; padding-bottom: 16px; margin-bottom: 16px; }}
  .badge {{ font-size: 11px; font-weight: 700; color: #047857; letter-spacing: .06em; text-transform: uppercase; }}
  h1 {{ margin: 4px 0; font-size: 20px; color: #0f172a; }}
  .sub {{ font-size: 11.5px; color: #475569; }}
  .scorecard {{ background: linear-gradient(135deg, #059669 0%, #047857 100%); color: #fff; border-radius: 12px; padding: 12px 20px; text-align: center; min-width: 175px; }}
  .scorecard .big {{ font-size: 22px; font-weight: 800; }}
  .meta {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; font-size: 12.5px; margin-bottom: 16px; }}
  .meta-item b {{ display: block; font-size: 10px; text-transform: uppercase; color: #64748b; margin-bottom: 2px; }}
  .meta-item span {{ font-weight: 700; color: #0f172a; font-size: 13px; }}
  .remarks {{ background: #fffbeb; border: 1px solid #fde68a; border-radius: 10px; padding: 10px 14px; font-size: 12.5px; color: #78350f; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 18px; font-size: 12.5px; }}
  th {{ background: #0f172a; color: #fff; padding: 9px 10px; text-align: left; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; }}
  .qid-cell {{ width: 70px; color: #312e81; font-weight: 800; }}
  .qprompt-cell {{ font-weight: 600; color: #1e293b; line-height: 1.4; }}
  .num {{ text-align: right; width: 65px; font-family: monospace; }}
  .win {{ color: #047857; font-weight: 800; font-size: 13.5px; }}
  .num-status {{ text-align: right; width: 130px; }}
  .st-full {{ background: #ecfdf5; color: #047857; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; border: 1px solid #a7f3d0; }}
  .st-part {{ background: #fefce8; color: #854d0e; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; border: 1px solid #fde047; }}
  .st-none {{ background: #fef2f2; color: #991b1b; padding: 2px 8px; border-radius: 6px; font-weight: 700; font-size: 11px; border: 1px solid #fecaca; }}
  tr.tot td {{ background: #ecfdf5; font-weight: 800; color: #065f46; border-top: 2px solid #059669; font-size: 13px; }}
  .cert {{ display: flex; justify-content: space-between; align-items: center; background: #f0fdf4; border: 1px solid #86efac; border-radius: 10px; padding: 12px 16px; font-size: 12px; color: #14532d; margin-top: 20px; }}
  .cert-left b {{ display: block; font-size: 12.5px; color: #166534; margin-bottom: 2px; }}
  .cert-sign {{ text-align: center; border-top: 1px solid #94a3b8; padding-top: 4px; min-width: 140px; font-size: 11px; color: #64748b; }}
  .actions {{ display: flex; justify-content: flex-end; gap: 10px; margin-bottom: 14px; }}
  .btn {{ background: #047857; color: #fff; border: none; padding: 9px 16px; border-radius: 8px; font-weight: 700; cursor: pointer; font-size: 13px; }}
  @media print {{ .actions {{ display: none; }} body {{ background: #fff; padding: 0; }} .sheet {{ border: none; box-shadow: none; }} }}
</style>
</head>
<body>
<div class="sheet">
  <div class="actions"><button class="btn" onclick="window.print()">Print / Save as PDF</button></div>
  <div class="topbar">
    <div>
      <div class="badge">Official Overall Scorecard (Short Summary)</div>
      <h1>{exam_title}</h1>
      <div class="sub">Strictness Level: {strictness_level}/10 &bull; Evaluated: {saved_at}</div>
    </div>
    <div class="scorecard">
      <div style="font-size:11px;opacity:.9;font-weight:700;">TOTAL OBTAINED</div>
      <div class="big">{total_awarded:.1f} / {total_max:.1f}</div>
      <div style="font-size:12px;font-weight:700;">{pct:.1f}% &bull; Grade {grade}</div>
    </div>
  </div>
  <div class="meta">
    <div class="meta-item"><b>Student Name</b><span>{student_name}</span></div>
    <div class="meta-item"><b>Roll Number</b><span>{roll_no}</span></div>
    <div class="meta-item"><b>Subject</b><span>{exam_title}</span></div>
    <div class="meta-item"><b>Total Questions</b><span>{len(q_results)} Questions</span></div>
  </div>
  {remarks_html}
  <table>
    <thead><tr><th>Q.No</th><th>Question Statement</th><th class="num">Max Marks</th><th class="num">Marks Obtained</th><th class="num-status">Status</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
      <tr class="tot"><td colspan="2">GRAND TOTAL</td><td class="num">{total_max:.1f}</td><td class="num win"><b>{total_awarded:.1f}</b></td><td class="num-status">{pct:.1f}% &bull; Grade {grade}</td></tr>
    </tbody>
  </table>
  <div class="cert">
    <div class="cert-left">
      <b>Certified Evaluation Result</b>
      Verified by Automated Vision OCR &amp; Rubric Scoring Model.
    </div>
    <div class="cert-sign">Authorized Examiner Signature</div>
  </div>
</div>
</body>
</html>"""


def save_report_bundle(saved_reports_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Generate and persist PDF, HTML, CSV, and JSON reports into `saved_reports_dir/<report_id>/`."""
    saved_reports_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    saved_at = now.strftime("%Y-%m-%d %H:%M:%S")
    stamp = now.strftime("%Y%m%d_%H%M%S")

    grading = payload.get("grading_report") or {}
    student_name = str(payload.get("student_name") or "").strip()
    roll_no = str(payload.get("roll_no") or "").strip()
    source_filename = str(payload.get("source_filename") or "").strip()
    if (not student_name or not roll_no) and source_filename:
        inf_name, inf_roll = parse_student_meta_from_filename(source_filename)
        if not student_name:
            student_name = inf_name or Path(source_filename).stem
        if not roll_no:
            roll_no = inf_roll

    slug_parts = [
        _safe_slug(roll_no, "") if roll_no else "",
        _safe_slug(student_name, "Student"),
        _safe_slug(str(grading.get("exam_title") or "Exam"), "Exam")[:20],
        stamp,
    ]
    report_id = "_".join(p for p in slug_parts if p)

    report_dir = saved_reports_dir / report_id
    report_dir.mkdir(parents=True, exist_ok=True)

    full_record = {
        "report_id": report_id,
        "session_id": payload.get("session_id") or "verified",
        "student_name": student_name or "Not Specified",
        "roll_no": roll_no or "-",
        "source_filename": source_filename or "-",
        "overall_remarks": str(payload.get("overall_remarks") or "").strip(),
        "strictness_level": int(grading.get("strictness_level") or payload.get("strictness_level") or 5),
        "saved_at": saved_at,
        "grading_report": grading,
    }

    pdf_bytes = build_report_pdf_bytes(full_record, report_mode="detailed")
    short_pdf_bytes = build_report_pdf_bytes(full_record, report_mode="short")
    csv_text = build_report_csv_text(full_record)
    html_text = build_report_html(full_record)
    short_html_text = build_short_report_html(full_record)

    (report_dir / "report.pdf").write_bytes(pdf_bytes)
    (report_dir / "report_detailed.pdf").write_bytes(pdf_bytes)
    (report_dir / "report_short.pdf").write_bytes(short_pdf_bytes)
    (report_dir / "marks_summary.csv").write_text(csv_text, encoding="utf-8-sig")
    (report_dir / "report.html").write_text(html_text, encoding="utf-8")
    (report_dir / "report_detailed.html").write_text(html_text, encoding="utf-8")
    (report_dir / "report_short.html").write_text(short_html_text, encoding="utf-8")
    (report_dir / "report_data.json").write_text(
        json.dumps(full_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total_awarded = float(grading.get("total_marks_awarded") or 0.0)
    total_max = float(grading.get("total_max_marks") or 0.0)
    pct, grade = _compute_grade_and_pct(total_awarded, total_max)

    return {
        "report_id": report_id,
        "student_name": full_record["student_name"],
        "roll_no": full_record["roll_no"],
        "exam_title": grading.get("exam_title") or "Exam Assessment",
        "total_marks_awarded": total_awarded,
        "total_max_marks": total_max,
        "percentage": pct,
        "grade": grade,
        "strictness_level": full_record["strictness_level"],
        "saved_at": saved_at,
        "folder_path": str(report_dir),
        "pdf_url": f"/saved_reports/{report_id}/report.pdf",
        "pdf_detailed_url": f"/saved_reports/{report_id}/report_detailed.pdf",
        "pdf_short_url": f"/saved_reports/{report_id}/report_short.pdf",
        "html_url": f"/saved_reports/{report_id}/report.html",
        "html_detailed_url": f"/saved_reports/{report_id}/report_detailed.html",
        "html_short_url": f"/saved_reports/{report_id}/report_short.html",
        "csv_url": f"/saved_reports/{report_id}/marks_summary.csv",
        "json_url": f"/saved_reports/{report_id}/report_data.json",
    }


def list_saved_reports(saved_reports_dir: Path) -> list[dict[str, Any]]:
    """List all persisted evaluation reports sorted newest-first."""
    if not saved_reports_dir.exists():
        return []
    items: list[dict[str, Any]] = []
    for d in saved_reports_dir.iterdir():
        if not d.is_dir():
            continue
        json_file = d / "report_data.json"
        if not json_file.exists():
            continue
        try:
            rec = json.loads(json_file.read_text(encoding="utf-8"))
            grading = rec.get("grading_report") or {}
            aw = float(grading.get("total_marks_awarded") or 0.0)
            mx = float(grading.get("total_max_marks") or 0.0)
            pct, grade = _compute_grade_and_pct(aw, mx)
            rid = d.name
            items.append(
                {
                    "report_id": rid,
                    "student_name": rec.get("student_name") or "Not Specified",
                    "roll_no": rec.get("roll_no") or "-",
                    "exam_title": grading.get("exam_title") or "Exam Assessment",
                    "total_marks_awarded": aw,
                    "total_max_marks": mx,
                    "percentage": pct,
                    "grade": grade,
                    "strictness_level": rec.get("strictness_level", 5),
                    "saved_at": rec.get("saved_at") or "",
                    "pdf_url": f"/saved_reports/{rid}/report.pdf",
                    "pdf_detailed_url": f"/saved_reports/{rid}/report_detailed.pdf" if (d / "report_detailed.pdf").exists() else f"/saved_reports/{rid}/report.pdf",
                    "pdf_short_url": f"/saved_reports/{rid}/report_short.pdf" if (d / "report_short.pdf").exists() else f"/saved_reports/{rid}/report.pdf",
                    "html_url": f"/saved_reports/{rid}/report.html",
                    "html_detailed_url": f"/saved_reports/{rid}/report_detailed.html" if (d / "report_detailed.html").exists() else f"/saved_reports/{rid}/report.html",
                    "html_short_url": f"/saved_reports/{rid}/report_short.html" if (d / "report_short.html").exists() else f"/saved_reports/{rid}/report.html",
                    "csv_url": f"/saved_reports/{rid}/marks_summary.csv",
                    "json_url": f"/saved_reports/{rid}/report_data.json",
                }
            )
        except Exception:
            continue
    items.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
    return items
