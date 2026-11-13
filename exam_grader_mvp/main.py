"""CLI entry point for the Automated Exam Answer Sheet Checking Engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import TROCR_MODEL_NAME
from grader import ExamGrader, ExamGradingReport, load_rubric
from ocr_engine import OCREngine, OCRPipelineResult

console = Console()


def build_parser() -> argparse.ArgumentParser:
    """Configure CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="exam-grader",
        description="Automated Exam Answer Sheet Checking Engine (OpenCV + TrOCR + MiniLM + Gemini VLM)",
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        required=True,
        help="One or more scanned/photographed exam sheet image paths (e.g. sample_page1.jpg sample_page2.jpg).",
    )
    parser.add_argument(
        "--rubric",
        default="sample_rubric.json",
        help="Path to the structured exam rubric JSON file (default: sample_rubric.json).",
    )
    parser.add_argument(
        "--trocr-model",
        default=TROCR_MODEL_NAME,
        help=f"Hugging Face TrOCR model name (default: {TROCR_MODEL_NAME}).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip Gemini API calls and run pure local OpenCV + TrOCR + MiniLM + heuristic pipeline.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output path to save the full grading report JSON (e.g. results.json).",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug_crops",
        help="Directory to write segmented line/diagram crops and annotated layout images (default: debug_crops).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed INFO logging.",
    )
    return parser


def render_segmentation_summary(ocr_result: OCRPipelineResult) -> None:
    """Print segmentation summary table using Rich."""
    table = Table(
        title="1. Layout Segmentation Summary",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Page", justify="center", style="bold")
    table.add_column("Source Image", overflow="fold")
    table.add_column("Deskew (deg)", justify="right")
    table.add_column("Text Lines", justify="right", style="green")
    table.add_column("Diagrams", justify="right", style="red")
    table.add_column("Layout Overlay", overflow="fold")

    for page in ocr_result.pages:
        table.add_row(
            str(page.page_number),
            str(page.source_path),
            f"{page.skew_angle_deg:+.2f}°",
            str(len(page.text_lines)),
            str(len(page.diagrams)),
            str(page.layout_image_path),
        )

    console.print(table)
    console.print(
        f"[dim]Debug crops saved under:[/dim] [bold]{ocr_result.debug_dir.resolve()}[/bold]\n"
    )


def render_ocr_transcript(mapped_answers: dict[str, str]) -> None:
    """Print reconstructed OCR text grouped by detected Question ID."""
    console.print("[bold cyan]2. Reconstructed OCR Text (Grouped by Question ID)[/bold cyan]")
    if not mapped_answers:
        console.print(
            Panel(
                "[yellow]No question markers (e.g., Q1, Q.1, Ans 1:, 1)) were detected in the transcript.[/yellow]",
                title="Question Mapping",
                border_style="yellow",
            )
        )
        return

    for qid, text in mapped_answers.items():
        body = text if text else "[dim](No answer text extracted)[/dim]"
        console.print(
            Panel(
                body,
                title=f"[bold green]{qid}[/bold green]",
                border_style="blue",
                expand=True,
            )
        )
    console.print()


def render_grading_report(report: ExamGradingReport) -> None:
    """Print question-by-question grading table and overall score summary."""
    table = Table(
        title=f"3. Step-Wise Grading Report — {report.exam_title} ({report.mode})",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
        expand=True,
    )
    table.add_column("QID", style="bold", justify="center", no_wrap=True)
    table.add_column("Sim", justify="right", no_wrap=True)
    table.add_column("Diagram", justify="center", min_width=12)
    table.add_column("Step Marks", min_width=24, ratio=2)
    table.add_column("Missing Points", min_width=14, ratio=1)
    table.add_column("Feedback", min_width=24, ratio=2)

    for res in report.question_results:
        sim_str = f"{res.semantic_similarity:.2f}"
        if res.diagram_evaluation is not None:
            diag_str = (
                f"{res.diagram_evaluation.structure_score:.2f} / 1.00\n"
                f"({'Relevant' if res.diagram_evaluation.is_relevant else 'Off-topic'}, "
                f"{'Labels OK' if res.diagram_evaluation.labels_accurate else 'Labels Weak'})"
            )
        else:
            diag_str = "[dim]N/A[/dim]"

        step_lines = [
            f"[bold]{res.marks_awarded:.2f} / {res.max_marks:.2f}[/bold]"
        ]
        for step in res.step_wise_breakdown:
            color = (
                "green"
                if step.status == "Hit"
                else ("yellow" if step.status == "Partial" else "red")
            )
            step_lines.append(
                f"[{color}]• {step.point_id} [{step.status}]: {step.awarded_point_marks:.2f}/{step.max_point_marks:.2f}[/{color}]"
            )

        missing_str = (
            "\n".join(f"• {m}" for m in res.missing_key_points)
            if res.missing_key_points
            else "[green]None[/green]"
        )

        feedback_parts = [res.student_feedback]
        if res.diagram_evaluation is not None and res.diagram_evaluation.critique:
            feedback_parts.append(f"Diagram: {res.diagram_evaluation.critique}")

        table.add_row(
            res.question_id,
            sim_str,
            diag_str,
            "\n".join(step_lines),
            missing_str,
            "\n".join(feedback_parts),
        )

    console.print(table)

    pct = (
        (report.total_marks_awarded / report.total_max_marks) * 100.0
        if report.total_max_marks > 0
        else 0.0
    )
    summary_text = Text.assemble(
        ("Total Score: ", "bold"),
        (f"{report.total_marks_awarded:.2f} / {report.total_max_marks:.2f} ", "bold green"),
        (f"({pct:.1f}%)", "cyan"),
    )
    console.print(Panel(summary_text, border_style="green", expand=False))


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end exam checking pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    for p in args.pages:
        if not Path(p).exists():
            console.print(f"[bold red]Error:[/bold red] Page image not found: {p}")
            return 1

    try:
        rubric = load_rubric(args.rubric)
    except Exception as exc:
        console.print(f"[bold red]Error loading rubric '{args.rubric}':[/bold red] {exc}")
        return 1

    console.print(
        Panel(
            f"[bold]Automated Exam Answer Sheet Checking Engine[/bold]\n"
            f"Pages: {', '.join(args.pages)}\n"
            f"Rubric: {args.rubric} ({len(rubric.questions)} questions)\n"
            f"TrOCR Model: {args.trocr_model}\n"
            f"Mode: {'Offline Local' if args.offline else 'Online (Gemini VLM + Local MiniLM/TrOCR)'}",
            border_style="cyan",
        )
    )

    with console.status("[bold cyan]Segmenting layout & running TrOCR...[/bold cyan]"):
        ocr_engine = OCREngine(trocr_model_name=args.trocr_model)
        ocr_result = ocr_engine.process_pages(args.pages, debug_dir=args.debug_dir)

    render_segmentation_summary(ocr_result)

    with console.status("[bold magenta]Mapping questions, scoring semantics & grading...[/bold magenta]"):
        grader = ExamGrader(offline=args.offline)
        report = grader.grade_exam(ocr_result, rubric)

    render_ocr_transcript(report.mapped_answers)
    render_grading_report(report)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "segmentation": [
                {
                    "page_number": page.page_number,
                    "source_path": str(page.source_path),
                    "skew_angle_deg": page.skew_angle_deg,
                    "text_lines_detected": len(page.text_lines),
                    "diagrams_detected": len(page.diagrams),
                    "layout_image_path": str(page.layout_image_path),
                    "line_transcripts": [
                        {"region_id": r.region_id, "text": r.text, "crop": str(r.crop_path)}
                        for r in page.text_lines
                    ],
                    "diagram_crops": [
                        {"region_id": d.region_id, "crop": str(d.crop_path)}
                        for d in page.diagrams
                    ],
                }
                for page in ocr_result.pages
            ],
            "grading_report": report.model_dump(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"\n[bold green]Saved full JSON report to:[/bold green] {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
