"""End-to-end CLI: extract -> classify -> group -> (optional) evaluate, for
one PDF, writing every output under --out.

PII BOUNDARY: this script only ever touches `PageResult`, `PhysicalGroup`,
`ReconstructedDocument`, `RunReport`, and `EvaluationReport` -- all
de-identified schema types. It never imports `ExtractedPage` text and never
reads an API key value (only `LLMRuntimeConfig.enabled`, a bool). Every file
this script writes is passed through `pipeline.check_no_pii` first.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import compute_metrics, format_report, load_ground_truth
from src.extract import load_rules
from src.grouping import physical_span_and_contiguity
from src.llm_classifier import load_llm_config
from src.pipeline import check_no_pii, run_pipeline
from src.schema import PageResult, PhysicalGroup, PipelineMode, ReconstructedDocument
from src.visualize import render_all


def _page_results_csv(results: list[PageResult]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "page_number",
            "doc_type",
            "decision_source",
            "rule_doc_type",
            "rule_score",
            "rule_margin",
            "matched_class_count",
            "strong_signature_matched",
            "llm_called",
            "llm_failed",
            "llm_doc_type",
            "llm_confidence",
            "marker_style",
            "marker_page",
            "marker_total",
            "instance_id",
            "is_orphan",
            "rotation",
            "normalized_text_length",
            "warnings",
        ]
    )
    for r in sorted(results, key=lambda r: r.page_number):
        writer.writerow(
            [
                r.page_number,
                r.doc_type.value,
                r.decision_source.value,
                r.rule_doc_type.value if r.rule_doc_type is not None else "",
                r.rule_score,
                r.rule_margin,
                r.matched_class_count,
                r.strong_signature_matched,
                r.llm_called,
                r.llm_failed,
                r.llm_doc_type.value if r.llm_doc_type is not None else "",
                r.llm_confidence if r.llm_confidence is not None else "",
                r.marker_style.value,
                r.marker_page if r.marker_page is not None else "",
                r.marker_total if r.marker_total is not None else "",
                r.instance_id or "",
                r.is_orphan,
                r.rotation,
                r.normalized_text_length,
                ";".join(w.value for w in r.warnings),
            ]
        )
    return buf.getvalue()


def _physical_groups_csv(groups: list[PhysicalGroup]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["group_id", "doc_type", "start_page", "end_page", "page_count", "physical_pages"])
    for g in groups:
        writer.writerow(
            [
                g.group_id,
                g.doc_type.value,
                g.start_page,
                g.end_page,
                g.page_count,
                ";".join(str(p) for p in g.physical_pages),
            ]
        )
    return buf.getvalue()


def _reconstructed_documents_csv(documents: list[ReconstructedDocument]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "doc_id",
            "doc_type",
            "instance_id",
            "marker_style",
            "expected_pages",
            "observed_logical_pages",
            "missing_logical_pages",
            "duplicate_logical_pages",
            "first_page_at",
            "last_page_at",
            "physical_span",
            "is_contiguous",
            "completeness",
            "issues",
            "logical_to_physical",
        ]
    )
    for d in documents:
        physical_pages = [pl.physical_page for pl in d.page_links]
        span, is_contiguous = physical_span_and_contiguity(physical_pages)
        mapping = ";".join(
            f"{pl.logical_page}->{pl.physical_page}"
            for pl in sorted(d.page_links, key=lambda pl: (pl.logical_page or 0, pl.physical_page))
        )
        writer.writerow(
            [
                d.doc_id,
                d.doc_type.value,
                d.key.instance_id or "",
                d.key.marker_style.value,
                d.expected_pages if d.expected_pages is not None else "",
                ";".join(str(p) for p in d.observed_logical_pages),
                ";".join(str(p) for p in d.missing_logical_pages),
                ";".join(str(p) for p in d.duplicate_logical_pages),
                d.start_physical_page if d.start_physical_page is not None else "",
                d.end_physical_page if d.end_physical_page is not None else "",
                span,
                is_contiguous,
                d.completeness.value,
                ";".join(i.value for i in d.issues),
                mapping,
            ]
        )
    return buf.getvalue()


def _write(path: Path, content: str) -> None:
    check_no_pii(content)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the full page-classification pipeline for one PDF "
        "and save every output under --out."
    )
    parser.add_argument("--pdf", required=True, help="Path to the source PDF.")
    parser.add_argument(
        "--mode",
        default=PipelineMode.RULE_ONLY.value,
        choices=[m.value for m in PipelineMode],
        help="Pipeline mode (default: rule-only).",
    )
    parser.add_argument("--out", required=True, help="Output directory (created if missing).")
    parser.add_argument(
        "--truth", default=None, help="Ground-truth CSV; if given, also runs evaluation."
    )
    parser.add_argument(
        "--rules", default="config/rules.yaml", help="Path to the rules YAML file."
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Also render page_strip.png and (if --truth given) confusion_matrix.png.",
    )
    args = parser.parse_args(argv)

    rules = load_rules(args.rules)
    llm_config = load_llm_config()
    mode = PipelineMode(args.mode)

    result = run_pipeline(args.pdf, rules, mode, llm_config)
    report = result.run_report

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []

    _write(out_dir / "page_results.csv", _page_results_csv(result.page_results))
    saved_files.append("page_results.csv")

    _write(out_dir / "physical_groups.csv", _physical_groups_csv(result.physical_groups))
    saved_files.append("physical_groups.csv")

    _write(
        out_dir / "reconstructed_documents.csv",
        _reconstructed_documents_csv(result.reconstructed_documents),
    )
    saved_files.append("reconstructed_documents.csv")

    _write(out_dir / "run_report.json", report.model_dump_json(indent=2))
    saved_files.append("run_report.json")

    truth = None
    eval_report = None
    if args.truth:
        truth = load_ground_truth(args.truth)
        eval_report = compute_metrics(
            truth=truth,
            predictions=result.page_results,
            mode=mode,
            dataset_name=Path(args.truth).name,
        )
        _write(out_dir / "evaluation.json", eval_report.model_dump_json(indent=2))
        saved_files.append("evaluation.json")

        eval_text = format_report(eval_report, truth=truth, predictions=result.page_results)
        _write(out_dir / "evaluation.txt", eval_text)
        saved_files.append("evaluation.txt")

    if args.viz:
        figure_paths = render_all(result.page_results, truth, eval_report, out_dir)
        saved_files.extend(str(p.relative_to(out_dir)) for p in figure_paths)

    print(f"input: {Path(args.pdf).name}")
    print(f"mode: {mode.value}")
    print(f"rules version: {rules.get('version')}")
    print()
    print(f"total_pages: {report.total_pages}")
    print(f"rule_resolved: {report.rule_resolved}")
    print(f"llm_target_pages: {len(report.llm_target_pages)}")
    print(f"llm_called: {report.llm_called}")
    print(f"rule_fallback_count: {report.rule_fallback_count}")
    print()
    print(f"physical_groups: {len(result.physical_groups)}")
    print(f"reconstructed_documents: {len(result.reconstructed_documents)}")
    print(f"orphan_pages: {len(report.orphan_pages)}")

    if not llm_config.enabled:
        print()
        print(
            "LLM disabled: no LLM_API_KEY found in the environment/.env -- "
            "this run made zero LLM calls."
        )

    print()
    if eval_report is not None:
        print(f"accuracy: {eval_report.accuracy:.4f}")
        print(f"macro_f1_supported: {eval_report.macro_f1_supported:.4f}")
    else:
        print("evaluation skipped: no --truth provided")

    print()
    print("saved files:")
    for name in saved_files:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
