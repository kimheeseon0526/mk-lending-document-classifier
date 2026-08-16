"""Scoring for the mortgage document page classifier.

PII BOUNDARY: reads only `GroundTruthEntry` (label + provenance, no PII) and
`PageResult` (de-identified, safe to persist). Never touches `ExtractedPage`
or raw page text -- there is nothing here that could leak PII even by
accident, and `format_report` prints only labels, page numbers, and counts.

Compares ground truth against pipeline predictions and reports accuracy,
per-class precision/recall/F1, macro-F1 over supported classes, and a
confusion matrix. Does not classify anything itself.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from pydantic import ValidationError

from src.extract import extract_pages, load_rules
from src.rule_classifier import classify_page
from src.schema import (
    DOC_TYPE_ORDER,
    ClassMetrics,
    DocType,
    EvaluationReport,
    GroundTruthEntry,
    PageResult,
    PipelineMode,
)

logger = logging.getLogger(__name__)


def load_ground_truth(path: str | Path) -> list[GroundTruthEntry]:
    """Load `data/ground_truth*.csv` into `GroundTruthEntry` rows.

    Raises FileNotFoundError if `path` does not exist, and ValueError (naming
    the offending row) if a row fails to parse or fails `GroundTruthEntry`
    validation -- e.g. a `doc_type` string outside the `DocType` enum.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    entries: list[GroundTruthEntry] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=2):  # header occupies row 1
            try:
                entries.append(
                    GroundTruthEntry(
                        page_number=int(row["page_number"]),
                        doc_type=row["doc_type"],
                        source_document=row.get("source_document") or None,
                        source_page=int(row["source_page"]) if row.get("source_page") else None,
                    )
                )
            except (ValidationError, ValueError, KeyError) as e:
                raise ValueError(
                    f"Invalid ground truth row {row_number} in {path}: {e}"
                ) from e
    return entries


def _duplicates(values: list[int]) -> list[int]:
    seen: set[int] = set()
    dupes: set[int] = set()
    for v in values:
        if v in seen:
            dupes.add(v)
        seen.add(v)
    return sorted(dupes)


def validate_inputs(truth: list[GroundTruthEntry], predictions: list[PageResult]) -> None:
    """Verify `truth` and `predictions` can be scored together.

    Every failure below raises ValueError with a specific message rather
    than letting `compute_metrics` silently score a partial or misaligned
    dataset. Must be called (and pass) before any metric is computed.
    """
    truth_pages = [t.page_number for t in truth]
    pred_pages = [p.page_number for p in predictions]

    truth_dupes = _duplicates(truth_pages)
    if truth_dupes:
        raise ValueError(f"Duplicate page_number(s) in ground truth: {truth_dupes}")

    pred_dupes = _duplicates(pred_pages)
    if pred_dupes:
        raise ValueError(f"Duplicate page_number(s) in predictions: {pred_dupes}")

    truth_set = set(truth_pages)
    pred_set = set(pred_pages)

    only_in_truth = sorted(truth_set - pred_set)
    if only_in_truth:
        raise ValueError(
            f"Page(s) present in ground truth but missing from predictions: {only_in_truth}"
        )

    only_in_predictions = sorted(pred_set - truth_set)
    if only_in_predictions:
        raise ValueError(
            f"Page(s) present in predictions but missing from ground truth: {only_in_predictions}"
        )

    if truth_set != pred_set:
        # Unreachable given the two checks above -- the spec calls out 1:1
        # correspondence as its own condition, so it is asserted explicitly
        # rather than only implied by the difference checks.
        raise ValueError("Ground truth and prediction page_number sets are not a 1:1 match")

    # DocType membership is already enforced by pydantic when GroundTruthEntry
    # / PageResult are constructed. This re-checks the invariant explicitly
    # (per the spec's validation list) so a future change that weakened
    # either model's typing would surface here too, not just silently score.
    for entry in truth:
        if entry.doc_type not in DocType:
            raise ValueError(
                f"Ground truth page {entry.page_number} has an invalid doc_type: {entry.doc_type!r}"
            )
    for pred in predictions:
        if pred.doc_type not in DocType:
            raise ValueError(
                f"Prediction page {pred.page_number} has an invalid doc_type: {pred.doc_type!r}"
            )


def compute_metrics(
    truth: list[GroundTruthEntry],
    predictions: list[PageResult],
    mode: PipelineMode,
    dataset_name: str,
) -> EvaluationReport:
    """Score `predictions` against `truth`, page by page.

    Calls `validate_inputs` first; a ValueError from that call propagates
    unchanged and nothing below it runs.

    A metric left undefined by a 0-denominator is stored as None, not 0.0.
    package_01 has zero OTHER ground-truth pages, so OTHER's recall and F1
    are mathematically undefined (0/0) rather than 0 -- filling them with
    0.0 would claim evidence that was never gathered and would silently
    drag `macro_f1_supported` down for a class the run never got to
    exercise. OTHER's precision is still computed whenever the pipeline
    predicts OTHER at least once, since a false positive there is real,
    observed evidence regardless of how many true OTHER pages exist.
    """
    validate_inputs(truth, predictions)

    truth_by_page = {t.page_number: t.doc_type for t in truth}
    pred_by_page = {p.page_number: p.doc_type for p in predictions}

    total_pages = len(truth_by_page)
    correct = sum(1 for page, actual in truth_by_page.items() if pred_by_page[page] == actual)
    accuracy = correct / total_pages if total_pages else 0.0

    # confusion[actual][predicted]; every DOC_TYPE_ORDER label gets a row and
    # a column even with zero support, so a class that never appears is
    # still visible as an all-zero row rather than a missing key.
    confusion: dict[DocType, dict[DocType, int]] = {
        actual: {predicted: 0 for predicted in DOC_TYPE_ORDER} for actual in DOC_TYPE_ORDER
    }
    for page, actual in truth_by_page.items():
        confusion[actual][pred_by_page[page]] += 1

    per_class: list[ClassMetrics] = []
    for doc_type in DOC_TYPE_ORDER:
        support = sum(1 for actual in truth_by_page.values() if actual == doc_type)
        predicted_count = sum(1 for predicted in pred_by_page.values() if predicted == doc_type)
        tp = confusion[doc_type][doc_type]
        fp = predicted_count - tp
        fn = support - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None

        if precision is None or recall is None:
            f1 = None
        elif precision == 0.0 and recall == 0.0:
            f1 = 0.0  # harmonic mean of two defined zeros is 0 by convention
        else:
            f1 = 2 * precision * recall / (precision + recall)

        per_class.append(
            ClassMetrics(
                doc_type=doc_type,
                support=support,
                predicted_count=predicted_count,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    supported_classes = [m.doc_type for m in per_class if m.support > 0]
    # A supported class can still carry f1=None (e.g. it was never predicted,
    # so precision is undefined) -- those are excluded from the average for
    # the same reason support==0 classes are: you cannot average an
    # undefined value without fabricating evidence.
    supported_f1_values = [m.f1 for m in per_class if m.support > 0 and m.f1 is not None]
    macro_f1_supported = (
        sum(supported_f1_values) / len(supported_f1_values) if supported_f1_values else 0.0
    )

    misclassified_pages = sorted(
        page for page, actual in truth_by_page.items() if pred_by_page[page] != actual
    )

    return EvaluationReport(
        mode=mode,
        dataset_name=dataset_name,
        total_pages=total_pages,
        accuracy=accuracy,
        per_class=per_class,
        macro_f1_supported=macro_f1_supported,
        supported_classes=supported_classes,
        confusion_matrix=confusion,
        misclassified_pages=misclassified_pages,
    )


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.extend("  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def format_report(
    report: EvaluationReport,
    truth: list[GroundTruthEntry] | None = None,
    predictions: list[PageResult] | None = None,
) -> str:
    """Render `report` as human-readable text. Never prints page text.

    `truth`/`predictions` are optional. `EvaluationReport.misclassified_pages`
    stores page numbers only -- the schema has no field for a page's actual
    or predicted label, since it is meant to be small and persistable. When
    the caller still has `truth`/`predictions` in scope (as the CLI does,
    right after calling `compute_metrics`), passing them here expands the
    misclassified section into (page, actual, predicted) rows; without them
    it falls back to a bare page-number list.
    """
    lines: list[str] = []
    lines.append(f"mode: {report.mode.value}")
    lines.append(f"dataset: {report.dataset_name}")
    lines.append(f"total_pages: {report.total_pages}")
    lines.append(f"accuracy: {report.accuracy:.4f}")
    lines.append(f"macro_f1_supported: {report.macro_f1_supported:.4f}")
    lines.append("")

    lines.append("per-class metrics:")
    headers = ["doc_type", "support", "predicted", "precision", "recall", "f1"]
    rows = [
        [
            m.doc_type.value,
            str(m.support),
            str(m.predicted_count),
            _fmt(m.precision),
            _fmt(m.recall),
            _fmt(m.f1),
        ]
        for m in report.per_class
    ]
    lines.append(_render_table(headers, rows))
    lines.append("")

    zero_support = [m.doc_type.value for m in report.per_class if m.support == 0]
    if zero_support:
        lines.append(
            f"Note: {', '.join(zero_support)} has support 0 in this dataset. "
            "recall/f1 are undefined (N/A), not 0.0, and excluded from "
            "macro_f1_supported."
        )
        lines.append("")

    lines.append("confusion matrix (rows=actual, cols=predicted):")
    cm_headers = ["actual\\predicted"] + [dt.value for dt in DOC_TYPE_ORDER]
    cm_rows = [
        [actual.value]
        + [str(report.confusion_matrix[actual][predicted]) for predicted in DOC_TYPE_ORDER]
        for actual in DOC_TYPE_ORDER
    ]
    lines.append(_render_table(cm_headers, cm_rows))
    lines.append("")

    lines.append(f"misclassified pages ({len(report.misclassified_pages)}):")
    if not report.misclassified_pages:
        lines.append("  none")
    elif truth is not None and predictions is not None:
        truth_by_page = {t.page_number: t.doc_type for t in truth}
        pred_by_page = {p.page_number: p.doc_type for p in predictions}
        mis_headers = ["page", "actual", "predicted"]
        mis_rows = [
            [str(page), truth_by_page[page].value, pred_by_page[page].value]
            for page in report.misclassified_pages
        ]
        lines.append(_render_table(mis_headers, mis_rows))
    else:
        lines.append("  " + ", ".join(str(p) for p in report.misclassified_pages))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline predictions against a ground-truth CSV."
    )
    parser.add_argument("--truth", required=True, help="Path to the ground-truth CSV.")
    parser.add_argument("--pdf", required=True, help="Path to the shuffled package PDF to classify.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[m.value for m in PipelineMode],
        help="Pipeline arm to evaluate.",
    )
    parser.add_argument(
        "--rules", default="config/rules.yaml", help="Path to the rules YAML file."
    )
    args = parser.parse_args(argv)

    mode = PipelineMode(args.mode)
    truth = load_ground_truth(args.truth)

    if mode is PipelineMode.RULE_ONLY:
        rules = load_rules(args.rules)
        pages = extract_pages(args.pdf, rules)
        predictions = [classify_page(page, rules) for page in pages]
    else:
        # No LLM delegation path exists yet. Raising here keeps a
        # not-yet-supported mode loud and explicit instead of quietly
        # returning a rule-only (or empty) report mislabeled with a mode
        # that implies LLM involvement.
        raise NotImplementedError(
            f"Pipeline mode {mode.value!r} is not implemented yet -- only "
            f"{PipelineMode.RULE_ONLY.value!r} can be evaluated today."
        )

    report = compute_metrics(
        truth=truth,
        predictions=predictions,
        mode=mode,
        dataset_name=Path(args.truth).name,
    )
    print(format_report(report, truth=truth, predictions=predictions))


if __name__ == "__main__":
    main()
