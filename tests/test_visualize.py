"""Tests for src/visualize.py. No real PDF is used -- inline PageResult /
GroundTruthEntry lists and an EvaluationReport built via evaluate.compute_metrics
stand in for real pipeline output. Files are written to tmp_path and only
checked for existence -- pixel content is never inspected."""

from __future__ import annotations

import matplotlib

from src.evaluate import compute_metrics
from src.schema import (
    DOC_TYPE_ORDER,
    DecisionSource,
    DocType,
    GroundTruthEntry,
    PageResult,
    PipelineMode,
)
from src.visualize import (
    get_label_colors,
    render_all,
    render_confusion_matrix,
    render_page_strip,
    split_pages_into_rows,
    zero_support_note,
)


def _truth(page_number: int, doc_type: DocType) -> GroundTruthEntry:
    return GroundTruthEntry(page_number=page_number, doc_type=doc_type)


def _pred(page_number: int, doc_type: DocType) -> PageResult:
    return PageResult(
        page_number=page_number, doc_type=doc_type, decision_source=DecisionSource.RULE
    )


def test_matplotlib_backend_is_agg() -> None:
    assert matplotlib.get_backend().lower() == "agg"


def test_all_five_labels_have_colors() -> None:
    colors = get_label_colors()
    assert set(colors.keys()) == set(DOC_TYPE_ORDER)


def test_color_mapping_is_deterministic() -> None:
    # Both render functions call get_label_colors() as their sole color
    # source, so the same label always gets the same color across figures
    # as long as this mapping itself is stable across calls.
    assert get_label_colors() == get_label_colors()


def test_other_color_is_neutral_gray() -> None:
    hex_color = get_label_colors()[DocType.OTHER].lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    assert max(r, g, b) - min(r, g, b) <= 5  # roughly equal channels == gray


def test_split_pages_into_rows_single_row_when_small() -> None:
    assert split_pages_into_rows(39, max_per_row=40) == [(1, 39)]


def test_split_pages_into_rows_splits_100_pages() -> None:
    ranges = split_pages_into_rows(100, max_per_row=40)
    assert ranges == [(1, 40), (41, 80), (81, 100)]
    assert len(ranges) > 1


def test_split_pages_into_rows_empty() -> None:
    assert split_pages_into_rows(0) == []


def test_page_strip_creates_file_with_ground_truth(tmp_path) -> None:
    truth = [_truth(i, DocType.URLA_1003) for i in range(1, 6)]
    predictions = [_pred(i, DocType.URLA_1003) for i in range(1, 6)]
    out_path = tmp_path / "strip.png"

    result = render_page_strip(predictions, truth, out_path, "Test Strip")

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_page_strip_creates_file_without_ground_truth(tmp_path) -> None:
    predictions = [_pred(i, DocType.CREDIT_REPORT) for i in range(1, 6)]
    out_path = tmp_path / "strip_no_truth.png"

    result = render_page_strip(predictions, None, out_path, "Test Strip No Truth")

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_page_strip_handles_100_pages_multi_row(tmp_path) -> None:
    predictions = [
        _pred(i, DOC_TYPE_ORDER[i % len(DOC_TYPE_ORDER)]) for i in range(1, 101)
    ]
    out_path = tmp_path / "strip_100.png"

    render_page_strip(predictions, None, out_path, "Test 100 Pages")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def _sample_report():
    truth = [
        _truth(1, DocType.URLA_1003),
        _truth(2, DocType.CREDIT_REPORT),
    ]
    predictions = [
        _pred(1, DocType.URLA_1003),
        _pred(2, DocType.CREDIT_REPORT),
    ]
    return compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")


def _full_support_report():
    truth = [_truth(i + 1, dt) for i, dt in enumerate(DOC_TYPE_ORDER)]
    predictions = [_pred(i + 1, dt) for i, dt in enumerate(DOC_TYPE_ORDER)]
    return compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test-full")


def test_confusion_matrix_creates_file(tmp_path) -> None:
    report = _sample_report()
    out_path = tmp_path / "cm.png"

    result = render_confusion_matrix(report, out_path, "Test CM")

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_zero_support_note_present_when_a_class_has_no_examples() -> None:
    report = _sample_report()  # only URLA_1003/CREDIT_REPORT have support
    note = zero_support_note(report)
    assert note is not None
    assert "support 0" in note


def test_zero_support_note_absent_when_all_classes_supported() -> None:
    report = _full_support_report()
    assert all(m.support > 0 for m in report.per_class)
    assert zero_support_note(report) is None


def test_render_all_creates_both_files(tmp_path) -> None:
    truth = [_truth(1, DocType.URLA_1003), _truth(2, DocType.CREDIT_REPORT)]
    predictions = [_pred(1, DocType.URLA_1003), _pred(2, DocType.CREDIT_REPORT)]
    report = compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")

    paths = render_all(predictions, truth, report, tmp_path)

    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_render_all_skips_confusion_matrix_when_report_none(tmp_path, capsys) -> None:
    predictions = [_pred(1, DocType.URLA_1003)]

    paths = render_all(predictions, None, None, tmp_path)

    assert len(paths) == 1
    assert paths[0].name == "page_strip.png"
    captured = capsys.readouterr()
    assert "confusion matrix skipped" in captured.out
