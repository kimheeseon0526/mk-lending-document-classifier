"""Tests for src/evaluate.py. No real PDF is used -- GroundTruthEntry and
PageResult lists are built inline and scored directly."""

from __future__ import annotations

import pytest

from src.evaluate import compute_metrics, validate_inputs
from src.schema import (
    DOC_TYPE_ORDER,
    DecisionSource,
    DocType,
    GroundTruthEntry,
    PageResult,
    PipelineMode,
)


def _truth(page_number: int, doc_type: DocType) -> GroundTruthEntry:
    return GroundTruthEntry(page_number=page_number, doc_type=doc_type)


def _pred(page_number: int, doc_type: DocType) -> PageResult:
    return PageResult(
        page_number=page_number,
        doc_type=doc_type,
        decision_source=DecisionSource.RULE,
    )


def test_all_correct_gives_accuracy_one() -> None:
    truth = [
        _truth(1, DocType.URLA_1003),
        _truth(2, DocType.CREDIT_REPORT),
        _truth(3, DocType.TITLE_REPORT),
        _truth(4, DocType.INCOME_DOC),
    ]
    predictions = [
        _pred(1, DocType.URLA_1003),
        _pred(2, DocType.CREDIT_REPORT),
        _pred(3, DocType.TITLE_REPORT),
        _pred(4, DocType.INCOME_DOC),
    ]

    report = compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")

    assert report.accuracy == 1.0
    assert report.misclassified_pages == []
    assert report.macro_f1_supported == 1.0


def test_partial_errors_match_hand_calculation() -> None:
    # actual: URLA, URLA, CREDIT, TITLE
    # pred:   URLA, CREDIT, CREDIT, CREDIT
    truth = [
        _truth(1, DocType.URLA_1003),
        _truth(2, DocType.URLA_1003),
        _truth(3, DocType.CREDIT_REPORT),
        _truth(4, DocType.TITLE_REPORT),
    ]
    predictions = [
        _pred(1, DocType.URLA_1003),
        _pred(2, DocType.CREDIT_REPORT),
        _pred(3, DocType.CREDIT_REPORT),
        _pred(4, DocType.CREDIT_REPORT),
    ]

    report = compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")

    assert report.accuracy == pytest.approx(2 / 4)
    assert report.misclassified_pages == [2, 4]

    by_type = {m.doc_type: m for m in report.per_class}

    urla = by_type[DocType.URLA_1003]
    assert (urla.true_positives, urla.false_positives, urla.false_negatives) == (1, 0, 1)
    assert urla.precision == pytest.approx(1.0)
    assert urla.recall == pytest.approx(0.5)
    assert urla.f1 == pytest.approx(2 / 3)

    credit = by_type[DocType.CREDIT_REPORT]
    assert (credit.true_positives, credit.false_positives, credit.false_negatives) == (1, 2, 0)
    assert credit.precision == pytest.approx(1 / 3)
    assert credit.recall == pytest.approx(1.0)
    assert credit.f1 == pytest.approx(0.5)

    # TITLE_REPORT has support but was never predicted -- precision is
    # undefined (0/0), so f1 must be None even though support > 0.
    title = by_type[DocType.TITLE_REPORT]
    assert title.support == 1
    assert title.predicted_count == 0
    assert title.precision is None
    assert title.recall == pytest.approx(0.0)
    assert title.f1 is None


def test_zero_support_class_has_none_recall_and_f1() -> None:
    truth = [_truth(1, DocType.URLA_1003), _truth(2, DocType.CREDIT_REPORT)]
    predictions = [_pred(1, DocType.URLA_1003), _pred(2, DocType.CREDIT_REPORT)]

    report = compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")
    by_type = {m.doc_type: m for m in report.per_class}

    other = by_type[DocType.OTHER]
    assert other.support == 0
    assert other.recall is None
    assert other.f1 is None


def test_zero_support_class_excluded_from_macro_f1() -> None:
    truth = [_truth(1, DocType.URLA_1003), _truth(2, DocType.CREDIT_REPORT)]
    predictions = [_pred(1, DocType.URLA_1003), _pred(2, DocType.CREDIT_REPORT)]

    report = compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")

    # Both supported classes (URLA_1003, CREDIT_REPORT) are perfectly
    # predicted, so if OTHER/INCOME_DOC/TITLE_REPORT (all support 0) were
    # incorrectly folded into the average, it could only pull the score
    # below 1.0 or raise a ZeroDivisionError. Neither happens.
    assert DocType.OTHER not in report.supported_classes
    assert set(report.supported_classes) == {DocType.URLA_1003, DocType.CREDIT_REPORT}
    assert report.macro_f1_supported == pytest.approx(1.0)


def test_other_false_prediction_is_counted_as_false_positive() -> None:
    truth = [_truth(1, DocType.URLA_1003), _truth(2, DocType.CREDIT_REPORT)]
    predictions = [_pred(1, DocType.OTHER), _pred(2, DocType.CREDIT_REPORT)]

    report = compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")
    by_type = {m.doc_type: m for m in report.per_class}

    other = by_type[DocType.OTHER]
    assert other.support == 0
    assert other.predicted_count == 1
    assert other.false_positives == 1
    # precision IS computable here: OTHER was predicted once and it was wrong.
    assert other.precision == pytest.approx(0.0)
    # but recall/f1 stay undefined -- OTHER has no ground-truth pages at all.
    assert other.recall is None
    assert other.f1 is None


def test_duplicate_ground_truth_page_raises() -> None:
    truth = [_truth(1, DocType.URLA_1003), _truth(1, DocType.CREDIT_REPORT)]
    predictions = [_pred(1, DocType.URLA_1003)]

    with pytest.raises(ValueError, match=r"(?i)duplicate.*ground truth"):
        validate_inputs(truth, predictions)


def test_missing_prediction_page_raises() -> None:
    truth = [_truth(1, DocType.URLA_1003), _truth(2, DocType.CREDIT_REPORT)]
    predictions = [_pred(1, DocType.URLA_1003)]

    with pytest.raises(ValueError, match=r"(?i)missing from predictions"):
        validate_inputs(truth, predictions)


def test_confusion_matrix_has_all_five_labels() -> None:
    truth = [_truth(1, DocType.URLA_1003)]
    predictions = [_pred(1, DocType.URLA_1003)]

    report = compute_metrics(truth, predictions, PipelineMode.RULE_ONLY, "unit-test")

    assert set(report.confusion_matrix.keys()) == set(DOC_TYPE_ORDER)
    for actual_row in report.confusion_matrix.values():
        assert set(actual_row.keys()) == set(DOC_TYPE_ORDER)
