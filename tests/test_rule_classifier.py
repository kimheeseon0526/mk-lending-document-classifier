"""Tests for src/rule_classifier.py. No real PDF is used -- ExtractedPage
instances are built inline so scoring and delegation logic can be checked
against known, hand-computed inputs."""

from __future__ import annotations

import pytest

from src.rule_classifier import classify_page, should_delegate, validate_never_strong
from src.schema import DocType, ExtractedPage, WarningCode

RULES: dict = {
    "scoring": {
        "strong_weight": 1.0,
        "weak_weight": 0.25,
        "evidence_saturation": 1.0,
        "case_sensitive": False,
    },
    "delegation": {
        "strong_signature_short_circuit": True,
        "short_text_threshold": 100,
        "rule_score_threshold": 0.60,
        "rule_margin_threshold": 0.20,
        "delegate_on_conflicting_strong_signals": True,
        "require_multiple_classes_for_margin_check": True,
    },
    "doc_types": {
        "URLA_1003": {
            "strong": ["Uniform Residential Loan Application"],
            "weak": ["Borrower Name"],
        },
        "INCOME_DOC": {
            "strong": ["Profit & Loss Statement"],
            "weak": ["Net Income"],
        },
        "CREDIT_REPORT": {
            "strong": ["XACTUS"],
            "weak": ["Tradeline"],
        },
        "TITLE_REPORT": {
            "strong": ["CLTA Preliminary Report Form"],
            "weak": ["Legal Description"],
        },
        "OTHER": {"strong": [], "weak": []},
    },
    "never_strong": ["Borrower", "Schedule A"],
}


def _page(text: str, normalized_text_length: int, page_number: int = 1) -> ExtractedPage:
    return ExtractedPage(
        page_number=page_number,
        text=text,
        normalized_text_length=normalized_text_length,
        rotation=0,
        image_count=0,
    )


def test_single_strong_signature_scores_one_and_is_not_delegated() -> None:
    text = "This page is a Uniform Residential Loan Application " + "filler " * 20
    page = _page(text, normalized_text_length=150)

    result = classify_page(page, RULES)

    assert result.doc_type == DocType.URLA_1003
    assert result.rule_score == 1.0
    assert result.strong_signature_matched is True

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is False
    assert reasons == []


def test_single_weak_signature_scores_quarter_and_is_delegated() -> None:
    text = "This page discusses Net Income " + "filler " * 20
    page = _page(text, normalized_text_length=150)

    result = classify_page(page, RULES)

    assert result.doc_type == DocType.INCOME_DOC
    assert result.rule_score == 0.25
    assert result.strong_signature_matched is False

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is True
    assert reasons == [WarningCode.LOW_RULE_SCORE]


def test_no_signature_falls_back_to_other_and_is_delegated() -> None:
    text = "Completely unrelated boilerplate text " + "filler " * 20
    page = _page(text, normalized_text_length=200)

    result = classify_page(page, RULES)

    assert result.doc_type == DocType.OTHER
    assert result.rule_score == 0.0
    assert result.rule_margin == 0.0
    assert WarningCode.NO_SIGNATURE in result.warnings

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is True
    assert reasons != []


def test_short_text_is_delegated_when_no_strong_signature() -> None:
    text = "too short"
    page = _page(text, normalized_text_length=68)

    result = classify_page(page, RULES)
    assert result.strong_signature_matched is False
    assert result.matched_class_count == 0

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is True
    # Multi-reason collection: zero matches means rule_score is also 0.0, so
    # LOW_RULE_SCORE fires alongside SHORT_TEXT now that steps 3-5 all run.
    assert WarningCode.SHORT_TEXT in reasons
    assert WarningCode.LOW_RULE_SCORE in reasons
    # rule_margin is 0.0 here (< the 0.20 threshold) but matched_class_count
    # is 0, so the require_multiple_classes_for_margin_check gate must keep
    # NARROW_RULE_MARGIN from firing on absent evidence.
    assert WarningCode.NARROW_RULE_MARGIN not in reasons


def test_delegation_reasons_exclude_classification_warnings() -> None:
    # classify_page's own observation (NO_SIGNATURE) must not leak into
    # should_delegate's reasons -- the two answer different questions ("what
    # did the rule stage conclude" vs "why was this page delegated").
    text = "too short"
    page = _page(text, normalized_text_length=68)

    result = classify_page(page, RULES)
    assert WarningCode.NO_SIGNATURE in result.warnings

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is True
    assert WarningCode.NO_SIGNATURE not in reasons
    assert reasons[0] == WarningCode.SHORT_TEXT


def test_conflicting_strong_signatures_are_delegated() -> None:
    text = (
        "Uniform Residential Loan Application appears alongside XACTUS "
        "on this malformed page " + "filler " * 20
    )
    page = _page(text, normalized_text_length=200)

    result = classify_page(page, RULES)
    # both classes fired a strong signature
    assert result.doc_type in (DocType.URLA_1003, DocType.CREDIT_REPORT)

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is True
    assert reasons == [WarningCode.CONFLICTING_SIGNALS]


def test_should_delegate_true_always_carries_a_reason() -> None:
    for text, length in [
        ("Net Income " + "filler " * 20, 150),
        ("nothing relevant " + "filler " * 20, 200),
        ("too short", 68),
    ]:
        page = _page(text, normalized_text_length=length)
        result = classify_page(page, RULES)
        delegate, reasons = should_delegate(result, RULES)
        if delegate:
            assert reasons, f"delegate=True with empty reasons for text={text!r}"


def test_narrow_margin_fires_on_real_class_conflict() -> None:
    # One weak signal from each of two different classes, tied in weight, so
    # share(top1) == share(top2) == 0.5 and rule_margin == 0.0 (< 0.20). This
    # is the genuine "two classes are actually contending" case that
    # NARROW_RULE_MARGIN exists to describe.
    text = "This page mentions Net Income alongside Tradeline history " + "filler " * 15
    page = _page(text, normalized_text_length=150)

    result = classify_page(page, RULES)
    assert result.matched_class_count == 2
    assert result.rule_margin < 0.20

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is True
    assert WarningCode.NARROW_RULE_MARGIN in reasons


def test_narrow_margin_does_not_fire_without_conflict() -> None:
    # No signature matches at all (matched_class_count == 0) but the text is
    # long enough that SHORT_TEXT should not fire. rule_margin is 0.0 here
    # purely because there is no evidence, not because two classes tied --
    # NARROW_RULE_MARGIN must not claim a contest that never happened.
    text = "Completely unrelated boilerplate text " + "filler " * 20
    page = _page(text, normalized_text_length=200)

    result = classify_page(page, RULES)
    assert result.matched_class_count == 0

    delegate, reasons = should_delegate(result, RULES)
    assert delegate is True
    assert WarningCode.LOW_RULE_SCORE in reasons
    assert WarningCode.NARROW_RULE_MARGIN not in reasons


def test_validate_never_strong_raises_on_violation() -> None:
    bad_rules = {
        "never_strong": ["Borrower"],
        "doc_types": {
            "URLA_1003": {"strong": ["Borrower"], "weak": []},
        },
    }
    with pytest.raises(ValueError):
        validate_never_strong(bad_rules)


def test_validate_never_strong_passes_on_clean_config() -> None:
    validate_never_strong(RULES)
