"""Tests for src/extract.py. No real PDF is required -- detect_marker,
extract_identifiers and normalize_text are pure functions over text and the
real rules.yaml, so they can be exercised directly."""

from __future__ import annotations

import pytest

from src.extract import detect_marker, extract_identifiers, load_rules, normalize_text
from src.schema import MarkerStyle


@pytest.fixture(scope="module")
def rules() -> dict:
    return load_rules("config/rules.yaml")


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------


def test_normalize_text_collapses_mixed_whitespace() -> None:
    text = "  Loan\tNumber: \n\n 123456   \r\n  more\ttext  "
    assert normalize_text(text) == "LoanNumber:123456moretext"


def test_normalize_text_empty_string() -> None:
    assert normalize_text("") == ""


def test_normalize_text_whitespace_only() -> None:
    assert normalize_text("   \n\t\r\n  ") == ""


# ---------------------------------------------------------------------------
# detect_marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Page 7 of 11", (MarkerStyle.PAGE_N_OF_M, 7, 11)),
        ("5 of 11", (MarkerStyle.N_OF_M, 5, 11)),
        (
            "CLTA Preliminary Report Form (02/03/2023) Printed: 08.10.23 @ 02:15 PM "
            "by AB Page 4 CA-FT-FITL-02090.235381-SPS-1-23-A-3609053",
            (MarkerStyle.CLTA_PAGE_N, 4, None),
        ),
        ("[ Plat map removed in anonymized sample ]", (MarkerStyle.NONE, None, None)),
    ],
)
def test_detect_marker(rules: dict, text: str, expected: tuple) -> None:
    assert detect_marker(text, rules) == expected


def test_detect_marker_priority_page_n_of_m_beats_n_of_m(rules: dict) -> None:
    """Regression guard: "Page 7 of 11" also satisfies the generic N_OF_M
    pattern, so PAGE_N_OF_M (lower priority number = tried first) must win."""
    style, page, total = detect_marker("Page 7 of 11", rules)
    assert style is MarkerStyle.PAGE_N_OF_M
    assert style is not MarkerStyle.N_OF_M
    assert (page, total) == (7, 11)


def test_detect_marker_rejects_page_greater_than_total(rules: dict) -> None:
    """page > total must be rejected under every candidate pattern (both
    PAGE_N_OF_M and N_OF_M match "Page 11 of 7"), falling through to NONE
    rather than returning an inconsistent marker."""
    assert detect_marker("Page 11 of 7", rules) == (MarkerStyle.NONE, None, None)


def test_detect_marker_rejects_total_over_max(rules: dict) -> None:
    max_total = rules["marker_constraints"]["max_total"]
    text = f"Page 3 of {max_total + 1}"
    style, page, total = detect_marker(text, rules)
    assert style is MarkerStyle.NONE
    assert page is None
    assert total is None


# ---------------------------------------------------------------------------
# extract_identifiers
# ---------------------------------------------------------------------------


def test_extract_identifiers_loan_number_key_present(rules: dict) -> None:
    text = "Borrower Information\nLoan Number: 123456789\nSection 1a."
    identifiers = extract_identifiers(text, rules)
    assert "loan_number" in identifiers
    assert "prelim_number" not in identifiers


def test_extract_identifiers_prelim_number_key_present(rules: dict) -> None:
    text = "CLTA Preliminary Report Form\nPRELIM NO.: 2024-001234\nEXHIBIT A"
    identifiers = extract_identifiers(text, rules)
    assert "prelim_number" in identifiers
    assert "loan_number" not in identifiers


def test_extract_identifiers_no_match_returns_empty_dict(rules: dict) -> None:
    text = "This page has no configured identifiers on it at all."
    assert extract_identifiers(text, rules) == {}
