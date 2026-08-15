"""Tests for scripts/build_ground_truth.py. No real PDF is used -- the hash
index and validation functions are pure and are exercised with inline
strings standing in for page text."""

from __future__ import annotations

import argparse

import pytest

from scripts.build_ground_truth import (
    build_source_index,
    check_page_count,
    check_page_numbers_complete,
    match_shuffled_pages,
    parse_source_arg,
)


def test_normal_matching_recovers_label_and_source_page() -> None:
    source_texts = {
        "URLA_1003": ["urla page one", "urla page two"],
        "CREDIT_REPORT": ["credit page one"],
    }
    # Shuffled order deliberately does not match source order.
    shuffled_texts = ["credit page one", "urla page two", "urla page one"]

    index = build_source_index(source_texts)
    matches = match_shuffled_pages(shuffled_texts, index)

    assert matches == [
        (1, "CREDIT_REPORT", 1),
        (2, "URLA_1003", 2),
        (3, "URLA_1003", 1),
    ]


def test_hash_collision_in_sources_is_detected() -> None:
    source_texts = {
        "URLA_1003": ["duplicate content"],
        "CREDIT_REPORT": ["duplicate content"],
    }
    with pytest.raises(ValueError, match="collision"):
        build_source_index(source_texts)


def test_unmatched_shuffled_page_is_detected() -> None:
    source_texts = {"URLA_1003": ["known page"]}
    index = build_source_index(source_texts)
    shuffled_texts = ["known page", "never seen before"]

    with pytest.raises(ValueError, match=r"(?i)unmatched|no matching|matched no"):
        match_shuffled_pages(shuffled_texts, index)


def test_multi_matched_source_page_is_detected() -> None:
    source_texts = {"URLA_1003": ["only page"]}
    index = build_source_index(source_texts)
    # Two shuffled pages both hash to the same single source page.
    shuffled_texts = ["only page", "only page"]

    with pytest.raises(ValueError, match=r"(?i)more than one"):
        match_shuffled_pages(shuffled_texts, index)


def test_page_count_mismatch_is_detected() -> None:
    with pytest.raises(ValueError, match=r"(?i)page count"):
        check_page_count(total_source_pages=3, shuffled_page_count=4)


def test_page_count_match_passes() -> None:
    check_page_count(total_source_pages=4, shuffled_page_count=4)


def test_check_page_numbers_complete_passes_on_full_range() -> None:
    matches = [(1, "URLA_1003", 1), (2, "CREDIT_REPORT", 1), (3, "URLA_1003", 2)]
    check_page_numbers_complete(matches, expected_count=3)


def test_check_page_numbers_complete_rejects_gap() -> None:
    matches = [(1, "URLA_1003", 1), (3, "URLA_1003", 2)]
    with pytest.raises(ValueError):
        check_page_numbers_complete(matches, expected_count=3)


def test_parse_source_arg_accepts_valid_label_and_splits_once() -> None:
    label, path = parse_source_arg("TITLE_REPORT=data/pkg & co/file=v2.pdf")
    assert label == "TITLE_REPORT"
    assert path == "data/pkg & co/file=v2.pdf"


def test_parse_source_arg_rejects_unknown_label() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_source_arg("NOT_A_DOC_TYPE=data/file.pdf")


def test_parse_source_arg_rejects_missing_equals() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_source_arg("URLA_1003_no_equals_sign")
