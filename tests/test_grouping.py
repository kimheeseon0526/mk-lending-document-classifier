"""Tests for src/grouping.py. No real PDF is used -- PageResult instances
are built inline and fed directly to the grouping functions."""

from __future__ import annotations

from src.grouping import (
    _physical_span_and_contiguity,
    build_grouping_key,
    build_physical_groups,
    reconstruct_documents,
)
from src.schema import (
    Completeness,
    DecisionSource,
    DocType,
    GroupIssue,
    MarkerStyle,
    PageResult,
    WarningCode,
)


def _result(
    page_number: int,
    doc_type: DocType,
    marker_style: MarkerStyle = MarkerStyle.NONE,
    marker_page: int | None = None,
    marker_total: int | None = None,
    instance_id: str | None = None,
) -> PageResult:
    return PageResult(
        page_number=page_number,
        doc_type=doc_type,
        decision_source=DecisionSource.RULE,
        marker_style=marker_style,
        marker_page=marker_page,
        marker_total=marker_total,
        instance_id=instance_id,
    )


# ---------------------------------------------------------------------------
# build_physical_groups
# ---------------------------------------------------------------------------


def test_physical_groups_consecutive_same_type_form_one_group() -> None:
    results = [
        _result(1, DocType.URLA_1003),
        _result(2, DocType.URLA_1003),
        _result(3, DocType.URLA_1003),
    ]
    groups = build_physical_groups(results)

    assert len(groups) == 1
    assert (groups[0].start_page, groups[0].end_page, groups[0].page_count) == (1, 3, 3)
    assert groups[0].physical_pages == [1, 2, 3]


def test_physical_groups_fully_interleaved_stay_separate() -> None:
    # Regression guard: a scatter-shuffled package with zero adjacent pairs
    # of the same doc_type must yield one group per page, not be merged.
    types = [
        DocType.URLA_1003,
        DocType.INCOME_DOC,
        DocType.CREDIT_REPORT,
        DocType.TITLE_REPORT,
        DocType.OTHER,
    ]
    results = [_result(i + 1, t) for i, t in enumerate(types)]

    groups = build_physical_groups(results)

    assert len(groups) == 5
    assert all(g.page_count == 1 for g in groups)


def test_physical_groups_empty_input_returns_empty_list() -> None:
    assert build_physical_groups([]) == []


# ---------------------------------------------------------------------------
# build_grouping_key
# ---------------------------------------------------------------------------


def test_grouping_key_differs_by_doc_type_despite_same_marker_total() -> None:
    urla = _result(1, DocType.URLA_1003, marker_style=MarkerStyle.N_OF_M, marker_total=11)
    credit = _result(
        2, DocType.CREDIT_REPORT, marker_style=MarkerStyle.PAGE_N_OF_M, marker_total=11
    )
    assert build_grouping_key(urla) != build_grouping_key(credit)


def test_grouping_key_differs_by_marker_style() -> None:
    a = _result(1, DocType.CREDIT_REPORT, marker_style=MarkerStyle.PAGE_N_OF_M, marker_total=11)
    b = _result(2, DocType.CREDIT_REPORT, marker_style=MarkerStyle.N_OF_M, marker_total=11)
    assert build_grouping_key(a) != build_grouping_key(b)


# ---------------------------------------------------------------------------
# reconstruct_documents
# ---------------------------------------------------------------------------


def test_reconstruct_complete_document() -> None:
    results = [
        _result(
            p,
            DocType.TITLE_REPORT,
            marker_style=MarkerStyle.CLTA_PAGE_N,
            marker_page=p,
            marker_total=5,
            instance_id="TITLE_REPORT#1",
        )
        for p in range(1, 6)
    ]
    docs = reconstruct_documents(results, {})

    assert len(docs) == 1
    assert docs[0].completeness == Completeness.COMPLETE
    assert docs[0].missing_logical_pages == []
    assert docs[0].observed_logical_pages == [1, 2, 3, 4, 5]


def test_reconstruct_incomplete_document_lists_missing_pages() -> None:
    results = [
        _result(
            idx,
            DocType.TITLE_REPORT,
            marker_style=MarkerStyle.CLTA_PAGE_N,
            marker_page=marker_page,
            marker_total=5,
            instance_id="TITLE_REPORT#1",
        )
        for idx, marker_page in enumerate([1, 2, 4], start=1)
    ]
    docs = reconstruct_documents(results, {})

    assert len(docs) == 1
    assert docs[0].completeness == Completeness.INCOMPLETE
    assert docs[0].missing_logical_pages == [3, 5]


def test_reconstruct_no_marker_total_is_unknown_extent() -> None:
    results = [
        _result(
            1,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=1,
            marker_total=None,
            instance_id="CREDIT_REPORT#1",
        )
    ]
    docs = reconstruct_documents(results, {})

    assert docs[0].completeness == Completeness.UNKNOWN_EXTENT
    assert docs[0].missing_logical_pages == []


def test_reconstruct_duplicate_marker_page_keeps_both_and_flags() -> None:
    results = [
        _result(
            1,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=1,
            marker_total=11,
            instance_id="CREDIT_REPORT#1",
        ),
        _result(
            2,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=1,
            marker_total=11,
            instance_id="CREDIT_REPORT#1",
        ),
    ]
    docs = reconstruct_documents(results, {})

    assert len(docs) == 1
    doc = docs[0]
    assert doc.duplicate_logical_pages == [1]
    assert GroupIssue.DUPLICATE_PAGE in doc.issues
    assert len(doc.page_links) == 2
    assert {pl.physical_page for pl in doc.page_links} == {1, 2}


def test_reconstruct_page_with_no_marker_is_orphan_and_unattached() -> None:
    result = _result(1, DocType.OTHER, marker_style=MarkerStyle.NONE)
    docs = reconstruct_documents([result], {})

    assert docs == []
    assert result.is_orphan is True
    assert WarningCode.ORPHAN_PAGE in result.warnings


def test_reconstruct_conflicting_marker_total_splits_into_two_documents() -> None:
    results = [
        _result(
            1,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=1,
            marker_total=11,
            instance_id="CREDIT_REPORT#1",
        ),
        _result(
            2,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=1,
            marker_total=12,
            instance_id="CREDIT_REPORT#1",
        ),
    ]
    docs = reconstruct_documents(results, {})

    assert len(docs) == 2
    assert all(GroupIssue.CONFLICTING_MARKER_TOTAL in d.issues for d in docs)


def test_reconstruct_start_physical_page_is_the_physical_page_of_logical_one() -> None:
    results = [
        _result(
            30,
            DocType.TITLE_REPORT,
            marker_style=MarkerStyle.CLTA_PAGE_N,
            marker_page=3,
            marker_total=3,
            instance_id="TITLE_REPORT#1",
        ),
        _result(
            10,
            DocType.TITLE_REPORT,
            marker_style=MarkerStyle.CLTA_PAGE_N,
            marker_page=1,
            marker_total=3,
            instance_id="TITLE_REPORT#1",
        ),
        _result(
            20,
            DocType.TITLE_REPORT,
            marker_style=MarkerStyle.CLTA_PAGE_N,
            marker_page=2,
            marker_total=3,
            instance_id="TITLE_REPORT#1",
        ),
    ]
    docs = reconstruct_documents(results, {})

    assert len(docs) == 1
    assert docs[0].start_physical_page == 10
    assert docs[0].end_physical_page == 30


def test_first_page_can_appear_after_last_page() -> None:
    # Not a bug: in a scatter-shuffled package, a document's logical first
    # page can land physically after its logical last page. start_physical_page
    # (physical position of logical page 1) and end_physical_page (physical
    # position of the last observed logical page) are correct exactly as
    # defined even when start_physical_page > end_physical_page.
    results = [
        _result(
            9,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=1,
            marker_total=3,
            instance_id="CREDIT_REPORT#1",
        ),
        _result(
            5,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=2,
            marker_total=3,
            instance_id="CREDIT_REPORT#1",
        ),
        _result(
            2,
            DocType.CREDIT_REPORT,
            marker_style=MarkerStyle.PAGE_N_OF_M,
            marker_page=3,
            marker_total=3,
            instance_id="CREDIT_REPORT#1",
        ),
    ]
    docs = reconstruct_documents(results, {})

    assert len(docs) == 1
    assert docs[0].start_physical_page == 9
    assert docs[0].end_physical_page == 2
    assert docs[0].start_physical_page > docs[0].end_physical_page


# ---------------------------------------------------------------------------
# CLI display helper: _physical_span_and_contiguity
# ---------------------------------------------------------------------------


def test_physical_span_and_contiguity_for_scattered_pages() -> None:
    span, is_contiguous = _physical_span_and_contiguity([9, 5, 2])
    assert span == "2-9"
    assert is_contiguous is False


def test_physical_span_and_contiguity_for_contiguous_pages() -> None:
    span, is_contiguous = _physical_span_and_contiguity([12, 10, 11])
    assert span == "10-12"
    assert is_contiguous is True
