"""Tests for src/pipeline.py and the CSV builders in scripts/run.py. No real
PDF is used -- src.pipeline.extract_pages is monkeypatched to return inline
ExtractedPage instances."""

from __future__ import annotations

import pytest

import src.pipeline as pipeline_module
from src.extract import load_rules
from src.pipeline import check_no_pii, run_pipeline
from src.schema import (
    DecisionSource,
    DocType,
    ExtractedPage,
    GroupingKey,
    LLMBatchResponse,
    LLMPageDecision,
    LLMRuntimeConfig,
    MarkerStyle,
    PageLink,
    PageResult,
    PhysicalGroup,
    PipelineMode,
    ReconstructedDocument,
    WarningCode,
)
from scripts.run import _page_results_csv, _physical_groups_csv, _reconstructed_documents_csv


@pytest.fixture(scope="module")
def rules() -> dict:
    return load_rules("config/rules.yaml")


def _pages() -> list[ExtractedPage]:
    confident = ExtractedPage(
        page_number=1,
        text="This page is a Uniform Residential Loan Application " + "filler " * 20,
        normalized_text_length=200,
        rotation=0,
        image_count=0,
        marker_style=MarkerStyle.N_OF_M,
        marker_page=1,
        marker_total=1,
    )
    needs_llm = ExtractedPage(
        page_number=2,
        text="no useful keywords here at all, just filler content",
        normalized_text_length=50,
        rotation=0,
        image_count=0,
    )
    return [confident, needs_llm]


def _patch_extract(monkeypatch, pages: list[ExtractedPage]) -> None:
    monkeypatch.setattr(pipeline_module, "extract_pages", lambda pdf_path, rules: pages)


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


def test_rule_only_makes_no_llm_calls_but_records_targets(rules: dict, monkeypatch) -> None:
    _patch_extract(monkeypatch, _pages())
    config = LLMRuntimeConfig(enabled=False)

    result = run_pipeline("fake.pdf", rules, PipelineMode.RULE_ONLY, config)
    report = result.run_report

    assert report.llm_called == 0
    assert 2 in report.llm_target_pages
    assert 1 not in report.llm_target_pages
    assert all(r.decision_source is DecisionSource.RULE for r in result.page_results)


def test_hybrid_with_disabled_llm_completes_without_raising(rules: dict, monkeypatch) -> None:
    _patch_extract(monkeypatch, _pages())
    config = LLMRuntimeConfig(enabled=False)

    result = run_pipeline("fake.pdf", rules, PipelineMode.HYBRID, config)
    report = result.run_report

    assert report.mode is PipelineMode.HYBRID
    assert report.llm_called == 0
    assert WarningCode.LLM_DISABLED in report.warnings


def test_llm_only_with_disabled_llm_raises_clear_error(rules: dict, monkeypatch) -> None:
    _patch_extract(monkeypatch, _pages())
    config = LLMRuntimeConfig(enabled=False)

    with pytest.raises(ValueError, match=r"(?i)LLM_ONLY.*enabled"):
        run_pipeline("fake.pdf", rules, PipelineMode.LLM_ONLY, config)


def test_run_report_count_consistency_with_successful_llm_call(rules: dict, monkeypatch) -> None:
    _patch_extract(monkeypatch, _pages())

    def _fake_call(batch, config):
        return LLMBatchResponse(
            decisions=[
                LLMPageDecision(page_id=req.page_id, doc_type=DocType.INCOME_DOC, confidence=0.7)
                for req in batch
            ]
        )

    monkeypatch.setattr("src.llm_classifier.call_llm", _fake_call)
    config = LLMRuntimeConfig(enabled=True, batch_size=10)

    result = run_pipeline("fake.pdf", rules, PipelineMode.HYBRID, config)
    report = result.run_report

    assert report.llm_resolved + report.rule_fallback_count <= report.llm_called
    assert report.llm_resolved + report.rule_fallback_count == report.llm_called


def test_source_pdf_name_has_no_path_separators(rules: dict, monkeypatch) -> None:
    _patch_extract(monkeypatch, _pages())
    config = LLMRuntimeConfig(enabled=False)

    result = run_pipeline(
        "data/package_01/some folder & path/01.990145627_shuffled.pdf",
        rules,
        PipelineMode.RULE_ONLY,
        config,
    )

    name = result.run_report.source_pdf_name
    assert "/" not in name
    assert "\\" not in name
    assert name == "01.990145627_shuffled.pdf"


# ---------------------------------------------------------------------------
# check_no_pii
# ---------------------------------------------------------------------------


def test_check_no_pii_rejects_ssn() -> None:
    with pytest.raises(ValueError):
        check_no_pii("some text containing 123-45-6789 an SSN")


def test_check_no_pii_rejects_email() -> None:
    with pytest.raises(ValueError):
        check_no_pii("contact john.doe@example.com for details")


def test_check_no_pii_passes_clean_text() -> None:
    check_no_pii("page_number,doc_type\n1,URLA_1003\n")


# ---------------------------------------------------------------------------
# CSV column shape (scripts/run.py)
# ---------------------------------------------------------------------------


def test_page_results_csv_header_matches_spec() -> None:
    result = PageResult(
        page_number=1, doc_type=DocType.URLA_1003, decision_source=DecisionSource.RULE
    )
    content = _page_results_csv([result])
    header = content.splitlines()[0].split(",")

    assert header == [
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


def test_physical_groups_csv_header_matches_spec() -> None:
    group = PhysicalGroup(
        group_id="PG001",
        doc_type=DocType.URLA_1003,
        start_page=1,
        end_page=1,
        page_count=1,
        physical_pages=[1],
    )
    content = _physical_groups_csv([group])
    header = content.splitlines()[0].split(",")

    assert header == [
        "group_id",
        "doc_type",
        "start_page",
        "end_page",
        "page_count",
        "physical_pages",
    ]


def test_reconstructed_documents_csv_header_matches_spec() -> None:
    doc = ReconstructedDocument(
        doc_id="RD001",
        doc_type=DocType.URLA_1003,
        key=GroupingKey(
            doc_type=DocType.URLA_1003,
            instance_id="URLA_1003#1",
            marker_style=MarkerStyle.N_OF_M,
            marker_total=1,
        ),
        expected_pages=1,
        observed_logical_pages=[1],
        missing_logical_pages=[],
        duplicate_logical_pages=[],
        page_links=[PageLink(physical_page=1, logical_page=1, evidence=["identifier_and_marker_match"])],
        start_physical_page=1,
        end_physical_page=1,
        completeness="COMPLETE",
        issues=[],
        warnings=[],
    )
    content = _reconstructed_documents_csv([doc])
    header = content.splitlines()[0].split(",")

    assert header == [
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
