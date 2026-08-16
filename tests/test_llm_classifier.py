"""Tests for src/llm_classifier.py. No real PDF and no real API call --
call_llm is monkeypatched everywhere it matters."""

from __future__ import annotations

import pytest

import src.llm_classifier as llm_classifier
from src.extract import load_rules
from src.llm_classifier import (
    build_batches,
    build_request,
    classify_pages,
    load_llm_config,
    mask_text,
    validate_response,
)
from src.rule_classifier import classify_page
from src.schema import (
    DecisionSource,
    DocType,
    ExtractedPage,
    LLMBatchResponse,
    LLMPageDecision,
    LLMPageRequest,
    LLMRuntimeConfig,
    MarkerStyle,
    WarningCode,
)


@pytest.fixture(scope="module")
def rules() -> dict:
    return load_rules("config/rules.yaml")


def _page(text: str, normalized_text_length: int, page_number: int = 1) -> ExtractedPage:
    return ExtractedPage(
        page_number=page_number,
        text=text,
        normalized_text_length=normalized_text_length,
        rotation=0,
        image_count=0,
    )


# ---------------------------------------------------------------------------
# mask_text
# ---------------------------------------------------------------------------


def test_mask_text_removes_ssn_email_phone(rules: dict) -> None:
    text = "SSN: 123-45-6789, email: john.doe@example.com, phone: (555) 123-4567"
    masked = mask_text(text, rules)

    assert "123-45-6789" not in masked
    assert "john.doe@example.com" not in masked
    assert "123-4567" not in masked
    assert "[SSN]" in masked
    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked


def test_mask_text_preserves_dollar_amounts(rules: dict) -> None:
    # Required regression: package_01's P&L page has no standard income
    # keyword anywhere on it -- dollar-amount structure is part of its only
    # classification signal, alongside "Realtor" and "CTEC".
    text = "Total Expenses: $231,239.00 and Net Income: 49,720.00 this month."
    masked = mask_text(text, rules)

    assert "$231,239.00" in masked
    assert "49,720.00" in masked


def test_mask_text_preserves_realtor(rules: dict) -> None:
    text = "Borrower Name: Jane Smith Realtor with CTEC #A183652 filed this Profit & Loss Statement."
    masked = mask_text(text, rules)

    assert "Realtor" in masked
    assert "Jane Smith" not in masked
    assert "[NAME]" in masked


def test_mask_text_ctec_number_keeps_label_masks_id(rules: dict) -> None:
    text = "Preparer CTEC #A183652 signed the return."
    masked = mask_text(text, rules)

    assert "CTEC" in masked
    assert "A183652" not in masked
    assert "[ID]" in masked


def test_mask_text_redacts_name_near_label(rules: dict) -> None:
    text = "Applicant: Robert Johnson resides at the property."
    masked = mask_text(text, rules)

    assert "Robert Johnson" not in masked
    assert "[NAME]" in masked


# ---------------------------------------------------------------------------
# load_llm_config
# ---------------------------------------------------------------------------


def test_load_llm_config_without_key_is_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_PROVIDER=openai\n", encoding="utf-8")

    config = load_llm_config(str(env_file))

    assert config.enabled is False
    assert "api_key" not in config.model_dump()


def test_load_llm_config_with_key_is_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=sk-test-not-a-real-key\n", encoding="utf-8")

    config = load_llm_config(str(env_file))

    assert config.enabled is True
    assert "api_key" not in config.model_dump()


# ---------------------------------------------------------------------------
# validate_response
# ---------------------------------------------------------------------------


def test_validate_response_count_mismatch_when_response_is_short() -> None:
    batch = [
        LLMPageRequest(page_id=1, excerpt="a"),
        LLMPageRequest(page_id=2, excerpt="b"),
        LLMPageRequest(page_id=3, excerpt="c"),
    ]
    response = LLMBatchResponse(
        decisions=[
            LLMPageDecision(page_id=1, doc_type=DocType.URLA_1003, confidence=0.9),
            LLMPageDecision(page_id=2, doc_type=DocType.CREDIT_REPORT, confidence=0.8),
        ]
    )

    validated, warnings = validate_response(response, batch)

    assert WarningCode.LLM_COUNT_MISMATCH in warnings
    assert {d.page_id for d in validated.decisions} == {1, 2}


def test_validate_response_drops_invalid_label() -> None:
    batch = [LLMPageRequest(page_id=1, excerpt="a"), LLMPageRequest(page_id=2, excerpt="b")]
    bad = LLMPageDecision.model_construct(
        page_id=1, doc_type="NOT_A_REAL_TYPE", confidence=0.9, evidence=""
    )
    good = LLMPageDecision(page_id=2, doc_type=DocType.CREDIT_REPORT, confidence=0.7)
    response = LLMBatchResponse.model_construct(decisions=[bad, good])

    validated, warnings = validate_response(response, batch)

    assert WarningCode.LLM_LABEL_INVALID in warnings
    page_ids = {d.page_id for d in validated.decisions}
    assert 1 not in page_ids
    assert 2 in page_ids


def test_validate_response_drops_out_of_range_confidence() -> None:
    batch = [LLMPageRequest(page_id=1, excerpt="a")]
    bad = LLMPageDecision.model_construct(
        page_id=1, doc_type=DocType.URLA_1003, confidence=1.5, evidence=""
    )
    response = LLMBatchResponse.model_construct(decisions=[bad])

    validated, warnings = validate_response(response, batch)

    assert validated.decisions == []
    assert WarningCode.LLM_LABEL_INVALID in warnings


def test_validate_response_drops_unrequested_page_id_without_warning() -> None:
    batch = [LLMPageRequest(page_id=1, excerpt="a")]
    response = LLMBatchResponse(
        decisions=[
            LLMPageDecision(page_id=1, doc_type=DocType.URLA_1003, confidence=0.9),
            LLMPageDecision(page_id=99, doc_type=DocType.OTHER, confidence=0.5),
        ]
    )

    validated, warnings = validate_response(response, batch)

    assert {d.page_id for d in validated.decisions} == {1}
    assert WarningCode.LLM_COUNT_MISMATCH not in warnings


def test_validate_response_unparseable_shape_is_parse_failed() -> None:
    batch = [LLMPageRequest(page_id=1, excerpt="a")]

    validated, warnings = validate_response("not a response at all", batch)

    assert validated.decisions == []
    assert warnings == [WarningCode.LLM_PARSE_FAILED]


# ---------------------------------------------------------------------------
# classify_pages orchestration
# ---------------------------------------------------------------------------


def _build_pages_and_rule_results(rules: dict) -> tuple[list[ExtractedPage], list]:
    delegate_page = _page(
        "no useful keywords on this page at all, just filler content",
        normalized_text_length=50,
        page_number=1,
    )
    confident_page = _page(
        "This is the Uniform Residential Loan Application form. " * 3,
        normalized_text_length=300,
        page_number=2,
    )
    pages = [delegate_page, confident_page]
    rule_results = [classify_page(p, rules) for p in pages]
    return pages, rule_results


def test_classify_pages_disabled_makes_no_calls(rules: dict, monkeypatch) -> None:
    calls = []

    def _record_call(batch, config):
        calls.append(batch)
        raise AssertionError("call_llm should not be invoked when config.enabled is False")

    monkeypatch.setattr(llm_classifier, "call_llm", _record_call)

    pages, rule_results = _build_pages_and_rule_results(rules)
    config = LLMRuntimeConfig(enabled=False)

    resolved, records = classify_pages(pages, rule_results, rules, config)

    assert calls == []
    assert records == []

    by_page = {r.page_number: r for r in resolved}
    assert by_page[2].decision_source == DecisionSource.RULE
    assert WarningCode.LLM_DISABLED not in by_page[2].warnings

    assert by_page[1].decision_source == DecisionSource.RULE
    assert WarningCode.LLM_DISABLED in by_page[1].warnings


def test_classify_pages_retries_then_falls_back_to_rule(rules: dict, monkeypatch) -> None:
    call_count = {"n": 0}

    def _always_fail(batch, config):
        call_count["n"] += 1
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(llm_classifier, "call_llm", _always_fail)

    pages, rule_results = _build_pages_and_rule_results(rules)
    original_doc_type = next(r for r in rule_results if r.page_number == 1).doc_type
    config = LLMRuntimeConfig(enabled=True, max_retries=2, batch_size=10)

    resolved, records = classify_pages(pages, rule_results, rules, config)

    assert call_count["n"] == config.max_retries + 1

    by_page = {r.page_number: r for r in resolved}
    fallback = by_page[1]
    assert fallback.decision_source == DecisionSource.RULE_FALLBACK
    assert fallback.doc_type == original_doc_type
    assert fallback.llm_called is True
    assert fallback.llm_failed is True
    assert WarningCode.LLM_CALL_FAILED in fallback.warnings

    assert len(records) == 1
    assert records[0].succeeded is False
    assert records[0].attempt_count == config.max_retries + 1


def test_classify_pages_success_updates_doc_type(rules: dict, monkeypatch) -> None:
    def _fake_call(batch, config):
        return LLMBatchResponse(
            decisions=[
                LLMPageDecision(page_id=req.page_id, doc_type=DocType.INCOME_DOC, confidence=0.83)
                for req in batch
            ]
        )

    monkeypatch.setattr(llm_classifier, "call_llm", _fake_call)

    pages, rule_results = _build_pages_and_rule_results(rules)
    config = LLMRuntimeConfig(enabled=True, batch_size=10)

    resolved, records = classify_pages(pages, rule_results, rules, config)

    by_page = {r.page_number: r for r in resolved}
    result = by_page[1]
    assert result.decision_source == DecisionSource.LLM
    assert result.doc_type == DocType.INCOME_DOC
    assert result.llm_doc_type == DocType.INCOME_DOC
    assert result.llm_confidence == pytest.approx(0.83)
    assert result.llm_called is True
    assert result.llm_failed is False

    assert len(records) == 1
    assert records[0].succeeded is True


# ---------------------------------------------------------------------------
# build_request / build_batches
# ---------------------------------------------------------------------------


def test_build_request_masks_and_sets_marker_hint(rules: dict) -> None:
    page = ExtractedPage(
        page_number=5,
        text="SSN: 123-45-6789 on this page.",
        normalized_text_length=30,
        rotation=0,
        image_count=0,
        marker_style=MarkerStyle.PAGE_N_OF_M,
        marker_page=4,
        marker_total=11,
    )
    request = build_request(page, rules, max_chars=200)

    assert request.page_id == 5
    assert "123-45-6789" not in request.excerpt
    assert request.marker_hint == "page 4 of 11"


def test_build_request_no_marker_gives_none_hint(rules: dict) -> None:
    page = _page("plain text with no marker", normalized_text_length=25)
    request = build_request(page, rules, max_chars=200)
    assert request.marker_hint is None


def test_build_batches_splits_by_size() -> None:
    requests = [LLMPageRequest(page_id=i, excerpt="x") for i in range(1, 6)]
    batches = build_batches(requests, batch_size=2)

    assert [len(b) for b in batches] == [2, 2, 1]
    assert [r.page_id for r in batches[0]] == [1, 2]
    assert [r.page_id for r in batches[2]] == [5]
