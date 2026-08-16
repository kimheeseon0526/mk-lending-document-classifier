"""End-to-end orchestration: extract -> rule_classifier -> llm_classifier ->
grouping, for one PDF and one `PipelineMode`.

PII BOUNDARY: `run_pipeline` touches raw page text only indirectly, by
calling `extract.extract_pages` and passing its output straight into
`rule_classifier`/`llm_classifier`/`grouping`, all of which already respect
the PII boundary on their own. Nothing here reads `ExtractedPage.text` or
`raw_identifiers` directly. `PipelineResult` and everything inside it
(`PageResult`, `PhysicalGroup`, `ReconstructedDocument`, `RunReport`) are
schema types already defined to be safe to persist.

This module does not re-implement scoring, masking, batching, retrying, or
grouping -- it only sequences the existing module functions and assembles
the run-level report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.extract import extract_pages
from src.grouping import group_pages
from src.llm_classifier import classify_pages, resolve_delegated_pages
from src.rule_classifier import classify_page, should_delegate
from src.schema import (
    DecisionSource,
    LLMCallRecord,
    LLMRuntimeConfig,
    PageResult,
    PhysicalGroup,
    PipelineMode,
    ReconstructedDocument,
    RunReport,
    WarningCode,
)


@dataclass
class PipelineResult:
    """Everything one `run_pipeline` call produces. Not a schema type --
    `schema.py` defines the individual pieces (`PageResult`, `PhysicalGroup`,
    `ReconstructedDocument`, `RunReport`); this is just a local container for
    passing all four back together without inventing a new persisted type.
    """

    page_results: list[PageResult]
    physical_groups: list[PhysicalGroup]
    reconstructed_documents: list[ReconstructedDocument]
    run_report: RunReport


def run_pipeline(
    pdf_path: str | Path,
    rules: dict,
    mode: PipelineMode,
    llm_config: LLMRuntimeConfig,
) -> PipelineResult:
    """Run the full pipeline for one PDF under one mode.

    Steps: `extract.extract_pages` -> `rule_classifier.classify_page` per
    page -> mode-dependent LLM handling -> `grouping.group_pages` -> a fully
    populated `RunReport`. Every step reuses the existing module function
    unchanged; this function only sequences them and counts outcomes.

    Mode behavior:
    - RULE_ONLY: no LLM call is made. Pages `should_delegate` flags are
      still recorded in `RunReport.llm_target_pages` (informational -- what
      *would* be delegated), but every page's final `doc_type` stays the
      rule result.
    - HYBRID: only pages `should_delegate` flags are sent to the LLM (via
      `llm_classifier.classify_pages`, which applies `should_delegate`
      itself). If `llm_config.enabled` is False, this degrades to the same
      outcome as RULE_ONLY (no calls, no exception) but keeps `mode` at
      HYBRID and records `WarningCode.LLM_DISABLED` on the report so the
      degradation is visible rather than silent.
    - LLM_ONLY: every page is sent to the LLM, bypassing `should_delegate`
      entirely (`resolve_delegated_pages` is called directly on all pages,
      not through `classify_pages`, precisely so this doesn't need a second
      copy of the batching/retry logic). Requires `llm_config.enabled`;
      raises ValueError immediately otherwise, since a disabled LLM makes
      this mode meaningless -- there is no sensible partial result to
      return quietly.
    """
    pages = extract_pages(pdf_path, rules)
    empty_text_pages = sorted(p.page_number for p in pages if p.normalized_text_length == 0)

    rule_results = [classify_page(page, rules) for page in pages]

    delegation_targets = sorted(
        result.page_number for result in rule_results if should_delegate(result, rules)[0]
    )

    run_warnings: list[WarningCode] = []
    llm_call_records: list[LLMCallRecord] = []

    if mode is PipelineMode.RULE_ONLY:
        final_results = rule_results
        llm_target_pages = delegation_targets

    elif mode is PipelineMode.HYBRID:
        llm_target_pages = delegation_targets
        final_results, llm_call_records = classify_pages(pages, rule_results, rules, llm_config)
        if not llm_config.enabled:
            run_warnings.append(WarningCode.LLM_DISABLED)

    elif mode is PipelineMode.LLM_ONLY:
        if not llm_config.enabled:
            raise ValueError(
                "PipelineMode.LLM_ONLY requires an enabled LLM configuration, but "
                "no API key was found (llm_config.enabled is False). Configure "
                "LLM_API_KEY, or run with rule-only or hybrid mode instead."
            )
        llm_target_pages = sorted(p.page_number for p in pages)
        pages_by_number = {p.page_number: p for p in pages}
        final_results, llm_call_records = resolve_delegated_pages(
            rule_results, pages_by_number, rules, llm_config
        )

    else:
        raise ValueError(f"Unsupported pipeline mode: {mode!r}")

    physical_groups, documents = group_pages(final_results, pages, rules)

    rule_resolved = sum(1 for r in final_results if r.decision_source is DecisionSource.RULE)
    llm_called = sum(1 for r in final_results if r.llm_called)
    llm_resolved = sum(1 for r in final_results if r.decision_source is DecisionSource.LLM)
    rule_fallback_count = sum(
        1 for r in final_results if r.decision_source is DecisionSource.RULE_FALLBACK
    )
    llm_parse_failures = sum(
        1 for r in final_results if WarningCode.LLM_PARSE_FAILED in r.warnings
    )
    orphan_pages = sorted(r.page_number for r in final_results if r.is_orphan)

    run_report = RunReport(
        mode=mode,
        source_pdf_name=Path(pdf_path).name,
        total_pages=len(pages),
        rule_resolved=rule_resolved,
        llm_target_pages=llm_target_pages,
        llm_called=llm_called,
        llm_resolved=llm_resolved,
        rule_fallback_count=rule_fallback_count,
        llm_parse_failures=llm_parse_failures,
        llm_call_records=llm_call_records,
        empty_text_pages=empty_text_pages,
        orphan_pages=orphan_pages,
        llm_config=llm_config,
        rules_version=rules.get("version"),
        warnings=run_warnings,
    )

    return PipelineResult(
        page_results=final_results,
        physical_groups=physical_groups,
        reconstructed_documents=documents,
        run_report=run_report,
    )


_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def check_no_pii(text: str) -> None:
    """Last line of defense, called immediately before writing any output file.

    Every value that reaches this point should already be de-identified --
    `PageResult`, `PhysicalGroup`, `ReconstructedDocument`, and `RunReport`
    are all defined in schema.py specifically to carry no raw page text or
    raw identifiers, and nothing in this pipeline builds an output string
    from anything else. This check existing at all does not mean PII is
    expected to slip through here; it exists so that IF an upstream bug
    ever put a raw SSN or email address into a value that reaches this
    point, the run fails loudly at the save step instead of writing the
    leak to disk. Passing without raising is the normal, expected outcome
    on every run -- a hit here means there is a bug upstream, not that this
    check is working as some kind of routine filter.

    Raises ValueError (and writes nothing) if `text` contains an SSN-like
    or email-like substring.
    """
    if _SSN_PATTERN.search(text):
        raise ValueError("Refusing to write output: text matches an SSN-like pattern")
    if _EMAIL_PATTERN.search(text):
        raise ValueError("Refusing to write output: text matches an email-like pattern")
