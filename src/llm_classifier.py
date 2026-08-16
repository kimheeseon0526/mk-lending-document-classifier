"""LLM delegation for pages the rule classifier is not confident about.

PII BOUNDARY: this is the module that turns raw `ExtractedPage.text` into
something that may leave the process. Every path from raw text to
`LLMPageRequest.excerpt` goes through `mask_text` first -- nothing here may
construct an `LLMPageRequest` from unmasked text. Everything downstream of
`build_request` (batches, prompts, responses, `PageResult`) is already
de-identified and safe to log.

SCOPE OF THIS FILE: the real OpenAI network call is not implemented yet.
`call_llm` is the single function reserved for it and currently raises
NotImplementedError. Every other function -- masking, request/prompt
building, batching, response validation, retry/fallback orchestration -- is
complete and exercised by tests with `call_llm` monkeypatched. This split
matters operationally: the full pipeline must be able to run end to end in
rule-only mode with no API key configured (`load_llm_config` degrades to
`enabled=False` rather than raising), and the failure/fallback paths need to
be proven correct before a real key and a real call are wired in.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from src.extract import extract_pages, load_rules
from src.rule_classifier import classify_page, should_delegate
from src.schema import (
    DecisionSource,
    DocType,
    ExtractedPage,
    LLMBatchResponse,
    LLMCallRecord,
    LLMPageDecision,
    LLMPageRequest,
    LLMRuntimeConfig,
    PageResult,
    WarningCode,
)

logger = logging.getLogger(__name__)

# Bump this (and LLMRuntimeConfig.prompt_version, which is set to match it in
# load_llm_config) whenever SYSTEM_PROMPT_TEMPLATE or the user-prompt layout
# in build_prompt changes. Evaluation runs are only comparable to each other
# when they used the same prompt version.
PROMPT_VERSION = "v1"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_llm_config(env_path: str | Path = ".env") -> LLMRuntimeConfig:
    """Build `LLMRuntimeConfig` from environment variables via python-dotenv.

    A missing `.env` file is not an error: `load_dotenv` simply leaves the
    process environment untouched, and every setting below then falls back
    to `LLMRuntimeConfig`'s own defaults -- this is the entry point for the
    pipeline's automatic degrade to rule-only mode.

    `LLM_API_KEY` is read into a local variable and never returned, logged,
    or otherwise stored: `LLMRuntimeConfig` has no `api_key` field by
    design. The only trace of the key's presence is `enabled` (a bool).

    `max_excerpt_chars` is sourced from rules.yaml's `masking.max_excerpt_chars`
    (loaded here with the default `config/rules.yaml` path, matching how
    `rule_classifier.py` loads rules.yaml for its own module-level check) so
    the two never drift apart. `batch_size` has no environment variable or
    rules.yaml setting of its own and keeps `LLMRuntimeConfig`'s schema
    default (10).
    """
    load_dotenv(dotenv_path=env_path, override=False)

    api_key = os.environ.get("LLM_API_KEY", "").strip()
    enabled = bool(api_key)

    rules = load_rules()
    max_excerpt_chars = rules["masking"]["max_excerpt_chars"]

    kwargs: dict = {
        "enabled": enabled,
        "max_excerpt_chars": max_excerpt_chars,
        "prompt_version": PROMPT_VERSION,
    }
    if "LLM_PROVIDER" in os.environ:
        kwargs["provider"] = os.environ["LLM_PROVIDER"]
    if "LLM_MODEL" in os.environ:
        kwargs["model"] = os.environ["LLM_MODEL"]
    if "LLM_TEMPERATURE" in os.environ:
        kwargs["temperature"] = float(os.environ["LLM_TEMPERATURE"])
    if "LLM_TIMEOUT_SECONDS" in os.environ:
        kwargs["timeout_seconds"] = int(os.environ["LLM_TIMEOUT_SECONDS"])
    if "LLM_MAX_RETRIES" in os.environ:
        kwargs["max_retries"] = int(os.environ["LLM_MAX_RETRIES"])

    return LLMRuntimeConfig(**kwargs)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def _protected_spans(text: str, masking_cfg: dict) -> list[tuple[int, int]]:
    """Character spans matching `never_mask_regex` (formatted dollar amounts)."""
    spans: list[tuple[int, int]] = []
    for pattern in masking_cfg.get("never_mask_regex", []):
        for m in re.finditer(pattern, text):
            spans.append((m.start(), m.end()))
    return spans


def _overlaps_any(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < s_end and end > s_start for s_start, s_end in spans)


def _apply_pattern(text: str, pattern_cfg: dict, protected: list[tuple[int, int]]) -> str:
    regex = re.compile(pattern_cfg["regex"])
    replacement = pattern_cfg["replacement"]

    def _sub(m: re.Match) -> str:
        if _overlaps_any(m.start(), m.end(), protected):
            return m.group(0)
        return m.expand(replacement)  # handles \1-style backreferences

    return regex.sub(_sub, text)


# Labels the signature_line_heuristic name-masking strategy looks for. These
# are an implementation detail of that specific strategy name -- rules.yaml's
# `name_masking` block only names the strategy and the replacement token, it
# has no field to configure this label list, so it lives here in code rather
# than being invented as an unrequested new rules.yaml key.
_NAME_LABEL_PATTERN = re.compile(
    r"\b(?:Borrower|Applicant|Signature|Name|Trustor|Attn)\b\s*[:\-]?\s*",
    re.IGNORECASE,
)
_CAPITALIZED_TOKEN = re.compile(r"[A-Z][A-Za-z'\-]*")
_NAME_LABEL_WORDS = {"borrower", "applicant", "signature", "name", "trustor", "attn"}


def _mask_names(text: str, masking_cfg: dict, protected: list[tuple[int, int]]) -> str:
    """Redact capitalized token runs that sit next to a name-adjacent label.

    LIMITATION (signature_line_heuristic, per rules.yaml): this is a regex
    heuristic, not a named-entity model. It only catches names that appear
    within a few tokens of one of the recognized labels; a name anywhere
    else on the page (e.g. in running prose with no nearby label) will not
    be caught, and any other capitalized proper noun that happens to sit
    right after a label could in principle be swept in. It is a best-effort
    reduction of exposure, not a guarantee that no name survives -- see the
    README for the documented residual risk.
    """
    preserve = set(masking_cfg.get("preserve_tokens", []))
    replacement = masking_cfg.get("name_masking", {}).get("replacement", "[NAME]")

    out: list[str] = []
    cursor = 0
    for label_match in _NAME_LABEL_PATTERN.finditer(text):
        search_start = label_match.end()
        tokens: list[re.Match] = []
        pos = search_start
        while len(tokens) < 3:
            candidate_pos = pos
            if tokens:
                # Require exactly one space between consecutive tokens (same
                # line, one name) -- then look for the next token past it.
                if candidate_pos >= len(text) or text[candidate_pos] != " ":
                    break
                candidate_pos += 1
            m = _CAPITALIZED_TOKEN.match(text, candidate_pos)
            if m is None:
                break
            token_text = m.group(0)
            if token_text in preserve or token_text.lower() in _NAME_LABEL_WORDS:
                break
            tokens.append(m)
            pos = m.end()

        if len(tokens) < 2:
            continue  # not enough evidence of an actual name

        name_start, name_end = tokens[0].start(), tokens[-1].end()
        if name_start < cursor or _overlaps_any(name_start, name_end, protected):
            continue

        out.append(text[cursor:name_start])
        out.append(replacement)
        cursor = name_end

    out.append(text[cursor:])
    return "".join(out)


def mask_text(text: str, rules: dict) -> str:
    """Mask PII in `text` per rules.yaml's `masking` policy.

    Order is fixed and matters:
    1. Locate every `never_mask_regex` span (formatted dollar amounts) in
       the text and treat those spans as protected.
    2. Apply `patterns` in the order given in rules.yaml; a match that
       overlaps a protected span is left untouched. Protected spans are
       recomputed from scratch before each pattern, since every prior
       substitution shifts character offsets in the text that follows.
    3. If `name_masking.enabled`, apply `_mask_names` (see its docstring for
       the heuristic's limitations).

    Regression constraint this function must never break: package_01's P&L
    page has no standard income keyword anywhere on it -- its only
    classification signal is dollar-amount structure plus the "Realtor" and
    "CTEC" tokens. If masking destroys those, that page becomes
    unclassifiable by the LLM as well as the rule stage.
    """
    masking_cfg = rules["masking"]
    result = text

    for pattern_cfg in masking_cfg.get("patterns", []):
        protected = _protected_spans(result, masking_cfg)
        result = _apply_pattern(result, pattern_cfg, protected)

    name_cfg = masking_cfg.get("name_masking", {})
    if name_cfg.get("enabled", False):
        protected = _protected_spans(result, masking_cfg)
        result = _mask_names(result, masking_cfg, protected)

    return result


# ---------------------------------------------------------------------------
# Request / prompt construction
# ---------------------------------------------------------------------------


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Cut `text` to at most `max_chars`, preferring a sentence/line/word
    boundary near the limit over a hard mid-token cut."""
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    for boundary, keep_len in ((". ", 2), ("\n", 1)):
        idx = window.rfind(boundary)
        if idx > max_chars * 0.5:  # don't discard most of the excerpt chasing a boundary
            return window[: idx + keep_len].rstrip()

    idx = window.rfind(" ")
    if idx > 0:
        return window[:idx].rstrip()
    return window


def build_request(page: ExtractedPage, rules: dict, max_chars: int) -> LLMPageRequest:
    """Build the de-identified request the LLM sees for one page.

    Masks first, then truncates -- truncating first could cut a raw
    identifier in half and leave the fragment unmasked. `marker_hint` is
    built only from `marker_page`/`marker_total` (structural, not PII);
    `raw_identifiers` never appears here.
    """
    masked = mask_text(page.text, rules)
    excerpt = _truncate_at_boundary(masked, max_chars)

    marker_hint = None
    if page.marker_page is not None:
        marker_hint = (
            f"page {page.marker_page} of {page.marker_total}"
            if page.marker_total is not None
            else f"page {page.marker_page}"
        )

    return LLMPageRequest(
        page_id=page.page_number,
        excerpt=excerpt,
        marker_style=page.marker_style,
        marker_hint=marker_hint,
    )


def build_batches(
    requests: list[LLMPageRequest], batch_size: int
) -> list[list[LLMPageRequest]]:
    """Split `requests` into fixed-size batches, preserving order."""
    return [requests[i : i + batch_size] for i in range(0, len(requests), batch_size)]


_DOC_TYPE_LINES = (
    "- URLA_1003: Uniform Residential Loan Application (the borrower's loan application form)",
    "- INCOME_DOC: income verification (pay stubs, W-2, P&L statement, employment "
    "verification, tax forms)",
    "- CREDIT_REPORT: consumer credit report",
    "- TITLE_REPORT: preliminary title report",
    "- OTHER: none of the above, or you cannot tell",
)

SYSTEM_PROMPT = (
    "You classify individual pages from a U.S. mortgage loan document package.\n\n"
    "Each page may be any physical page of a multi-page original document -- "
    "it is not necessarily a cover page or the first page of that document, "
    "so classify it from its own content and structural hints, not from "
    "whether it looks like the start of something.\n\n"
    "Assign exactly one of these five labels to each page:\n"
    + "\n".join(_DOC_TYPE_LINES)
    + "\n\nUse only these five labels -- never invent a new one.\n"
    "If you are not confident, choose OTHER and give it a low confidence "
    "score rather than guessing at one of the other four.\n"
    "You must return exactly one decision for every page_id you are given, "
    "no more and no fewer.\n"
    "All personal identifiers in the excerpts have already been masked. Do "
    "not include any identifier-like text (names, numbers, account IDs) in "
    "your evidence field -- describe the page in general terms only."
)


def build_prompt(batch: list[LLMPageRequest]) -> tuple[str, str]:
    """Build the (system, user) prompt pair for one batch of requests.

    Bump PROMPT_VERSION (and keep LLMRuntimeConfig.prompt_version in sync
    via load_llm_config) whenever SYSTEM_PROMPT or the layout below changes
    -- evaluation runs are only comparable across runs that used the same
    prompt version.
    """
    lines = ["Classify each of the following pages:"]
    for req in batch:
        lines.append(f"\npage_id: {req.page_id}")
        lines.append(f"marker_hint: {req.marker_hint or 'none'}")
        lines.append(f"excerpt: {req.excerpt}")
    user_prompt = "\n".join(lines)

    return SYSTEM_PROMPT, user_prompt


def response_json_schema() -> dict:
    """JSON schema for `LLMBatchResponse`, for OpenAI Structured Outputs."""
    return LLMBatchResponse.model_json_schema()


# ---------------------------------------------------------------------------
# The network call -- not implemented yet
# ---------------------------------------------------------------------------


def call_llm(batch: list[LLMPageRequest], config: LLMRuntimeConfig) -> LLMBatchResponse:
    """Call the LLM provider and return its structured decision for `batch`.

    Not implemented yet. This is the one function reserved for the real
    OpenAI Structured Outputs call (using `build_prompt` for the messages
    and `response_json_schema()` for the schema); everything else in this
    module is already complete and tested against a monkeypatched stand-in
    for this function.
    """
    raise NotImplementedError(
        "OpenAI Structured Outputs call is not implemented yet. "
        "Run with rule-only mode, or implement this function."
    )


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


def _extract_raw_decisions(response: object) -> list[object] | None:
    """Pull the `decisions` list out of whatever `call_llm` returned.

    Accepts an `LLMBatchResponse`, a plain dict (e.g. parsed straight from
    JSON), or anything else -- returns None when the shape can't be
    interpreted at all, which is exactly what LLM_PARSE_FAILED means.
    """
    if isinstance(response, LLMBatchResponse):
        return list(response.decisions)
    if isinstance(response, dict):
        decisions = response.get("decisions")
        if isinstance(decisions, list):
            return decisions
    return None


def _coerce_decision(item: object) -> LLMPageDecision | None:
    """Re-validate one raw decision, returning None if it fails.

    `item` may already be an `LLMPageDecision` built with `model_construct`
    (which bypasses validation -- used to represent a Structured Outputs
    response that satisfied the JSON shape but not every field constraint,
    since minimum/maximum on numeric fields is not strictly enforced by
    every provider's structured-output mode) or a plain dict. Either way,
    running it back through full validation is what catches an invalid
    `doc_type` label or an out-of-[0,1] confidence.
    """
    if isinstance(item, LLMPageDecision):
        payload = item.model_dump()
    elif isinstance(item, dict):
        payload = item
    else:
        return None

    try:
        return LLMPageDecision.model_validate(payload)
    except ValidationError:
        return None


def validate_response(
    response: object, batch: list[LLMPageRequest]
) -> tuple[LLMBatchResponse, list[WarningCode]]:
    """Defensively validate `response` against the requested `batch`.

    Never raises. Returns only the decisions that both passed full
    `LLMPageDecision` validation and belong to a `page_id` present in
    `batch`, plus every WarningCode describing what was wrong:
    - the response's shape can't be read as decisions at all -> LLM_PARSE_FAILED
    - an individual decision fails validation (bad label, confidence
      outside [0, 1], missing field, ...) -> LLM_LABEL_INVALID, item dropped
    - a decision's page_id isn't one that was requested -> silently dropped
      (not itself warning-worthy; a genuine mismatch is still caught below)
    - the final valid page_id set doesn't equal the requested set -> LLM_COUNT_MISMATCH
    """
    warnings: list[WarningCode] = []
    requested_ids = {req.page_id for req in batch}

    raw_decisions = _extract_raw_decisions(response)
    if raw_decisions is None:
        return LLMBatchResponse(decisions=[]), [WarningCode.LLM_PARSE_FAILED]

    valid_decisions: list[LLMPageDecision] = []
    for item in raw_decisions:
        decision = _coerce_decision(item)
        if decision is None:
            warnings.append(WarningCode.LLM_LABEL_INVALID)
            continue
        if decision.page_id not in requested_ids:
            continue
        valid_decisions.append(decision)

    if {d.page_id for d in valid_decisions} != requested_ids:
        warnings.append(WarningCode.LLM_COUNT_MISMATCH)

    return LLMBatchResponse(decisions=valid_decisions), warnings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def classify_pages(
    pages: list[ExtractedPage],
    rule_results: list[PageResult],
    rules: dict,
    config: LLMRuntimeConfig,
) -> tuple[list[PageResult], list[LLMCallRecord]]:
    """Resolve every page `should_delegate` flags; pass the rest through untouched.

    `llm_confidence` and `rule_score` are never compared or combined here
    (or anywhere in this module): they are heuristic numbers on different,
    uncalibrated scales -- rule_score comes from keyword-weight arithmetic,
    llm_confidence is the model's self-reported number. Treating them as
    comparable would imply a relationship that doesn't exist.

    Never mutates the `PageResult` objects passed in via `rule_results`:
    every revised result is a fresh `model_copy`, so the caller's original
    list is unaffected by this call.
    """
    pages_by_number = {p.page_number: p for p in pages}

    passthrough: list[PageResult] = []
    delegated: list[PageResult] = []
    for result in rule_results:
        delegate, _ = should_delegate(result, rules)
        (delegated if delegate else passthrough).append(result)

    if not config.enabled:
        disabled = [
            r.model_copy(update={"warnings": r.warnings + [WarningCode.LLM_DISABLED]})
            for r in delegated
        ]
        return passthrough + disabled, []

    rule_by_page = {r.page_number: r for r in delegated}
    requests = [
        build_request(pages_by_number[page_number], rules, config.max_excerpt_chars)
        for page_number in rule_by_page
    ]
    batches = build_batches(requests, config.batch_size)

    resolved: list[PageResult] = []
    call_records: list[LLMCallRecord] = []
    max_attempts = config.max_retries + 1

    for batch_index, batch in enumerate(batches):
        requested_ids = [req.page_id for req in batch]

        response = None
        call_failed = True
        attempt_count = 0
        start = time.perf_counter()
        for attempt in range(1, max_attempts + 1):
            attempt_count = attempt
            try:
                response = call_llm(batch, config)
                call_failed = False
                break
            except Exception:
                logger.warning(
                    "llm call failed batch=%d attempt=%d/%d",
                    batch_index,
                    attempt,
                    max_attempts,
                )
        latency_ms = int((time.perf_counter() - start) * 1000)

        if call_failed:
            for page_id in requested_ids:
                rule_result = rule_by_page[page_id]
                resolved.append(
                    rule_result.model_copy(
                        update={
                            "decision_source": DecisionSource.RULE_FALLBACK,
                            "llm_called": True,
                            "llm_failed": True,
                            "warnings": rule_result.warnings + [WarningCode.LLM_CALL_FAILED],
                        }
                    )
                )
            call_records.append(
                LLMCallRecord(
                    batch_index=batch_index,
                    requested_page_ids=requested_ids,
                    returned_page_ids=[],
                    attempt_count=attempt_count,
                    succeeded=False,
                    warnings=[WarningCode.LLM_CALL_FAILED],
                    latency_ms=latency_ms,
                )
            )
            continue

        validated, response_warnings = validate_response(response, batch)
        decisions_by_page = {d.page_id: d for d in validated.decisions}

        for page_id in requested_ids:
            rule_result = rule_by_page[page_id]
            decision = decisions_by_page.get(page_id)
            if decision is not None:
                resolved.append(
                    rule_result.model_copy(
                        update={
                            "decision_source": DecisionSource.LLM,
                            "doc_type": decision.doc_type,
                            "llm_called": True,
                            "llm_failed": False,
                            "llm_doc_type": decision.doc_type,
                            "llm_confidence": decision.confidence,
                        }
                    )
                )
            else:
                # response_warnings explains why this specific page has no
                # valid decision (parse failure / count mismatch / an
                # invalid item that got dropped); a page that DID resolve
                # cleanly does not need that noise in its own warnings.
                resolved.append(
                    rule_result.model_copy(
                        update={
                            "decision_source": DecisionSource.RULE_FALLBACK,
                            "llm_called": True,
                            "llm_failed": True,
                            "warnings": rule_result.warnings + response_warnings,
                        }
                    )
                )

        call_records.append(
            LLMCallRecord(
                batch_index=batch_index,
                requested_page_ids=requested_ids,
                returned_page_ids=sorted(decisions_by_page.keys()),
                attempt_count=attempt_count,
                succeeded=True,
                warnings=response_warnings,
                latency_ms=latency_ms,
            )
        )

    return passthrough + resolved, call_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Show which pages the rule classifier would delegate to "
        "the LLM. Does not call any API."
    )
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument(
        "--rules", default="config/rules.yaml", help="Path to the rules YAML file."
    )
    parser.add_argument("--env", default=".env", help="Path to the .env file.")
    args = parser.parse_args(argv)

    config = load_llm_config(args.env)
    print(f"llm enabled: {config.enabled}")
    print(
        f"provider: {config.provider}  model: {config.model}  "
        f"prompt_version: {config.prompt_version}"
    )

    rules = load_rules(args.rules)
    pages = extract_pages(args.pdf_path, rules)
    pages_by_number = {p.page_number: p for p in pages}
    rule_results = [classify_page(page, rules) for page in pages]

    targets: list[tuple[PageResult, list[WarningCode]]] = []
    for result in rule_results:
        delegate, reasons = should_delegate(result, rules)
        if delegate:
            targets.append((result, reasons))

    if not config.enabled:
        print(f"LLM disabled, would delegate {len(targets)} pages")
        for result, reasons in targets:
            print(f"  page {result.page_number}: {','.join(r.value for r in reasons)}")
        return

    print(f"would delegate {len(targets)} pages:")
    headers = ["page", "reasons", "marker_hint", "excerpt_len"]
    rows = []
    for result, reasons in targets:
        request = build_request(pages_by_number[result.page_number], rules, config.max_excerpt_chars)
        rows.append(
            [
                str(result.page_number),
                ",".join(r.value for r in reasons),
                request.marker_hint or "-",
                str(len(request.excerpt)),
            ]
        )
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
    ]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


if __name__ == "__main__":
    main()
