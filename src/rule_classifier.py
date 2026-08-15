"""Rule-based page classification for the mortgage document page classifier.

PII BOUNDARY: this module reads `ExtractedPage.text` (raw PII) only inside
`classify_page` to test keyword membership. It never returns, logs, or
persists that text. Its output is `PageResult` (`src/schema.py`), which is
safe to write to disk. `matched_signals` on that result holds only signature
strings copied verbatim from `config/rules.yaml` -- those are configuration
constants, not page content, so they carry no PII.

Does not call an LLM. Only decides (a) a heuristic doc_type for a page from
keyword evidence and (b) whether that evidence is strong enough to skip LLM
delegation. `llm_classifier.py` and `pipeline.py` own the actual delegation.
"""

from __future__ import annotations

import argparse
import logging

from src.extract import extract_pages, load_rules
from src.schema import (
    DecisionSource,
    DocType,
    ExtractedPage,
    PageResult,
    WarningCode,
)

logger = logging.getLogger(__name__)


def validate_never_strong(rules: dict) -> None:
    """Raise ValueError if any `never_strong` string appears in a `strong` list.

    `never_strong` exists precisely so that an ambiguous string (one that
    occurs across multiple document types, or is generic English) can never
    be silently promoted to a single-signal confirmation by a careless edit
    to rules.yaml. This is a configuration-integrity check, not a runtime
    classification concern, so a violation raises rather than being folded
    into `PageResult.warnings`.
    """
    never_strong = set(rules.get("never_strong", []))
    if not never_strong:
        return

    violations = [
        (doc_type_name, signal)
        for doc_type_name, cfg in rules.get("doc_types", {}).items()
        for signal in cfg.get("strong", [])
        if signal in never_strong
    ]
    if violations:
        detail = ", ".join(f"{dt}:{signal!r}" for dt, signal in violations)
        raise ValueError(
            f"rules.yaml misconfiguration: never_strong string(s) found in a "
            f"'strong' list -- {detail}"
        )


# Module-load-time integrity check against the real rule configuration, so a
# bad edit to rules.yaml fails immediately at import rather than at whatever
# page happens to trigger the ambiguous signal.
validate_never_strong(load_rules())


def _strong_matched_doc_types(matched_signals: list[str], rules: dict) -> set[DocType]:
    """Recover which doc_types had a *strong* signal among `matched_signals`.

    `PageResult.matched_signals` deliberately holds every signal that matched
    across *all* classes (not just the winning class) -- see the comment in
    `classify_page`. That is what lets `should_delegate`, which only receives
    the already-built `PageResult`, reconstruct cross-class conflicts without
    re-reading the page text.
    """
    matched = set(matched_signals)
    return {
        DocType(doc_type_name)
        for doc_type_name, cfg in rules.get("doc_types", {}).items()
        if matched.intersection(cfg.get("strong", []))
    }


def classify_page(page: ExtractedPage, rules: dict) -> PageResult:
    """Score `page.text` against `rules["doc_types"]` keyword signatures.

    `rule_score` is a heuristic evidence score built from keyword-weight
    arithmetic (see the `scoring` formula documented in rules.yaml). It is
    NOT a calibrated probability and must not be interpreted or compared as
    one -- e.g. it is on a different scale than `LLMPageDecision.confidence`.

    `marker_style` is intentionally NOT used as classification evidence
    here, even though rules.yaml declares a `marker_affinity` per doc_type.
    Two reasons: (1) URLA_1003 and CREDIT_REPORT share the same marker_total
    (11) in package_01, so marker alone cannot discriminate between them;
    (2) a page's position/marker is a structural fact used for grouping
    reconstructed documents, not evidence of what the page's text says it
    is. Folding it into scoring would let position outvote content.
    """
    scoring_cfg = rules["scoring"]
    strong_weight = scoring_cfg["strong_weight"]
    weak_weight = scoring_cfg["weak_weight"]
    evidence_saturation = scoring_cfg["evidence_saturation"]
    case_sensitive = scoring_cfg["case_sensitive"]

    haystack = page.text if case_sensitive else page.text.lower()

    raw_scores: dict[DocType, float] = {}
    # Every matched signal from every class, not just the eventual top-1.
    # should_delegate needs this to detect "two different classes both fired
    # a strong signature" without re-scanning page text.
    all_matched_signals: list[str] = []

    for doc_type_name, cfg in rules["doc_types"].items():
        doc_type = DocType(doc_type_name)
        raw = 0.0
        for signal in cfg.get("strong", []):
            needle = signal if case_sensitive else signal.lower()
            if needle in haystack:
                raw += strong_weight
                all_matched_signals.append(signal)
        for signal in cfg.get("weak", []):
            needle = signal if case_sensitive else signal.lower()
            if needle in haystack:
                raw += weak_weight
                all_matched_signals.append(signal)
        if raw > 0:
            raw_scores[doc_type] = raw

    warnings: list[WarningCode] = []

    if not raw_scores:
        doc_type = DocType.OTHER
        rule_score = 0.0
        rule_margin = 0.0
        rule_scores_out: dict[DocType, float] = {}
        warnings.append(WarningCode.NO_SIGNATURE)
    else:
        total_raw = sum(raw_scores.values())
        rule_scores_out = {dt: raw / total_raw for dt, raw in raw_scores.items()}

        ranked = sorted(rule_scores_out.items(), key=lambda kv: kv[1], reverse=True)
        doc_type, top1_share = ranked[0]
        # "top2가 없으면 share[top1]" -- with a single scoring class, margin
        # equals the top1 share itself, not top1 minus itself (which would
        # collapse every single-class page to a margin of 0).
        rule_margin = top1_share - ranked[1][1] if len(ranked) > 1 else top1_share

        mass = min(1.0, raw_scores[doc_type] / evidence_saturation)
        rule_score = top1_share * mass

    strong_signature_matched = doc_type in _strong_matched_doc_types(
        all_matched_signals, rules
    )

    return PageResult(
        page_number=page.page_number,
        doc_type=doc_type,
        decision_source=DecisionSource.RULE,
        rule_doc_type=doc_type,
        rule_score=rule_score,
        rule_margin=rule_margin,
        rule_scores=rule_scores_out,
        strong_signature_matched=strong_signature_matched,
        matched_signals=all_matched_signals,
        # Classes with a non-zero raw score, i.e. len(raw_scores) before it
        # was reduced to shares. should_delegate uses this to tell "margin
        # between two real contenders" apart from "margin computed with
        # fewer than two classes in the running" (see should_delegate).
        matched_class_count=len(raw_scores),
        marker_style=page.marker_style,
        marker_page=page.marker_page,
        marker_total=page.marker_total,
        instance_id=None,
        rotation=page.rotation,
        normalized_text_length=page.normalized_text_length,
        warnings=warnings,
    )


def should_delegate(result: PageResult, rules: dict) -> tuple[bool, list[WarningCode]]:
    """Decide whether `result` needs LLM delegation, per rules.yaml `delegation`.

    Steps 1-2 are short-circuits, evaluated first and in order: a strong
    signature -- whether conflicting across classes or confirmed in exactly
    one -- is decisive and returns immediately without looking at steps 3-5.

    Steps 3-5 (SHORT_TEXT, LOW_RULE_SCORE, NARROW_RULE_MARGIN) are no longer
    a priority chain that stops at the first hit. All three are evaluated
    unconditionally and every one that is satisfied is collected, in this
    order, into the returned list. A page that is both short AND low-scoring
    is useful signal for tuning these thresholds and was previously hidden
    by only reporting one reason. Callers that fold this list into
    `PageResult.warnings` (as the CLI does) get a stable "primary reason" for
    free by reading `warnings[0]`, since append order is fixed -- no separate
    field is needed for that.

    NARROW_RULE_MARGIN is additionally gated by
    `matched_class_count >= 2` (unless rules.yaml sets
    `require_multiple_classes_for_margin_check: false`). `rule_margin` is
    only meaningful as "how narrowly did two document types beat each other"
    when two or more classes actually scored. With 0 matched classes,
    rule_margin == 0.0 by construction (see classify_page) and the page is
    already explained by NO_SIGNATURE + LOW_RULE_SCORE. With exactly 1
    matched class, rule_margin == 1.0 by construction (top1 has no
    competitor), which is the opposite of "narrow" and would never fire
    the threshold anyway -- but gating explicitly documents that a page
    delegated for a real margin contest is a materially different situation
    from a page delegated for absent evidence, even though the numbers
    happen not to collide in practice.

    Never returns `(True, [])` -- every delegation carries at least one
    WarningCode explaining why.
    """
    delegation_cfg = rules["delegation"]
    strong_doc_types = _strong_matched_doc_types(result.matched_signals, rules)

    # 1. Two or more classes each fired a strong signature: irreconcilable,
    #    delegate immediately.
    if (
        len(strong_doc_types) >= 2
        and delegation_cfg.get("delegate_on_conflicting_strong_signals", False)
    ):
        return True, [WarningCode.CONFLICTING_SIGNALS]

    # 2. Exactly one class fired a strong signature: trust it, skip the
    #    evidence-strength checks below entirely.
    if (
        len(strong_doc_types) == 1
        and delegation_cfg.get("strong_signature_short_circuit", False)
    ):
        return False, []

    # 3-5. No strong short-circuit applies. Evaluate every remaining check
    # and collect all that are satisfied, rather than stopping at the first.
    reasons: list[WarningCode] = []

    if result.normalized_text_length < delegation_cfg["short_text_threshold"]:
        reasons.append(WarningCode.SHORT_TEXT)

    if result.rule_score < delegation_cfg["rule_score_threshold"]:
        reasons.append(WarningCode.LOW_RULE_SCORE)

    require_multiple = delegation_cfg.get(
        "require_multiple_classes_for_margin_check", False
    )
    if (
        (result.matched_class_count >= 2 or not require_multiple)
        and result.rule_margin < delegation_cfg["rule_margin_threshold"]
    ):
        reasons.append(WarningCode.NARROW_RULE_MARGIN)

    return bool(reasons), reasons


TABLE_TITLE = "Rule top-1 candidates (before LLM delegation)"


def _format_table(rows: list[tuple[PageResult, bool, list[WarningCode]]]) -> str:
    """Render a summary table. `matched_signals` content is never printed,
    only its count -- the strings are config constants, not PII, but the
    table is a debugging aid and has no reason to echo rule internals.

    Titled explicitly as candidates, not final predictions: rows still need
    LLM delegation (and grouping, downstream) before any row is a finished
    per-page classification.

    `primary_reason`/`all_reasons` come straight from `should_delegate`'s
    return value, not from `PageResult.warnings`. The two are different
    questions: `warnings` is everything observed about the page, including
    classification-stage observations like NO_SIGNATURE that describe the
    rule outcome itself, not what caused delegation -- a page can carry
    NO_SIGNATURE while being delegated for SHORT_TEXT, and showing
    NO_SIGNATURE as the "reason" would misattribute the delegation to the
    wrong condition. `warnings` still gets its own column so both views stay
    visible side by side; nothing about what's stored on `PageResult`
    changes, only how it is displayed here.
    """
    headers = [
        "page",
        "doc_type",
        "rule_score",
        "rule_margin",
        "matched_classes",
        "strong",
        "delegate",
        "primary_reason",
        "all_reasons",
        "warnings",
    ]
    table_rows = [
        [
            str(result.page_number),
            result.doc_type.value,
            f"{result.rule_score:.2f}",
            f"{result.rule_margin:.2f}",
            str(result.matched_class_count),
            "Y" if result.strong_signature_matched else "N",
            "Y" if delegate else "N",
            reasons[0].value if reasons else "",
            ",".join(r.value for r in reasons) or "-",
            ",".join(w.value for w in result.warnings) or "-",
        ]
        for result, delegate, reasons in rows
    ]

    widths = [
        max(len(header), max((len(row[i]) for row in table_rows), default=0))
        for i, header in enumerate(headers)
    ]
    lines = [TABLE_TITLE, ""]
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    lines.extend(
        "  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in table_rows
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Rule-classify every page of a PDF (no LLM calls)."
    )
    # A single positional string captures paths containing spaces or '&'
    # as long as the caller quotes the argument; no manual splitting is
    # done anywhere on this value.
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument(
        "--rules", default="config/rules.yaml", help="Path to the rules YAML file."
    )
    args = parser.parse_args(argv)

    rules = load_rules(args.rules)
    pages = extract_pages(args.pdf_path, rules)

    rows = []
    for page in pages:
        result = classify_page(page, rules)
        delegate, reasons = should_delegate(result, rules)
        # PageResult.warnings still accumulates delegation reasons on top of
        # classify_page's own (e.g. NO_SIGNATURE) -- that storage contract is
        # unchanged. `reasons` is also kept separately here and passed
        # through to the table so the display layer can distinguish "why
        # delegated" from "everything observed about this page" instead of
        # reading the answer back out of the merged warnings list.
        result.warnings = result.warnings + reasons
        rows.append((result, delegate, reasons))

    print(_format_table(rows))


if __name__ == "__main__":
    main()
