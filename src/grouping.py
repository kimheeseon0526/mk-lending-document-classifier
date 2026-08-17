"""Grouping of classified pages into physical runs and logical documents.

PII BOUNDARY: reads only `PageResult` (de-identified) and `ExtractedPage.raw_identifiers`
(the latter only inside `assign_instance_ids`, and only to derive a run-local
surrogate string such as "CREDIT_REPORT#1" -- the raw identifier values
themselves are never copied onto `PageResult` or printed). Safe to persist
and log everything this module produces.

Two outputs, always both:
1. `PhysicalGroup` -- runs of physically adjacent pages sharing a predicted
   doc_type. This is the assessment's required output.
2. `ReconstructedDocument` -- an attempt to rebuild each original logical
   document from scattered pages, using identifiers and page markers as
   evidence. This is an extension on top of (1), never a replacement for it.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

from src.extract import extract_pages, load_rules
from src.rule_classifier import classify_page
from src.schema import (
    Completeness,
    DocType,
    ExtractedPage,
    GroupingKey,
    GroupIssue,
    MarkerStyle,
    PageLink,
    PageResult,
    PhysicalGroup,
    ReconstructedDocument,
    WarningCode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Physical grouping
# ---------------------------------------------------------------------------


def build_physical_groups(results: list[PageResult]) -> list[PhysicalGroup]:
    """Group physically adjacent pages that share a predicted doc_type.

    "Adjacent" means consecutive `page_number` values with the same
    `doc_type` -- not merely similar or nearby; a gap of even one page
    number ends the run.

    package_01 is a scatter shuffle: across all 39 pages, zero adjacent
    pairs share a doc_type. 39 one-page groups is therefore the CORRECT
    output for that input, not a symptom of a bug -- do not merge groups
    across gaps, smooth over single-page groups, or otherwise "fix" this.
    It reflects how the input was actually shuffled, not a flaw in the
    grouping logic.
    """
    if not results:
        return []

    ordered = sorted(results, key=lambda r: r.page_number)

    runs: list[list[PageResult]] = [[ordered[0]]]
    for prev, curr in zip(ordered, ordered[1:]):
        if curr.page_number == prev.page_number + 1 and curr.doc_type == prev.doc_type:
            runs[-1].append(curr)
        else:
            runs.append([curr])

    groups: list[PhysicalGroup] = []
    for index, run in enumerate(runs, start=1):
        groups.append(
            PhysicalGroup(
                group_id=f"PG{index:03d}",
                doc_type=run[0].doc_type,
                start_page=run[0].page_number,
                end_page=run[-1].page_number,
                page_count=len(run),
                physical_pages=[r.page_number for r in run],
            )
        )
    return groups


# ---------------------------------------------------------------------------
# 2. Instance identifiers
# ---------------------------------------------------------------------------


def assign_instance_ids(
    results: list[PageResult],
    extracted_pages: list[ExtractedPage],
    rules: dict,
) -> None:
    """Assign a run-local surrogate `instance_id` to pages sharing an identifier.

    Mutates `results` in place (sets `.instance_id` on each `PageResult`);
    returns nothing. `rules["instance_identifiers"]`'s `scope` is applied
    here for the first time in the pipeline -- extraction couldn't filter by
    doc_type because classification hadn't happened yet, but it has now.

    Two pages are the same instance iff `(doc_type, sorted scoped-identifier
    values)` match exactly. Only the derived surrogate string
    ("{doc_type}#{n}") is ever written to `PageResult.instance_id` --
    `raw_identifiers` values themselves never leave `ExtractedPage`.

    KNOWN CONSTRAINT on this dataset: in package_01, `loan_number`,
    `report_id`, and `prelim_number` are never actually captured by
    `extract_identifiers` -- in the PDF's text stream, the label and its
    value are not adjacent (other fields intervene between "Loan Number:"
    and the digits that eventually follow much later in reading order). So
    for most pages here, no scoped identifier value is available and
    `instance_id` stays None. That is the expected, normal outcome on this
    dataset, not an error condition -- nothing in this function raises
    because of it.
    """
    pages_by_number = {p.page_number: p for p in extracted_pages}

    names_by_doc_type: dict[DocType, list[str]] = {}
    for item in rules.get("instance_identifiers", []):
        for scoped in item.get("scope", []):
            names_by_doc_type.setdefault(DocType(scoped), []).append(item["name"])

    surrogate_by_key: dict[tuple, str] = {}
    counters: dict[DocType, int] = {}

    for result in results:
        page = pages_by_number.get(result.page_number)
        if page is None:
            result.instance_id = None
            continue

        names = names_by_doc_type.get(result.doc_type, [])
        values = tuple(
            sorted(
                (name, page.raw_identifiers[name])
                for name in names
                if name in page.raw_identifiers
            )
        )
        if not values:
            result.instance_id = None
            continue

        key = (result.doc_type, values)
        if key not in surrogate_by_key:
            counters[result.doc_type] = counters.get(result.doc_type, 0) + 1
            surrogate_by_key[key] = f"{result.doc_type.value}#{counters[result.doc_type]}"

        result.instance_id = surrogate_by_key[key]


# ---------------------------------------------------------------------------
# 3. Grouping key
# ---------------------------------------------------------------------------


def build_grouping_key(result: PageResult) -> GroupingKey:
    """Build the composite key `reconstruct_documents` groups pages by.

    All four fields are required -- on package_01, any single field alone
    over-merges:
    - `doc_type` alone merges the four distinct documents of one loan file
      into one group (they're all "a mortgage document").
    - `instance_id` alone merges URLA_1003 and CREDIT_REPORT, since both
      carry the same loan number in package_01.
    - `marker_total` alone merges them again: both report a total of 11.
    - `marker_style` is what actually tells them apart: URLA_1003 uses
      "N of M" while CREDIT_REPORT uses "Page N of M" -- same numbers,
      different style.
    """
    return GroupingKey(
        doc_type=result.doc_type,
        instance_id=result.instance_id,
        marker_style=result.marker_style,
        marker_total=result.marker_total,
    )


# ---------------------------------------------------------------------------
# 4. Logical reconstruction
# ---------------------------------------------------------------------------


def reconstruct_documents(
    results: list[PageResult], rules: dict
) -> list[ReconstructedDocument]:
    """Rebuild logical documents from `results`, per rules.yaml's 3-tier evidence ladder.

    Also mutates `results` in place: any page that lands in tier 3
    (insufficient evidence) gets `is_orphan = True` and
    `WarningCode.ORPHAN_PAGE` appended to `warnings`, and is not included in
    any returned `ReconstructedDocument`.

    Tiers (evaluated per page, highest first):
    1. strong_identifier_and_marker: `instance_id` and `marker_page` both
       present -- confirmed attachment. Evidence: "identifier_and_marker_match".
    2. form_family_and_marker: no `instance_id`, but `marker_style` is not
       NONE and `marker_page` is present -- pages sharing the same
       `GroupingKey` (instance_id=None, but same doc_type/style/total) are
       combined on marker evidence alone. Evidence:
       "marker_match_within_unique_instance". Flagged with
       `GroupIssue.AMBIGUOUS_ATTACHMENT`. (No `WarningCode` matches this
       condition in the fixed enum without overloading one that means
       something else -- `AMBIGUOUS_ATTACHMENT` in `issues` is the sole
       signal for this tier, by explicit agreement; `warnings` is left
       alone here.)
    3. insufficient: no `marker_page`, or neither tier above applies ->
       orphan.

    Duplicate policy (`keep_both_and_flag`): if the same logical page number
    is observed twice within a document, neither observation is dropped --
    both stay in `page_links`, and `GroupIssue.DUPLICATE_PAGE` is set. There
    is no basis in the available evidence for guessing which occurrence is
    "the real one", so discarding either would be a bigger risk than
    surfacing both and flagging it.

    Conflicting marker_total (`split_into_separate_documents`): since
    `marker_total` is part of `GroupingKey`, pages that agree on
    doc_type/instance_id/marker_style but disagree on marker_total already
    land in different groups by construction. What this function adds on
    top is detection: any such sibling groups are flagged with
    `GroupIssue.CONFLICTING_MARKER_TOTAL` on every one of them, rather than
    silently letting the split pass without comment.
    """
    ordered = sorted(results, key=lambda r: r.page_number)

    groups: dict[GroupingKey, list[PageResult]] = {}
    group_order: list[GroupingKey] = []

    for result in ordered:
        if result.marker_page is None:
            result.is_orphan = True
            result.warnings = result.warnings + [WarningCode.ORPHAN_PAGE]
            continue

        tier1 = result.instance_id is not None
        tier2 = result.instance_id is None and result.marker_style is not MarkerStyle.NONE
        if not (tier1 or tier2):
            result.is_orphan = True
            result.warnings = result.warnings + [WarningCode.ORPHAN_PAGE]
            continue

        key = build_grouping_key(result)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(result)

    # Detect marker_total conflicts among sibling keys that agree on
    # everything except marker_total.
    totals_by_triple: dict[tuple, set] = defaultdict(set)
    for key in group_order:
        totals_by_triple[(key.doc_type, key.instance_id, key.marker_style)].add(key.marker_total)
    conflicting_triples = {t for t, totals in totals_by_triple.items() if len(totals) > 1}

    documents: list[ReconstructedDocument] = []
    for index, key in enumerate(group_order, start=1):
        members = sorted(groups[key], key=lambda m: m.page_number)
        is_tier2 = key.instance_id is None

        observed = sorted(m.marker_page for m in members)  # marker_page is guaranteed non-None here
        duplicates = sorted(page for page, count in Counter(observed).items() if count > 1)

        expected_pages = key.marker_total
        if expected_pages is None:
            missing: list[int] = []
            completeness = Completeness.UNKNOWN_EXTENT
        else:
            missing = sorted(set(range(1, expected_pages + 1)) - set(observed))
            completeness = Completeness.COMPLETE if not missing else Completeness.INCOMPLETE

        evidence_label = (
            "marker_match_within_unique_instance" if is_tier2 else "identifier_and_marker_match"
        )
        page_links = [
            PageLink(
                physical_page=m.page_number,
                logical_page=m.marker_page,
                evidence=[evidence_label],
            )
            for m in members
        ]

        start_matches = sorted(m.page_number for m in members if m.marker_page == 1)
        start_physical_page = start_matches[0] if start_matches else None

        end_physical_page = None
        if observed:
            end_matches = sorted(m.page_number for m in members if m.marker_page == max(observed))
            end_physical_page = end_matches[0] if end_matches else None

        issues: list[GroupIssue] = []
        if duplicates:
            issues.append(GroupIssue.DUPLICATE_PAGE)
        if is_tier2:
            issues.append(GroupIssue.AMBIGUOUS_ATTACHMENT)
        if (key.doc_type, key.instance_id, key.marker_style) in conflicting_triples:
            issues.append(GroupIssue.CONFLICTING_MARKER_TOTAL)

        documents.append(
            ReconstructedDocument(
                doc_id=f"RD{index:03d}",
                doc_type=key.doc_type,
                key=key,
                expected_pages=expected_pages,
                observed_logical_pages=observed,
                missing_logical_pages=missing,
                duplicate_logical_pages=duplicates,
                page_links=page_links,
                start_physical_page=start_physical_page,
                end_physical_page=end_physical_page,
                completeness=completeness,
                issues=issues,
                warnings=[],
            )
        )

    return documents


# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------


def group_pages(
    results: list[PageResult],
    extracted_pages: list[ExtractedPage],
    rules: dict,
) -> tuple[list[PhysicalGroup], list[ReconstructedDocument]]:
    """Run identifier assignment, physical grouping, and reconstruction, in order.

    Identifiers must be assigned before reconstruction (which needs
    `instance_id` to tell tier 1 from tier 2); physical grouping doesn't
    depend on identifiers and could run first, but is kept after for
    readability -- both outputs are independent of each other's order.
    """
    assign_instance_ids(results, extracted_pages, rules)
    physical_groups = build_physical_groups(results)
    documents = reconstruct_documents(results, rules)
    return physical_groups, documents


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.extend("  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def physical_span_and_contiguity(physical_pages: list[int]) -> tuple[str, bool]:
    """Display-layer helper only: summarize a document's physical footprint.

    Does not touch `ReconstructedDocument` in any way -- `start_physical_page`
    and `end_physical_page` keep their field names and meaning (physical
    position of logical page 1 / of the last observed logical page). This
    just derives a min-max span string and a contiguity flag from the same
    `page_links` data, for the CLI table only.
    """
    if not physical_pages:
        return "-", False
    lo, hi = min(physical_pages), max(physical_pages)
    span = f"{lo}-{hi}" if lo != hi else str(lo)
    is_contiguous = set(physical_pages) == set(range(lo, hi + 1))
    return span, is_contiguous


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Group classified pages into physical runs and reconstructed "
        "logical documents."
    )
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument(
        "--rules", default="config/rules.yaml", help="Path to the rules YAML file."
    )
    args = parser.parse_args(argv)

    rules = load_rules(args.rules)
    pages = extract_pages(args.pdf_path, rules)
    results = [classify_page(page, rules) for page in pages]

    physical_groups, documents = group_pages(results, pages, rules)

    print("Physical groups:")
    headers = ["group_id", "doc_type", "start_page", "end_page", "page_count"]
    rows = [
        [g.group_id, g.doc_type.value, str(g.start_page), str(g.end_page), str(g.page_count)]
        for g in physical_groups
    ]
    print(_render_table(headers, rows))
    print()

    print("Reconstructed documents:")
    # Display-layer renaming only: "first_page_at"/"last_page_at" below are
    # ReconstructedDocument.start_physical_page/end_physical_page under a
    # more self-explanatory label. The schema field names are unchanged.
    headers2 = [
        "doc_id",
        "doc_type",
        "instance_id",
        "marker_style",
        "expected",
        "observed_count",
        "missing",
        "duplicate",
        "first_page_at",
        "last_page_at",
        "physical_span",
        "is_contiguous",
        "completeness",
        "issues",
    ]
    rows2 = []
    for d in documents:
        physical_pages = [pl.physical_page for pl in d.page_links]
        span, is_contiguous = physical_span_and_contiguity(physical_pages)
        rows2.append(
            [
                d.doc_id,
                d.doc_type.value,
                d.key.instance_id or "-",
                d.key.marker_style.value,
                str(d.expected_pages) if d.expected_pages is not None else "-",
                str(len(d.observed_logical_pages)),
                ",".join(str(p) for p in d.missing_logical_pages) or "-",
                ",".join(str(p) for p in d.duplicate_logical_pages) or "-",
                str(d.start_physical_page) if d.start_physical_page is not None else "-",
                str(d.end_physical_page) if d.end_physical_page is not None else "-",
                span,
                "Y" if is_contiguous else "N",
                d.completeness.value,
                ",".join(i.value for i in d.issues) or "-",
            ]
        )
    print(_render_table(headers2, rows2))
    print()

    for d in documents:
        mapping = ", ".join(
            f"{pl.logical_page}->{pl.physical_page}"
            for pl in sorted(d.page_links, key=lambda pl: (pl.logical_page or 0, pl.physical_page))
        )
        print(f"{d.doc_id} logical->physical: {mapping}")
    print()

    print(
        "Note: first_page_at may be greater than last_page_at. In a "
        "scatter-shuffled package the first logical page of a document can "
        "appear physically after its last page. This is the actual layout "
        "of the input, not an error."
    )
    print()

    orphans = [r for r in results if r.is_orphan]
    print("Orphan pages:")
    headers3 = ["page_number", "doc_type", "reason"]
    rows3 = [
        [
            str(r.page_number),
            r.doc_type.value,
            "no marker_page" if r.marker_page is None else "no instance_id, no marker style",
        ]
        for r in orphans
    ]
    print(_render_table(headers3, rows3))
    print()

    print("Summary:")
    print(f"  physical groups: {len(physical_groups)}")
    print(f"  reconstructed documents: {len(documents)}")
    print(f"  orphan pages: {len(orphans)}")


if __name__ == "__main__":
    main()
