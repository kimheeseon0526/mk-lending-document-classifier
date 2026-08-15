"""Page-level extraction for the mortgage document page classifier.

PII BOUNDARY: this module reads raw PDF text and raw document identifiers
(loan numbers, report IDs, etc.) and packs them into `ExtractedPage`
(`src/schema.py`), the only model allowed to carry that data. Nothing built
here does classification and nothing built here may be written to disk,
logged, or sent anywhere. Log statements in this module must stay limited to
page number, length, and marker style -- never raw text or identifier
values.

Does not classify. Only extracts text and structural signals (rotation,
image count, page markers, instance identifiers) so that downstream modules
(`rule_classifier.py`, `llm_classifier.py`, `grouping.py`) can work from a
single, PII-scoped representation.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pymupdf
import yaml
from pydantic import ValidationError

from src.schema import ExtractedPage, MarkerStyle

logger = logging.getLogger(__name__)


def load_rules(path: str | Path = "config/rules.yaml") -> dict:
    """Load the rule configuration from a YAML file.

    Raises whatever `open`/`yaml.safe_load` raises (e.g. FileNotFoundError,
    yaml.YAMLError), unmodified -- the built-in messages already include the
    offending path.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    """Collapse all whitespace runs so layout cannot affect length checks.

    `ExtractedPage.normalized_text_length` is the length of this result; it
    feeds the short-text delegation trigger, which must not fire just
    because a page happens to be laid out with lots of blank space.
    """
    return re.sub(r"\s+", "", text)


def detect_marker(text: str, rules: dict) -> tuple[MarkerStyle, int | None, int | None]:
    """Find the in-document page marker, if any, and its page/total numbers.

    Patterns are tried in ascending `priority` order and the first match
    that survives `marker_constraints` wins; no further patterns are
    attempted after that. Order matters here: "Page 7 of 11" also satisfies
    the generic "N of M" pattern, so the more specific PAGE_N_OF_M pattern
    must be tried first, otherwise it would be misclassified as N_OF_M.

    A match that violates a constraint (total too large, or page > total)
    is discarded and the loop continues to the next-priority pattern rather
    than giving up immediately -- a bad capture on a high-priority pattern
    should not suppress a good capture on a lower-priority one.
    """
    constraints = rules.get("marker_constraints", {})
    max_total = constraints.get("max_total")
    reject_if_page_gt_total = constraints.get("reject_if_page_gt_total", False)

    for marker_cfg in sorted(rules["markers"], key=lambda m: m["priority"]):
        flags = 0
        for flag_name in marker_cfg.get("flags", []):
            flags |= getattr(re, flag_name)

        pattern = re.compile(marker_cfg["regex"], flags)
        match = pattern.search(text)
        if match is None:
            continue

        groups = match.groupdict()
        # KeyError here means the pattern is missing the required 'page'
        # named group -- a rules.yaml authoring bug that must surface, not
        # be swallowed.
        page = int(groups["page"])
        total_str = groups.get("total")  # absent entirely for CLTA_PAGE_N
        total = int(total_str) if total_str is not None else None

        if total is not None and max_total is not None and total > max_total:
            continue
        if reject_if_page_gt_total and total is not None and page > total:
            continue

        return MarkerStyle(marker_cfg["style"]), page, total

    return MarkerStyle.NONE, None, None


def extract_identifiers(text: str, rules: dict) -> dict[str, str]:
    """Extract every configured instance identifier found on the page.

    Returns `{name: captured_value}` for whichever `instance_identifiers`
    entries matched; unmatched entries are simply absent, not None.

    `scope` is deliberately NOT applied here -- it restricts which doc_type
    an identifier is meaningful for, but classification hasn't happened yet
    at extraction time. All identifiers are extracted unconditionally; the
    grouping stage is responsible for deciding which of them to trust for a
    given page's assigned doc_type.
    """
    identifiers: dict[str, str] = {}
    for item in rules.get("instance_identifiers", []):
        match = re.search(item["regex"], text)
        if match:
            identifiers[item["name"]] = match.group(1)
    return identifiers


def extract_pages(pdf_path: str | Path, rules: dict) -> list[ExtractedPage]:
    """Extract one `ExtractedPage` per physical page of `pdf_path`.

    Raises:
        FileNotFoundError: `pdf_path` does not exist.
        RuntimeError: PyMuPDF failed to open the file (chained from the
            original exception).
        ValueError: the PDF is password-protected, or an extracted page
            fails `ExtractedPage` validation.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        doc = pymupdf.open(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF: {path}") from e

    try:
        if doc.needs_pass:
            raise ValueError(f"Encrypted PDF requires a password: {path}")

        pages: list[ExtractedPage] = []
        for index, page in enumerate(doc, start=1):
            # PyMuPDF's text layer is rotation-independent: get_text() returns
            # reading-order text regardless of page.rotation, so no rotation
            # correction is applied before/after extraction. Verified against
            # all 39 pages of package_01; not asserted to generalize to every
            # PDF producer.
            text = page.get_text()
            normalized_length = len(normalize_text(text))
            marker_style, marker_page, marker_total = detect_marker(text, rules)
            raw_identifiers = extract_identifiers(text, rules)

            if normalized_length == 0:
                logger.warning("page=%d normalized_length=0", index)
            else:
                logger.debug(
                    "page=%d normalized_length=%d marker_style=%s",
                    index,
                    normalized_length,
                    marker_style.value,
                )

            try:
                pages.append(
                    ExtractedPage(
                        page_number=index,
                        text=text,
                        normalized_text_length=normalized_length,
                        rotation=page.rotation,
                        image_count=len(page.get_images()),
                        marker_style=marker_style,
                        marker_page=marker_page,
                        marker_total=marker_total,
                        raw_identifiers=raw_identifiers,
                    )
                )
            except ValidationError as e:
                raise ValueError(
                    f"ExtractedPage validation failed on page {index}: {e}"
                ) from e

        return pages
    finally:
        doc.close()


def _format_table(pages: list[ExtractedPage]) -> str:
    """Render a summary table. Identifier VALUES are never included -- only
    the sorted key names, so the table cannot leak PII."""
    headers = [
        "page",
        "rotation",
        "norm_len",
        "images",
        "marker_style",
        "marker_page",
        "marker_total",
        "identifier_keys",
    ]
    rows = [
        [
            str(p.page_number),
            str(p.rotation),
            str(p.normalized_text_length),
            str(p.image_count),
            p.marker_style.value,
            str(p.marker_page) if p.marker_page is not None else "-",
            str(p.marker_total) if p.marker_total is not None else "-",
            ",".join(sorted(p.raw_identifiers.keys())) or "-",
        ]
        for p in pages
    ]

    widths = [
        max(len(header), max((len(row[i]) for row in rows), default=0))
        for i, header in enumerate(headers)
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.extend("  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Extract page-level text and structure signals from a PDF."
    )
    parser.add_argument("pdf_path", help="Path to the source PDF.")
    parser.add_argument(
        "--rules", default="config/rules.yaml", help="Path to the rules YAML file."
    )
    args = parser.parse_args(argv)

    rules = load_rules(args.rules)
    pages = extract_pages(args.pdf_path, rules)
    print(_format_table(pages))


if __name__ == "__main__":
    main()
