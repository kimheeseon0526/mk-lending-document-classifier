"""Build a ground-truth CSV for a shuffled package PDF.

package_01 (and packages like it) is produced by concatenating several
original single-document PDFs and shuffling their pages into one file. Page
text survives that process byte-for-byte once whitespace layout is
normalized, so the original (label, source_page) of every shuffled page can
be recovered by hashing normalized page text and looking up the same hash
computed from the un-shuffled originals -- no separate answer key is needed.

PII BOUNDARY: this script reads full page text from every source and
shuffled PDF (to normalize and hash it) but never writes, logs, or returns
that text. Only SHA-256 hex digests, page numbers, and doc_type labels cross
into the CSV or stdout.

Does not classify. `doc_type` here comes from which original file a page
came from (ground truth), not from any heuristic -- unrelated to
`rule_classifier.classify_page`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path

# Running this file directly (`python scripts/build_ground_truth.py`) puts
# only `scripts/` on sys.path, not the project root, so `import src...` would
# otherwise fail. Running it under pytest does not need this -- pytest's own
# import machinery already puts the project root on sys.path because both
# `scripts/` and `tests/` are packages (have __init__.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf

from src.extract import normalize_text
from src.schema import DocType, GroundTruthEntry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure matching/validation logic -- no file I/O, testable with inline strings.
# ---------------------------------------------------------------------------


def hash_text(text: str) -> str:
    """SHA-256 hex digest of the whitespace-normalized text.

    Reuses `src.extract.normalize_text` rather than re-implementing
    whitespace collapsing, so this script's notion of "identical page text"
    stays identical to the one `extract.py` uses for length checks.
    """
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def build_source_index(source_texts: dict[str, list[str]]) -> dict[str, tuple[str, int]]:
    """Hash every original page and index it by hash.

    `source_texts` is `{label: [page_1_text, page_2_text, ...]}`, 1-indexed
    by list position. Returns `{hash: (label, source_page)}`.

    Raises ValueError if any hash occurs more than once across all sources
    (two original pages -- possibly in different documents -- normalize to
    identical text). The reconstruction from hash alone cannot tell those
    pages apart, so this must stop the run rather than silently pick one.
    """
    by_hash: dict[str, list[tuple[str, int]]] = {}
    for label, texts in source_texts.items():
        for source_page, text in enumerate(texts, start=1):
            by_hash.setdefault(hash_text(text), []).append((label, source_page))

    collisions = {h: pages for h, pages in by_hash.items() if len(pages) > 1}
    if collisions:
        detail = "; ".join(str(pages) for pages in collisions.values())
        raise ValueError(
            f"Source page hash collision(s) detected -- {len(collisions)} "
            f"hash value(s) shared by more than one original page: {detail}"
        )

    return {h: pages[0] for h, pages in by_hash.items()}


def match_shuffled_pages(
    shuffled_texts: list[str],
    index: dict[str, tuple[str, int]],
) -> list[tuple[int, str, int]]:
    """Match every shuffled page (1-indexed) to its (label, source_page).

    Returns a list of `(page_number, label, source_page)`, one per shuffled
    page.

    Raises ValueError if:
    - any shuffled page's hash is absent from `index` (unmatched), listing
      every unmatched page number, or
    - any `(label, source_page)` is claimed by more than one shuffled page
      (multi-match) -- in a page-preserving shuffle this should never
      happen, and if it does, the reconstruction cannot be trusted.
    """
    matches: list[tuple[int, str, int]] = []
    unmatched: list[int] = []
    claims: dict[tuple[str, int], list[int]] = {}

    for page_number, text in enumerate(shuffled_texts, start=1):
        hit = index.get(hash_text(text))
        if hit is None:
            unmatched.append(page_number)
            continue
        label, source_page = hit
        matches.append((page_number, label, source_page))
        claims.setdefault((label, source_page), []).append(page_number)

    if unmatched:
        raise ValueError(
            f"{len(unmatched)} shuffled page(s) matched no source page hash: "
            f"{unmatched}"
        )

    multi = {k: v for k, v in claims.items() if len(v) > 1}
    if multi:
        raise ValueError(
            f"Source page(s) matched by more than one shuffled page: {multi}"
        )

    return matches


def check_page_count(total_source_pages: int, shuffled_page_count: int) -> None:
    """Raise ValueError unless the source page total equals the shuffled count."""
    if total_source_pages != shuffled_page_count:
        raise ValueError(
            f"Total source page count ({total_source_pages}) does not equal "
            f"shuffled page count ({shuffled_page_count})"
        )


def check_page_numbers_complete(matches: list[tuple[int, str, int]], expected_count: int) -> None:
    """Raise ValueError unless matched page_numbers are exactly 1..expected_count."""
    page_numbers = sorted(m[0] for m in matches)
    expected = list(range(1, expected_count + 1))
    if page_numbers != expected:
        raise ValueError(
            f"Matched page_number values are not exactly 1..{expected_count} "
            f"with no duplicates; got {page_numbers}"
        )


# ---------------------------------------------------------------------------
# PDF I/O
# ---------------------------------------------------------------------------


def _read_page_texts(pdf_path: str | Path) -> list[str]:
    """Return raw `get_text()` output per physical page, in page order.

    Deliberately not `extract.extract_pages`: this script only needs text to
    hash, not markers, identifiers, or `ExtractedPage` validation, and
    should not require a `rules.yaml` to run.
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
        return [page.get_text() for page in doc]
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_source_arg(value: str) -> tuple[str, str]:
    """Parse one `--source LABEL=PATH` argument.

    Splits on the first `=` only, so paths containing `=` are not mangled.
    Raises ArgumentTypeError immediately if LABEL is not a DocType value --
    argparse reports this as a usage error rather than failing deep inside
    the matching logic.
    """
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"--source must be LABEL=PATH, got: {value!r}"
        )
    label, path = value.split("=", 1)
    if label not in DocType.__members__:
        valid = ", ".join(DocType.__members__)
        raise argparse.ArgumentTypeError(
            f"--source label {label!r} is not a valid DocType. Valid labels: {valid}"
        )
    return label, path


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Build a ground-truth CSV by hash-matching shuffled "
        "package pages back to their original source PDFs."
    )
    parser.add_argument("--shuffled", required=True, help="Path to the shuffled package PDF.")
    parser.add_argument(
        "--source",
        dest="sources",
        action="append",
        required=True,
        type=parse_source_arg,
        metavar="LABEL=PATH",
        help="Repeatable. LABEL must be a DocType value, e.g. "
        "URLA_1003=\"data/package_01/1003 - URLA_990145627.pdf\"",
    )
    parser.add_argument("--out", required=True, help="Output ground-truth CSV path.")
    args = parser.parse_args(argv)

    sources: dict[str, str] = {}
    for label, path in args.sources:
        if label in sources:
            raise ValueError(f"--source label {label!r} was passed more than once")
        sources[label] = path

    source_texts: dict[str, list[str]] = {
        label: _read_page_texts(path) for label, path in sources.items()
    }
    shuffled_texts = _read_page_texts(args.shuffled)

    total_source_pages = sum(len(texts) for texts in source_texts.values())
    check_page_count(total_source_pages, len(shuffled_texts))

    index = build_source_index(source_texts)
    matches = match_shuffled_pages(shuffled_texts, index)
    check_page_numbers_complete(matches, len(shuffled_texts))

    matches.sort(key=lambda m: m[0])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["page_number", "doc_type", "source_document", "source_page"])
        for page_number, label, source_page in matches:
            # Routed through GroundTruthEntry so the row is validated against
            # the real schema contract (page_number/source_page >= 1, doc_type
            # is a real DocType) before it ever reaches disk.
            entry = GroundTruthEntry(
                page_number=page_number,
                doc_type=DocType(label),
                source_document=Path(sources[label]).name,
                source_page=source_page,
            )
            writer.writerow(
                [entry.page_number, entry.doc_type.value, entry.source_document, entry.source_page]
            )

    print(f"shuffled pages: {len(shuffled_texts)}")
    for label, texts in source_texts.items():
        print(f"  source {label}: {len(texts)} pages ({Path(sources[label]).name})")

    print(
        "validations passed: page_count_match, source_hash_uniqueness, "
        "full_shuffled_match, no_multi_match, page_number_complete"
    )

    doc_type_counts: dict[str, int] = {}
    for _, label, _ in matches:
        doc_type_counts[label] = doc_type_counts.get(label, 0) + 1
    print("doc_type distribution:")
    for label in sorted(doc_type_counts):
        print(f"  {label}: {doc_type_counts[label]}")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
