"""
Interface contract for the mortgage document page classifier.

This module is the single source of truth for every data structure that crosses
a module boundary. It is owned by the integrator only; `extract.py`,
`rule_classifier.py`, `llm_classifier.py`, `pipeline.py`, `grouping.py`,
`evaluate.py` and `visualize.py` import from here and must not redefine these
types locally.

Security boundary
-----------------
`ExtractedPage` is the only model that carries raw page text or raw document
identifiers. It never leaves the process. Everything that may be written to
`outputs/`, committed to Git, or sent to an external API must use the
de-identified models below (`PageResult`, `PhysicalGroup`,
`ReconstructedDocument`, `RunReport`, `LLMPageRequest`).

Requires Python 3.11+ and pydantic>=2.7.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DocType(str, Enum):
    """Official assessment labels. Never abbreviate these strings anywhere."""

    URLA_1003 = "URLA_1003"
    INCOME_DOC = "INCOME_DOC"
    CREDIT_REPORT = "CREDIT_REPORT"
    TITLE_REPORT = "TITLE_REPORT"
    OTHER = "OTHER"


#: Fixed label order for confusion matrices and report columns.
DOC_TYPE_ORDER: tuple[DocType, ...] = (
    DocType.URLA_1003,
    DocType.INCOME_DOC,
    DocType.CREDIT_REPORT,
    DocType.TITLE_REPORT,
    DocType.OTHER,
)


class MarkerStyle(str, Enum):
    """Shape of the in-document page marker found on a page.

    Retained separately from the parsed numbers because URLA_1003 and
    CREDIT_REPORT both use a total of 11 in package_01. The style is what
    distinguishes them, so collapsing `Page N of M` and `N of M` into one
    pattern would discard a usable signal and could merge two distinct
    documents during reconstruction.
    """

    PAGE_N_OF_M = "PAGE_N_OF_M"  # "Page 7 of 11"
    N_OF_M = "N_OF_M"            # "7 of 11"
    CLTA_PAGE_N = "CLTA_PAGE_N"  # "CLTA Preliminary Report Form ... Page 4"
    NONE = "NONE"


class DecisionSource(str, Enum):
    """Which stage produced the final label for a page."""

    RULE = "RULE"                    # rule classifier was confident enough
    LLM = "LLM"                      # delegated to the LLM and it answered
    RULE_FALLBACK = "RULE_FALLBACK"  # delegated, LLM failed, rule result kept
    CONTEXT = "CONTEXT"              # inferred from reconstruction context


class Completeness(str, Enum):
    """Whether a reconstructed document appears to have all of its pages."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNKNOWN_EXTENT = "UNKNOWN_EXTENT"  # no marker total, so not judgeable


class GroupIssue(str, Enum):
    """Additional anomalies. Orthogonal to `Completeness`; a document can be
    INCOMPLETE and carry DUPLICATE_PAGE at the same time."""

    DUPLICATE_PAGE = "DUPLICATE_PAGE"
    AMBIGUOUS_ATTACHMENT = "AMBIGUOUS_ATTACHMENT"
    CONFLICTING_MARKER_TOTAL = "CONFLICTING_MARKER_TOTAL"


class PipelineMode(str, Enum):
    """Evaluation arms. All three share one ground truth and one metric set."""

    RULE_ONLY = "rule-only"
    LLM_ONLY = "llm-only"
    HYBRID = "hybrid"


class WarningCode(str, Enum):
    """Stable warning identifiers. Free-text warnings are not used so that
    failure paths can be asserted in tests and counted in RunReport."""

    EMPTY_TEXT = "EMPTY_TEXT"
    SHORT_TEXT = "SHORT_TEXT"
    NO_SIGNATURE = "NO_SIGNATURE"
    LOW_RULE_SCORE = "LOW_RULE_SCORE"
    NARROW_RULE_MARGIN = "NARROW_RULE_MARGIN"
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"
    LLM_DISABLED = "LLM_DISABLED"
    LLM_CALL_FAILED = "LLM_CALL_FAILED"
    LLM_PARSE_FAILED = "LLM_PARSE_FAILED"
    LLM_LABEL_INVALID = "LLM_LABEL_INVALID"
    LLM_COUNT_MISMATCH = "LLM_COUNT_MISMATCH"
    ORPHAN_PAGE = "ORPHAN_PAGE"
    MARKER_TOTAL_CONFLICT = "MARKER_TOTAL_CONFLICT"


Score = Annotated[float, Field(ge=0.0, le=1.0)]
PageNumber = Annotated[int, Field(ge=1)]


# ---------------------------------------------------------------------------
# Internal-only model (carries raw text and raw identifiers)
# ---------------------------------------------------------------------------


class ExtractedPage(BaseModel):
    """In-process representation of a single physical PDF page.

    NEVER serialize this to `outputs/`, a log file, or an API request body.
    `text` and `raw_identifiers` contain PII.
    """

    model_config = ConfigDict(extra="forbid")

    page_number: PageNumber
    text: str
    normalized_text_length: int = Field(
        ge=0,
        description="Length after collapsing all whitespace. Used for the "
        "short-text delegation trigger so that layout does not affect it.",
    )
    rotation: int = Field(description="0, 90, 180 or 270 as reported by PyMuPDF.")
    image_count: int = Field(default=0, ge=0)

    marker_style: MarkerStyle = MarkerStyle.NONE
    marker_page: Optional[int] = Field(default=None, ge=1)
    marker_total: Optional[int] = Field(default=None, ge=1)

    raw_identifiers: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'loan_number': '...', 'prelim_number': '...'}. "
        "Used only to build a run-local instance_id. Never persisted.",
    )

    @model_validator(mode="after")
    def _check_marker_consistency(self) -> "ExtractedPage":
        if self.marker_style is MarkerStyle.NONE and self.marker_page is not None:
            raise ValueError("marker_page set while marker_style is NONE")
        if (
            self.marker_page is not None
            and self.marker_total is not None
            and self.marker_page > self.marker_total
        ):
            raise ValueError(
                f"marker_page {self.marker_page} exceeds marker_total {self.marker_total}"
            )
        return self


# ---------------------------------------------------------------------------
# LLM boundary (de-identified request, structured response)
# ---------------------------------------------------------------------------


class LLMPageRequest(BaseModel):
    """One page as sent to the LLM. Built from masked, truncated text only.

    `excerpt` must already have been through the masking rules in
    `config/rules.yaml`. The masking preserves classification signal
    (occupation names, form titles, monetary structure, tokens such as
    `CTEC #`) while removing identifiers (SSN, loan number, report ID,
    account numbers, phone, email, address, personal names).
    """

    model_config = ConfigDict(extra="forbid")

    page_id: PageNumber
    excerpt: str = Field(
        description="Masked and length-capped text. Not the full page body."
    )
    marker_style: MarkerStyle = MarkerStyle.NONE
    marker_hint: Optional[str] = Field(
        default=None,
        description="e.g. 'page 4 of 11'. Structural hint only, no identifiers.",
    )


class LLMPageDecision(BaseModel):
    """One page decision returned by the LLM under Structured Outputs.

    `extra='forbid'` and the DocType enum together reject any label outside
    the five official strings.
    """

    model_config = ConfigDict(extra="forbid")

    page_id: PageNumber
    doc_type: DocType
    confidence: Score = Field(
        description="The model's self-reported confidence. NOT comparable to "
        "rule_score; the two are on different scales and must not be mixed."
    )
    evidence: str = Field(
        default="",
        max_length=200,
        description="Short justification. Must not echo identifiers.",
    )


class LLMBatchResponse(BaseModel):
    """Full structured response for one batch request."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[LLMPageDecision] = Field(default_factory=list)

    def page_ids(self) -> set[int]:
        return {d.page_id for d in self.decisions}


class LLMCallRecord(BaseModel):
    """Audit trail for one LLM call. Contains no prompt or response text."""

    model_config = ConfigDict(extra="forbid")

    batch_index: int = Field(ge=0)
    requested_page_ids: list[PageNumber] = Field(default_factory=list)
    returned_page_ids: list[PageNumber] = Field(default_factory=list)
    attempt_count: int = Field(default=1, ge=1)
    succeeded: bool = False
    warnings: list[WarningCode] = Field(default_factory=list)
    latency_ms: Optional[int] = Field(default=None, ge=0)


class LLMRuntimeConfig(BaseModel):
    """Resolved LLM settings. `api_key` is never stored on this model."""

    model_config = ConfigDict(extra="forbid")

    provider: str = "openai"
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=2, ge=0)
    batch_size: int = Field(default=10, ge=1)
    max_excerpt_chars: int = Field(default=1200, ge=100)
    prompt_version: str = "v1"
    enabled: bool = Field(
        default=False,
        description="False when no API key is present. The pipeline then runs "
        "in rule-only mode and records LLM_DISABLED instead of raising.",
    )


# ---------------------------------------------------------------------------
# Page-level result (safe to persist)
# ---------------------------------------------------------------------------


class PageResult(BaseModel):
    """Final per-page classification. Safe to write to CSV/JSON and commit.

    Contains no page text and no raw identifiers.
    """

    model_config = ConfigDict(extra="forbid")

    page_number: PageNumber
    doc_type: DocType
    decision_source: DecisionSource

    # --- rule evidence -----------------------------------------------------
    rule_doc_type: Optional[DocType] = Field(
        default=None, description="Rule top-1 before any LLM delegation."
    )
    rule_score: Score = Field(
        default=0.0,
        description="Heuristic evidence score, not a calibrated probability.",
    )
    rule_margin: Score = Field(
        default=0.0, description="share(top1) - share(top2)."
    )
    rule_scores: dict[DocType, float] = Field(default_factory=dict)
    strong_signature_matched: bool = False
    matched_signals: list[str] = Field(
        default_factory=list,
        description="Signature strings that fired. These come from "
        "config/rules.yaml, not from page content, so they carry no PII.",
    )

    # --- llm evidence ------------------------------------------------------
    llm_called: bool = False
    llm_failed: bool = False
    llm_doc_type: Optional[DocType] = None
    llm_confidence: Optional[Score] = None

    # --- structure ---------------------------------------------------------
    marker_style: MarkerStyle = MarkerStyle.NONE
    marker_page: Optional[int] = Field(default=None, ge=1)
    marker_total: Optional[int] = Field(default=None, ge=1)
    instance_id: Optional[str] = Field(
        default=None,
        description="Run-local surrogate such as 'CREDIT_REPORT#1'. Derived "
        "from raw identifiers but not reversible to them.",
    )
    is_orphan: bool = Field(
        default=False,
        description="Classified but not attachable to any reconstructed "
        "document. Assignment state of the page, not a document status.",
    )

    rotation: int = 0
    normalized_text_length: int = Field(default=0, ge=0)
    warnings: list[WarningCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_source_consistency(self) -> "PageResult":
        if self.decision_source is DecisionSource.LLM and not self.llm_called:
            raise ValueError("decision_source=LLM but llm_called is False")
        if self.decision_source is DecisionSource.RULE_FALLBACK and not self.llm_failed:
            raise ValueError("decision_source=RULE_FALLBACK but llm_failed is False")
        if self.llm_failed and not self.llm_called:
            raise ValueError("llm_failed is True but llm_called is False")
        return self


# ---------------------------------------------------------------------------
# Grouping — primary output
# ---------------------------------------------------------------------------


class PhysicalGroup(BaseModel):
    """A run of consecutive physical pages sharing a predicted doc_type.

    This is the assessment's required grouping output. On a scatter-shuffled
    package this legitimately yields one group per page; that is the structure
    of the input, not a failure of the algorithm.
    """

    model_config = ConfigDict(extra="forbid")

    group_id: str
    doc_type: DocType
    start_page: PageNumber
    end_page: PageNumber
    page_count: int = Field(ge=1)
    physical_pages: list[PageNumber] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_run(self) -> "PhysicalGroup":
        if self.end_page < self.start_page:
            raise ValueError("end_page precedes start_page")
        expected = list(range(self.start_page, self.end_page + 1))
        if self.physical_pages != expected:
            raise ValueError("physical_pages must be the contiguous start..end run")
        if self.page_count != len(expected):
            raise ValueError("page_count does not match the run length")
        return self


# ---------------------------------------------------------------------------
# Grouping — extension output
# ---------------------------------------------------------------------------


class GroupingKey(BaseModel):
    """Composite key used to reconstruct a logical document.

    All four components are required because no single one is sufficient:
    - `doc_type` alone merges the four documents of one loan file
    - `instance_id` alone merges URLA_1003 and CREDIT_REPORT, which share the
      same loan number in package_01 (14 CREDIT pages and 4 URLA pages)
    - `marker_total` alone merges them again, since both report a total of 11
    - `marker_style` is what actually separates `Page N of M` from `N of M`
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_type: DocType
    instance_id: Optional[str] = None
    marker_style: MarkerStyle = MarkerStyle.NONE
    marker_total: Optional[int] = Field(default=None, ge=1)


class PageLink(BaseModel):
    """One page's attachment to a reconstructed document."""

    model_config = ConfigDict(extra="forbid")

    physical_page: PageNumber
    logical_page: Optional[int] = Field(default=None, ge=1)
    evidence: list[str] = Field(
        default_factory=list,
        description="Named reasons such as 'marker_total_match'. Deliberately "
        "not a numeric score; there is no data to calibrate one against.",
    )


class ReconstructedDocument(BaseModel):
    """A logical document rebuilt from scattered pages.

    Extension output. It supplements `PhysicalGroup` and never replaces it.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    doc_type: DocType
    key: GroupingKey

    expected_pages: Optional[int] = Field(default=None, ge=1)
    observed_logical_pages: list[int] = Field(default_factory=list)
    missing_logical_pages: list[int] = Field(default_factory=list)
    duplicate_logical_pages: list[int] = Field(default_factory=list)
    page_links: list[PageLink] = Field(default_factory=list)

    start_physical_page: Optional[PageNumber] = Field(
        default=None,
        description="Physical position of logical page 1, when identifiable.",
    )
    end_physical_page: Optional[PageNumber] = Field(
        default=None,
        description="Physical position of the last logical page, when identifiable.",
    )

    completeness: Completeness
    issues: list[GroupIssue] = Field(default_factory=list)
    warnings: list[WarningCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_completeness(self) -> "ReconstructedDocument":
        if self.expected_pages is None:
            if self.completeness is not Completeness.UNKNOWN_EXTENT:
                raise ValueError(
                    "expected_pages is None so completeness must be UNKNOWN_EXTENT"
                )
        else:
            want = Completeness.COMPLETE if not self.missing_logical_pages else Completeness.INCOMPLETE
            if self.completeness is not want:
                raise ValueError(
                    f"completeness {self.completeness} contradicts "
                    f"missing_logical_pages {self.missing_logical_pages}"
                )
        if self.duplicate_logical_pages and GroupIssue.DUPLICATE_PAGE not in self.issues:
            raise ValueError("duplicate pages present but DUPLICATE_PAGE issue missing")
        return self


# ---------------------------------------------------------------------------
# Run report — instrumentation of every quiet failure path
# ---------------------------------------------------------------------------


class RunReport(BaseModel):
    """Execution summary. Always written, even on a fully successful run.

    The counters exist so that a degraded run is visibly degraded rather than
    silently producing plausible output.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    mode: PipelineMode
    source_pdf_name: str = Field(
        description="Basename only. Never a full local path."
    )
    total_pages: int = Field(ge=0)

    rule_resolved: int = Field(default=0, ge=0)
    llm_target_pages: list[PageNumber] = Field(
        default_factory=list,
        description="Pages selected for delegation. Recorded even when the "
        "LLM is disabled, so a rule-only run still shows what would have "
        "been delegated.",
    )
    llm_called: int = Field(default=0, ge=0)
    llm_resolved: int = Field(default=0, ge=0)
    rule_fallback_count: int = Field(default=0, ge=0)
    llm_parse_failures: int = Field(default=0, ge=0)
    llm_call_records: list[LLMCallRecord] = Field(default_factory=list)

    empty_text_pages: list[PageNumber] = Field(default_factory=list)
    orphan_pages: list[PageNumber] = Field(default_factory=list)

    llm_config: Optional[LLMRuntimeConfig] = None
    rules_version: Optional[str] = None
    warnings: list[WarningCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_counts(self) -> "RunReport":
        if self.llm_resolved + self.rule_fallback_count > self.llm_called:
            raise ValueError(
                "llm_resolved + rule_fallback_count exceeds llm_called"
            )
        if not self.llm_config or not self.llm_config.enabled:
            if self.llm_called and WarningCode.LLM_DISABLED in self.warnings:
                raise ValueError("LLM_DISABLED recorded but calls were made")
        return self


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class ClassMetrics(BaseModel):
    """Per-class metrics. `None` means undefined, not zero.

    OTHER has support 0 in package_01, so its recall and F1 are undefined and
    must be rendered as N/A rather than 0.0.
    """

    model_config = ConfigDict(extra="forbid")

    doc_type: DocType
    support: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None


class EvaluationReport(BaseModel):
    """Metrics for one pipeline mode against one ground truth file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    mode: PipelineMode
    dataset_name: str
    total_pages: int = Field(ge=0)

    accuracy: float
    per_class: list[ClassMetrics] = Field(default_factory=list)
    macro_f1_supported: float = Field(
        description="Mean F1 over classes with support > 0 only."
    )
    supported_classes: list[DocType] = Field(default_factory=list)

    confusion_matrix: dict[DocType, dict[DocType, int]] = Field(
        default_factory=dict,
        description="Always all five labels on both axes, including OTHER.",
    )
    misclassified_pages: list[PageNumber] = Field(default_factory=list)

    model_name: Optional[str] = None
    prompt_version: Optional[str] = None
    rules_version: Optional[str] = None
    run_report: Optional[RunReport] = None


class GroundTruthEntry(BaseModel):
    """One row of `data/ground_truth.csv`. Label only; carries no PII."""

    model_config = ConfigDict(extra="forbid")

    page_number: PageNumber
    doc_type: DocType
    source_document: Optional[str] = Field(
        default=None, description="Origin PDF basename, for provenance."
    )
    source_page: Optional[int] = Field(default=None, ge=1)


__all__ = [
    "SCHEMA_VERSION",
    "DocType",
    "DOC_TYPE_ORDER",
    "MarkerStyle",
    "DecisionSource",
    "Completeness",
    "GroupIssue",
    "PipelineMode",
    "WarningCode",
    "ExtractedPage",
    "LLMPageRequest",
    "LLMPageDecision",
    "LLMBatchResponse",
    "LLMCallRecord",
    "LLMRuntimeConfig",
    "PageResult",
    "PhysicalGroup",
    "GroupingKey",
    "PageLink",
    "ReconstructedDocument",
    "RunReport",
    "ClassMetrics",
    "EvaluationReport",
    "GroundTruthEntry",
]
