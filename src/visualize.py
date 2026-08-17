"""Figures summarizing classification results: a per-page color strip and a
confusion-matrix heatmap.

PII BOUNDARY / DESIGN CONSTRAINT: this module never renders actual PDF page
content or thumbnails -- every page is represented as a single flat color
tile keyed only by its `DocType` label. That is a deliberate choice, not an
optimization: a thumbnail or rendered snippet could expose page text or
identifiers in an image file that this project has no PII-masking pipeline
for (unlike `LLMPageRequest.excerpt`, which goes through `mask_text`).
Reducing every page to "which of five labels" removes that risk by
construction rather than relying on care taken while drawing it.

Requires a headless-safe matplotlib backend -- `matplotlib.use("Agg")` is
called immediately after importing `matplotlib`, before `pyplot` is
imported, so this module works in CI/grader environments with no display.

All figure text is plain ASCII/English. No Korean (or other non-Latin)
text is placed in any figure, so it never depends on a CJK font being
installed on the machine that renders these images -- a missing font would
otherwise show as tofu boxes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
from matplotlib.patches import Rectangle  # noqa: E402

from src.schema import (  # noqa: E402
    DOC_TYPE_ORDER,
    DocType,
    EvaluationReport,
    GroundTruthEntry,
    PageResult,
)


def get_label_colors() -> dict[DocType, str]:
    """Fixed color per DocType label, shared by every figure this module draws.

    Colors are drawn from the Okabe-Ito palette, chosen because it stays
    distinguishable under the common forms of color vision deficiency
    (deuteranopia/protanopia/tritanopia) -- not just visually varied under
    typical vision. OTHER is pulled out to a neutral gray, per spec, rather
    than reusing one of the four data-bearing hues.
    """
    return {
        DocType.URLA_1003: "#0072B2",  # blue
        DocType.INCOME_DOC: "#E69F00",  # orange
        DocType.CREDIT_REPORT: "#009E73",  # bluish green
        DocType.TITLE_REPORT: "#CC79A7",  # reddish purple
        DocType.OTHER: "#999999",  # neutral gray
    }


def split_pages_into_rows(n_pages: int, max_per_row: int = 40) -> list[tuple[int, int]]:
    """Split physical pages 1..n_pages into consecutive (start, end) row
    ranges of at most `max_per_row` pages each.

    Pure function, no page count hardcoded -- package_02's page count is
    unknown ahead of time, so `render_page_strip` must size its rows from
    whatever `n_pages` turns out to be rather than assuming package_01's 39.
    """
    if n_pages <= 0:
        return []
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= n_pages:
        end = min(start + max_per_row - 1, n_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges


def render_page_strip(
    page_results: list[PageResult],
    ground_truth: list[GroundTruthEntry] | None,
    out_path: str | Path,
    title: str,
) -> Path:
    """Render a color-tile strip: one column per physical page, a
    Ground Truth row and a Prediction row (or Prediction only, when
    `ground_truth` is None -- package_02 has no ground truth, so this path
    is required, not optional). Wraps into multiple horizontal bands when
    there are more than 40 pages (see `split_pages_into_rows`).

    Every tile is a flat color from `get_label_colors()`; no page text,
    identifier, or rendered PDF content ever appears here (see module
    docstring). Mismatched pages get a bold black outline spanning both
    rows so a disagreement is visible without reading colors precisely.
    """
    colors = get_label_colors()
    out_path = Path(out_path)

    pred_by_page = {r.page_number: r.doc_type for r in page_results}
    truth_by_page = {t.page_number: t.doc_type for t in ground_truth} if ground_truth else None
    show_truth = truth_by_page is not None

    n_pages = max(pred_by_page) if pred_by_page else 0
    row_ranges = split_pages_into_rows(n_pages)
    n_rows = max(len(row_ranges), 1)
    n_bands = 2 if show_truth else 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 1.1 * n_rows + 1.2), squeeze=False)
    axes = axes[:, 0]

    for ax, (start, end) in zip(axes, row_ranges):
        width = end - start + 1
        for page in range(start, end + 1):
            x = page - start
            predicted = pred_by_page.get(page)
            actual = truth_by_page.get(page) if show_truth else None

            if predicted is not None:
                ax.add_patch(
                    Rectangle((x, 0), 1, 1, facecolor=colors[predicted], edgecolor="white", linewidth=0.5)
                )
            if show_truth and actual is not None:
                ax.add_patch(
                    Rectangle((x, 1), 1, 1, facecolor=colors[actual], edgecolor="white", linewidth=0.5)
                )

            if show_truth and actual is not None and predicted is not None and actual != predicted:
                ax.add_patch(
                    Rectangle(
                        (x, 0), 1, n_bands, fill=False, edgecolor="black", linewidth=2.2, zorder=5
                    )
                )

        ax.set_xlim(0, width)
        ax.set_ylim(0, n_bands)
        if show_truth:
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(["Prediction", "Ground Truth"], fontsize=8)
        else:
            ax.set_yticks([0.5])
            ax.set_yticklabels(["Prediction"], fontsize=8)

        tick_step = 1 if width <= 20 else 5
        tick_pages = list(range(start, end + 1, tick_step))
        ax.set_xticks([p - start + 0.5 for p in tick_pages])
        ax.set_xticklabels([str(p) for p in tick_pages], fontsize=7)
        ax.set_xlabel("physical page", fontsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    if show_truth:
        common = [p for p in truth_by_page if p in pred_by_page]
        if common:
            correct = sum(1 for p in common if truth_by_page[p] == pred_by_page[p])
            subtitle = f"{n_pages} pages | accuracy {correct / len(common):.4f}"
        else:
            subtitle = f"{n_pages} pages | accuracy N/A"
    else:
        subtitle = f"{n_pages} pages | no ground truth available"

    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", label=doc_type.value)
        for doc_type, color in colors.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(colors), fontsize=8, frameon=False)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def zero_support_note(report: EvaluationReport) -> str | None:
    """Build the zero-support caption for the confusion matrix, or None if
    every class has support > 0. Pulled from `report.per_class` -- never
    hardcodes which label (e.g. "OTHER") might be the one with no examples,
    since that is a fact about the dataset, not a fixed property of the code.
    """
    zero_support = [m.doc_type.value for m in report.per_class if m.support == 0]
    if not zero_support:
        return None
    return (
        f"Note: {', '.join(zero_support)} has support 0 in this dataset; "
        "recall and F1 are undefined (N/A) and excluded from macro-F1."
    )


def render_confusion_matrix(report: EvaluationReport, out_path: str | Path, title: str) -> Path:
    """Render `report.confusion_matrix` as a heatmap (rows=actual, cols=predicted).

    Both axes always list all five `DOC_TYPE_ORDER` labels, including any
    with zero support, so an unrepresented class still shows as an
    all-zero row rather than being silently dropped from the axis. Cell
    shading encodes count magnitude (what a confusion matrix needs to
    show); axis tick labels are colored with the same per-DocType palette
    `render_page_strip` uses, so a label reads as the same color across
    both figures even though a cell itself represents an (actual,
    predicted) pair and can't sensibly be a single label's flat color.
    Diagonal (correct) cells get a colored border and bold count text;
    off-diagonal (incorrect) cells get a thin neutral border.
    """
    colors = get_label_colors()
    out_path = Path(out_path)
    labels = list(DOC_TYPE_ORDER)
    n = len(labels)

    matrix = [[report.confusion_matrix[actual][predicted] for predicted in labels] for actual in labels]
    max_val = max((v for row in matrix for v in row), default=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(max_val, 1))

    ax.set_xticks(range(n))
    ax.set_xticklabels([label.value for label in labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label.value for label in labels], fontsize=8)
    for tick_label, label in zip(ax.get_xticklabels(), labels):
        tick_label.set_color(colors[label])
    for tick_label, label in zip(ax.get_yticklabels(), labels):
        tick_label.set_color(colors[label])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)

    for i in range(n):
        for j in range(n):
            value = matrix[i][j]
            on_diagonal = i == j
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#2c7fb8" if on_diagonal else "#bbbbbb",
                    linewidth=2.0 if on_diagonal else 0.5,
                )
            )
            text_color = "white" if value > max_val * 0.6 else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold" if on_diagonal else "normal",
            )

    note = zero_support_note(report)
    if note:
        fig.text(0.5, 0.01, note, ha="center", fontsize=7.5, wrap=True)

    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_all(
    page_results: list[PageResult],
    ground_truth: list[GroundTruthEntry] | None,
    report: EvaluationReport | None,
    out_dir: str | Path,
) -> list[Path]:
    """Render both figures into `out_dir` and return the paths written.

    Skips the confusion matrix (printing why) when `report` is None -- a
    confusion matrix has no meaning without an `EvaluationReport` to draw
    it from, and this must not fail silently.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        render_page_strip(
            page_results, ground_truth, out_dir / "page_strip.png", "Page Classification Strip"
        )
    ]

    if report is not None:
        paths.append(
            render_confusion_matrix(report, out_dir / "confusion_matrix.png", "Confusion Matrix")
        )
    else:
        print("confusion matrix skipped: no EvaluationReport available")

    return paths
