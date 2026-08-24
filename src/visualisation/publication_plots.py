"""
Publication-quality plotting utilities with consistent styling and annotations.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger_config import setup_logger


logger = setup_logger("publication_plots")


def set_publication_style():
    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        plt.style.use("seaborn")
    sns.set_context("paper")
    sns.set_palette("colorblind")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def bar_with_cis(
    labels: List[str],
    means: List[float],
    ci_lows: List[float],
    ci_highs: List[float],
    ylabel: str,
    out_path: Path,
    title: Optional[str] = None,
    highlight_label: Optional[str] = None,
    sort_descending: bool = True,
) -> None:
    """Plot bars with 95% CI error bars.

    Args:
        highlight_label: if given and present in `labels`, that bar is
            drawn in a distinct accent colour while every other bar shares
            a neutral colour -- used to call out the dissertation's own
            proposed method against baseline methods.
        sort_descending: if True (default), bars are ordered by descending
            `means` rather than the order `labels` were passed in
            (typically alphabetical), which is otherwise an arbitrary and
            uninformative ordering for a results comparison figure.
    """
    set_publication_style()

    if sort_descending:
        order = sorted(range(len(labels)), key=lambda i: means[i], reverse=True)
        labels = [labels[i] for i in order]
        means = [means[i] for i in order]
        ci_lows = [ci_lows[i] for i in order]
        ci_highs = [ci_highs[i] for i in order]

    errs_low = np.array(means) - np.array(ci_lows)
    errs_high = np.array(ci_highs) - np.array(means)
    errs = [errs_low, errs_high]

    # Neutral colour for baseline methods, distinct accent colour for the
    # dissertation's own proposed method (if identified via
    # `highlight_label`) so it stands out from the baselines it's compared
    # against, rather than every bar being a uniform, undifferentiated blue.
    baseline_color = "#8C9BAB"  # neutral slate grey-blue
    highlight_color = "#D65F2C"  # distinct warm accent
    if highlight_label is not None and highlight_label in labels:
        bar_colors = [
            highlight_color if label == highlight_label else baseline_color
            for label in labels
        ]
    else:
        bar_colors = None

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=errs, capsize=4, color=bar_colors)
    ax.set_ylabel(ylabel)
    if title:
        # Single-panel figures are captioned in the surrounding LaTeX
        # (examiner G-05: an internal title duplicates a caption that
        # already describes the figure), so no title is drawn unless the
        # caller explicitly asks for one.
        ax.set_title(title, pad=20)
    
    # Calculate annotation offset and adjust ylim to accommodate annotations
    max_err = max(errs_high) if len(errs_high) > 0 else 0.02
    max_value = max(means) if len(means) > 0 else 1.0
    annotation_offset = max(0.01, min(0.03, max_value * 0.05))
    max_bar_height = max(means) if len(means) > 0 else 1.0
    max_err_height = max(errs_high) if len(errs_high) > 0 else 0.02
    # Set ylim to accommodate annotations with extra space
    y_max = max_bar_height + max_err_height + annotation_offset + max_value * 0.1
    ax.set_ylim(bottom=0, top=y_max)
    
    # Rotate x-axis labels to vertical to prevent overlap
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    
    for plot_bar, mean, err_h in zip(bars, means, errs_high):
        # Place annotation higher to avoid intersection with error bar cap
        ax.text(
            plot_bar.get_x() + plot_bar.get_width() / 2,
            plot_bar.get_height() + err_h + annotation_offset,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if bar_colors is not None:
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=highlight_color, label=highlight_label),
            plt.Rectangle((0, 0), 1, 1, color=baseline_color, label="Baseline methods"),
        ]
        ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for title
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bar_with_cis to %s", out_path)


def heatmap_matrix(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    out_path: Path,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    annotations: Optional[np.ndarray] = None,
) -> None:
    """Plot a heatmap, optionally with per-cell text annotations.

    ``annotations``, if given, must be a string array the same shape as
    ``matrix`` and is drawn in each cell instead of relying on colour
    alone to convey the value (examiner F-09/R-08/R-09: colour-only
    encoding cannot distinguish a handful of very small p-values, and
    conveys magnitude but not direction).
    """

    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annotations if annotations is not None else False,
        fmt="" if annotations is not None else "g",
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
    )
    if title:
        # Single-panel figures are captioned in the surrounding LaTeX
        # (examiner G-05), so no title is drawn unless explicitly requested.
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap_matrix to %s", out_path)


def multi_histograms(
    data_dict: Dict[str, np.ndarray],
    title: str,
    out_path: Path,
    bins: int = 20,
    cols: int = 3,
) -> None:
    set_publication_style()
    keys = list(data_dict.keys())
    n = len(keys)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4))
    axes = np.array(axes).reshape(rows, cols)
    for idx, key in enumerate(keys):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        arr = np.asarray(data_dict[key])
        ax.hist(arr, bins=bins, alpha=0.85)
        ax.set_title(key)
    # hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved multi_histograms to %s", out_path)
