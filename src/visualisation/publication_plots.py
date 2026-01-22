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
    title: str,
    out_path: Path,
) -> None:
    set_publication_style()
    errs_low = np.array(means) - np.array(ci_lows)
    errs_high = np.array(ci_highs) - np.array(means)
    errs = [errs_low, errs_high]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=errs, capsize=4)
    ax.set_ylabel(ylabel)
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
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for title
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bar_with_cis to %s", out_path)


def heatmap_matrix(
    matrix: np.ndarray,
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    set_publication_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=False,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
    )
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
