"""
Visualisation module for analysing and visualising path signatures and audio features.
"""

from typing import List, Optional, Tuple
import json
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def _set_style():
    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        plt.style.use("seaborn")
    sns.set_context("paper")
    sns.set_palette("colorblind")


def _limit_ticks(ax: plt.Axes, max_ticks: int = 40) -> None:
    # Reduce tick labels to avoid overlap
    def _downsample(labels: List[str], max_n: int) -> List[str]:
        n = len(labels)
        if n <= max_n:
            return labels
        step = max(1, n // max_n)
        return [lbl if (i % step == 0) else "" for i, lbl in enumerate(labels)]

    ax.set_xticklabels(
        _downsample([t.get_text() for t in ax.get_xticklabels()], max_ticks),
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels(
        _downsample([t.get_text() for t in ax.get_yticklabels()], max_ticks)
    )


def perform_pca(data: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    # Ensure n_components doesn't exceed number of features
    n_components = min(n_components, data.shape[1])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components, pca


def visualise_signatures(
    principal_components: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Path Signature Visualisation",
    save_path: Optional[str] = None,
) -> None:
    if principal_components.shape[1] != 2:
        raise ValueError("Principal components must have exactly 2 dimensions")
    if labels is not None and len(labels) != len(principal_components):
        raise ValueError("Number of labels must match number of samples")

    _set_style()
    plt.figure(figsize=(8, 6))

    if labels is None:
        plt.scatter(
            principal_components[:, 0], principal_components[:, 1], alpha=0.6, s=12
        )
    else:
        unique_labels = list(set(labels))
        palette = sns.color_palette(n_colors=len(unique_labels))
        for label, color in zip(unique_labels, palette):
            mask = [l == label for l in labels]
            plt.scatter(
                principal_components[mask, 0],
                principal_components[mask, 1],
                label=label,
                color=color,
                alpha=0.6,
                s=12,
            )
        plt.legend(frameon=False, fontsize=8, ncol=2)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_song_genres(json_path: str) -> Tuple[list, list, dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        songs = json.load(f)
    if songs and "name" in songs[0]:
        song_names = [s["name"] for s in songs]
        main_genres = [
            s["genre"][0] if isinstance(s["genre"], list) and s["genre"] else "Unknown"
            for s in songs
        ]
        name_to_genres = {s["name"]: s["genre"] for s in songs}
    elif songs and "title" in songs[0]:
        song_names = [s["title"] for s in songs]
        main_genres = [s["genre"] if s["genre"] else "Unknown" for s in songs]
        name_to_genres = {s["title"]: [s["genre"]] if s["genre"] else [] for s in songs}
    else:
        raise ValueError(
            "JSON file must contain either 'name' or 'title' field for songs"
        )
    return song_names, main_genres, name_to_genres


def plot_similarity_matrix(
    sim_matrix: np.ndarray,
    song_names: list,
    save_path: str = None,
    title: str = "Similarity Matrix",
) -> None:
    if sim_matrix.size == 0:
        return
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    show_labels = len(song_names) <= 60
    sns.heatmap(
        sim_matrix,
        xticklabels=song_names if show_labels else False,
        yticklabels=song_names if show_labels else False,
        cmap="plasma",
        annot=False,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)
    if show_labels:
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_distribution(
    categories: list, save_path: str = None, title: str = "Category Distribution"
) -> None:
    if not categories:
        return
    counts = Counter(categories)
    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))
    _set_style()
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color="skyblue")
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list = None,
    save_path: str = None,
    title: str = "Confusion Matrix",
) -> None:
    if not y_true or not y_pred:
        return
    if class_names is None:
        class_names = sorted(list(set(list(y_true) + list(y_pred))))
    cm = sk_confusion_matrix(y_true, y_pred, labels=class_names)
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Ensure proper color scaling - use raw counts with explicit vmin/vmax
    cm_max = cm.max() if cm.max() > 0 else 1
    cm_min = cm.min()

    # Use a diverging or sequential colormap that shows variation better
    # If all values are the same, we'll still see that, but with proper scaling
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        vmin=cm_min,
        vmax=cm_max,
        square=False,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_embedding(
    features: np.ndarray,
    labels: list = None,
    method: str = "pca",
    save_path: str = None,
    title: str = "Feature Embedding Visualisation",
) -> None:
    if features.size == 0:
        return
    _set_style()
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        n_samples = features.shape[0]
        perplexity = min(30, max(5, n_samples // 3))
        reducer = TSNE(
            n_components=2, random_state=2025, perplexity=perplexity, init="pca"
        )
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    embedding = reducer.fit_transform(features)
    plt.figure(figsize=(8, 6))
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=12)
    else:
        unique_labels = list(set(labels))
        palette = sns.color_palette(n_colors=len(unique_labels))
        for label, color in zip(unique_labels, palette):
            mask = [l == label for l in labels]
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                label=label,
                color=color,
                alpha=0.7,
                s=12,
            )
        plt.legend(
            frameon=False,
            fontsize=8,
            ncol=1,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
