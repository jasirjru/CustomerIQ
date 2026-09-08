"""
CustomerIQ — PCA Visualization & Dimensionality Reduction Utilities

Provides PCA fitting, explained variance analysis, 2D cluster projection,
and feature loading / biplot visualizations.
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def fit_pca_projection(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[PCA, np.ndarray]:
    """
    Fit PCA on standardized features and project onto n principal components.

    Args:
        X: Standardized feature matrix.
        n_components: Target number of dimensions (default: 2 for 2D visualization).
        random_state: Random seed for deterministic SVD solver.

    Returns:
        (fitted_pca, projected_X)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_projected = pca.fit_transform(X)
    return pca, X_projected


def plot_explained_variance(
    pca_full: PCA,
    save_path: str = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Plot Scree plot and Cumulative Explained Variance curve.

    Args:
        pca_full: PCA fitted with multiple components.
        save_path: Optional path to save the generated figure.

    Returns:
        (fig, (ax1, ax2))
    """
    exp_var = pca_full.explained_variance_ratio_ * 100
    cum_var = np.cumsum(exp_var)
    components = np.arange(1, len(exp_var) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Individual variance per component (Scree Plot)
    ax1.bar(components, exp_var, color="#3498db", edgecolor="black", alpha=0.85)
    ax1.plot(components, exp_var, color="#2980b9", marker="o", lw=2)
    ax1.set_title("Scree Plot: Individual Explained Variance", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Principal Component", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # 2. Cumulative explained variance
    ax2.plot(components, cum_var, color="#27ae60", marker="s", lw=2.5)
    ax2.axhline(y=80, color="red", linestyle="--", label="80% Threshold")
    ax2.axhline(y=90, color="orange", linestyle=":", label="90% Threshold")
    ax2.set_title("Cumulative Explained Variance", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Principal Components", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, (ax1, ax2)


def plot_2d_pca_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[int, str] = None,
    title: str = "2D PCA Projection",
    palette: List[str] = None,
    save_path: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D scatter of customers along PC1 and PC2.

    Args:
        X_2d: 2D array of shape (N, 2) containing PC1 and PC2 coordinates.
        labels: Grouping labels (e.g. cluster labels or churn labels).
        label_names: Mapping of label integer -> descriptive text.
        title: Title of the scatter plot.
        palette: Colors for classes.
        save_path: Optional path to save image.

    Returns:
        (fig, ax)
    """
    if palette is None:
        palette = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = np.unique(labels)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names and label in label_names else f"Class {label}"
        color = palette[i % len(palette)]
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=color,
            label=name,
            alpha=0.6,
            edgecolors="none",
            s=25,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=10, markerscale=1.8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
