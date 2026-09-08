"""
CustomerIQ — Model Evaluation Utilities

Provides standardized metrics computation and visualization for classification models.
Ensures identical, fair evaluation across baseline and future comparison models.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute standard binary classification performance metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for class 1 (churn).

    Returns:
        Dict[str, float]: Dictionary of rounded evaluation metrics.
    """
    metrics = {
        "accuracy": float(round(accuracy_score(y_true, y_pred), 4)),
        "precision": float(round(precision_score(y_true, y_pred, zero_division=0), 4)),
        "recall": float(round(recall_score(y_true, y_pred, zero_division=0), 4)),
        "f1_score": float(round(f1_score(y_true, y_pred, zero_division=0), 4)),
    }

    if y_pred_proba is not None:
        metrics["roc_auc"] = float(round(roc_auc_score(y_true, y_pred_proba), 4))
    else:
        metrics["roc_auc"] = None

    return metrics


def plot_confusion_matrix_custom(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a rich, publication-ready Confusion Matrix with clear business labels.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Title for the figure.
        save_path: Optional path to save the plot figure.

    Returns:
        (fig, ax)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, fraction=0.046, pad=0.04)

    classes = ["No Churn (0)", "Churn (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=11)

    # Detailed cell annotations
    labels = [
        [f"True Negative (TN)\n{tn}\n(Stayed & Pred Stayed)", f"False Positive (FP)\n{fp}\n(Stayed but Pred Churn)"],
        [f"False Negative (FN)\n{fn}\n(Churned but Pred Stayed)\n⚠️ High Cost!", f"True Positive (TP)\n{tp}\n(Churned & Caught!)"]
    ]

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_roc_curve_custom(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str = "Model",
    save_path: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC Curve with AUC score and baseline random guess reference.

    Args:
        y_true: Ground truth labels.
        y_pred_proba: Predicted probabilities for class 1.
        model_name: Name of model for legend.
        save_path: Optional path to save figure.

    Returns:
        (fig, ax)
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"{model_name} (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax
