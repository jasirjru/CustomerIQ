"""Calibration measurement and training-only calibrator selection.

Nothing in this module reads or accepts a reserved final-test partition.  The
selection helper uses nested cross-validation on development training data and
returns an in-memory estimator; it never replaces a production artifact.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from src.data.make_dataset import final_test_sealed


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_rate: float


@dataclass(frozen=True)
class CalibrationReport:
    sample_count: int
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    bins: tuple[CalibrationBin, ...]


@dataclass(frozen=True)
class CalibrationSelectionResult:
    """Training-only selection evidence plus the fitted chosen calibrator."""

    method: str
    mean_outer_brier: float
    fold_brier_scores: dict[str, tuple[float, ...]]
    fitted_estimator: BaseEstimator


def _validated_binary_inputs(
    y_true: Sequence[int], probabilities: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("Labels and probabilities must be equal-length 1D arrays.")
    if len(labels) == 0:
        raise ValueError("Calibration data must not be empty.")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Labels must contain only binary values 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return labels.astype(int), scores


def evaluate_calibration(
    y_true: Sequence[int], probabilities: Sequence[float], n_bins: int = 10
) -> CalibrationReport:
    """Measure probability quality without implying that calibration was fitted."""
    labels, scores = _validated_binary_inputs(y_true, probabilities)
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or not 2 <= n_bins <= 100:
        raise ValueError("n_bins must be an integer between 2 and 100.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.searchsorted(edges, scores, side="right") - 1, n_bins - 1)
    bins: list[CalibrationBin] = []
    weighted_error = 0.0
    for index in range(n_bins):
        mask = assignments == index
        count = int(mask.sum())
        if count == 0:
            continue
        mean_probability = float(scores[mask].mean())
        observed_rate = float(labels[mask].mean())
        weighted_error += count * abs(mean_probability - observed_rate)
        bins.append(
            CalibrationBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_probability=mean_probability,
                observed_rate=observed_rate,
            )
        )

    return CalibrationReport(
        sample_count=len(labels),
        brier_score=float(brier_score_loss(labels, scores)),
        log_loss=float(log_loss(labels, scores, labels=[0, 1])),
        expected_calibration_error=float(weighted_error / len(labels)),
        bins=tuple(bins),
    )


@final_test_sealed()
def select_and_fit_calibrator(
    base_estimator: BaseEstimator,
    X_train: Any,
    y_train: Sequence[int],
    methods: Sequence[str] = ("none", "sigmoid", "isotonic"),
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> CalibrationSelectionResult:
    """Select calibration by nested CV using development training data only.

    The uncalibrated estimator is an explicit candidate so calibration must
    demonstrate an improvement. The lowest mean outer-fold Brier score wins;
    ties prefer no calibration, then sigmoid, then isotonic. The selected
    estimator is then fitted on all supplied training data.
    """
    labels = np.asarray(y_train)
    if labels.ndim != 1 or len(labels) != len(X_train) or len(labels) == 0:
        raise ValueError("X_train and y_train must be non-empty and aligned.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Training labels must contain both binary classes.")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must be at least 2.")
    allowed = {"none", "sigmoid", "isotonic"}
    unique_methods = tuple(dict.fromkeys(methods))
    if not unique_methods or set(unique_methods) - allowed:
        raise ValueError("Calibration methods must be none, sigmoid, and/or isotonic.")
    if int(np.bincount(labels.astype(int)).min()) < max(outer_splits, inner_splits):
        raise ValueError("Each class needs at least max(outer_splits, inner_splits) rows.")

    outer_cv = StratifiedKFold(
        n_splits=outer_splits, shuffle=True, random_state=random_state
    )
    fold_scores: dict[str, list[float]] = {method: [] for method in unique_methods}
    for train_indices, validation_indices in outer_cv.split(np.zeros(len(labels)), labels):
        if hasattr(X_train, "iloc"):
            X_outer_train = X_train.iloc[train_indices]
            X_outer_validation = X_train.iloc[validation_indices]
        else:
            values = np.asarray(X_train)
            X_outer_train = values[train_indices]
            X_outer_validation = values[validation_indices]
        y_outer_train = labels[train_indices]
        y_outer_validation = labels[validation_indices]

        for method in unique_methods:
            if method == "none":
                estimator = clone(base_estimator).fit(X_outer_train, y_outer_train)
            else:
                inner_cv = StratifiedKFold(
                    n_splits=inner_splits, shuffle=True, random_state=random_state
                )
                estimator = CalibratedClassifierCV(
                    estimator=clone(base_estimator),
                    method=method,
                    cv=inner_cv,
                    ensemble=False,
                ).fit(X_outer_train, y_outer_train)
            probabilities = estimator.predict_proba(X_outer_validation)[:, 1]
            fold_scores[method].append(
                float(brier_score_loss(y_outer_validation, probabilities))
            )

    means = {method: float(np.mean(scores)) for method, scores in fold_scores.items()}
    preference = {"none": 0, "sigmoid": 1, "isotonic": 2}
    chosen_method = min(unique_methods, key=lambda method: (means[method], preference[method]))
    if chosen_method == "none":
        fitted = clone(base_estimator).fit(X_train, labels)
    else:
        final_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=random_state
        )
        fitted = CalibratedClassifierCV(
            estimator=clone(base_estimator),
            method=chosen_method,
            cv=final_cv,
            ensemble=False,
        ).fit(X_train, labels)
    return CalibrationSelectionResult(
        method=chosen_method,
        mean_outer_brier=means[chosen_method],
        fold_brier_scores={key: tuple(value) for key, value in fold_scores.items()},
        fitted_estimator=fitted,
    )
