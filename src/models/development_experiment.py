"""Reproducible development-only candidate and calibration evidence.

This module accepts only explicitly supplied training and validation partitions.
It does not select a decision threshold, persist a model, or evaluate release
data. The command-line runner loads those partitions through the sealed
development loader.
"""

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.make_dataset import final_test_sealed
from src.evaluation.calibration import evaluate_calibration, select_and_fit_calibrator
from src.models.model_selection import select_champion_repeated_cv


@dataclass(frozen=True)
class DevelopmentExperimentConfig:
    """Recorded controls for a deterministic development experiment."""

    cv_splits: int = 5
    cv_repeats: int = 3
    calibration_outer_splits: int = 5
    calibration_inner_splits: int = 3
    random_state: int = 42
    n_jobs: int = 1


def _score_summary(labels: pd.Series, probabilities: np.ndarray) -> dict[str, object]:
    calibration = evaluate_calibration(labels, probabilities)
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "calibration": {
            "sample_count": calibration.sample_count,
            "brier_score": calibration.brier_score,
            "log_loss": calibration.log_loss,
            "expected_calibration_error": calibration.expected_calibration_error,
            "bins": [asdict(item) for item in calibration.bins],
        },
    }


@final_test_sealed()
def run_development_experiment(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    *,
    config: DevelopmentExperimentConfig = DevelopmentExperimentConfig(),
    candidates: Mapping[str, BaseEstimator] | None = None,
) -> dict[str, object]:
    """Return model-selection and calibration evidence without a release decision."""
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("Feature and label partitions must be aligned.")
    if not X_train.index.equals(y_train.index) or not X_validation.index.equals(
        y_validation.index
    ):
        raise ValueError("Feature and label indices must match in order.")
    if not X_train.index.is_unique or not X_validation.index.is_unique:
        raise ValueError("Partition indices must be unique.")
    if set(X_train.index) & set(X_validation.index):
        raise ValueError("Training and validation indices must be disjoint.")
    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError("Training and validation feature schemas must match.")
    if set(y_train.unique()) != {0, 1} or set(y_validation.unique()) != {0, 1}:
        raise ValueError("Training and validation labels must contain both binary classes.")

    selection = select_champion_repeated_cv(
        X_train,
        y_train,
        candidates=candidates,
        n_splits=config.cv_splits,
        n_repeats=config.cv_repeats,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    uncalibrated = clone(selection.champion_pipeline).fit(X_train, y_train)
    uncalibrated_probabilities = uncalibrated.predict_proba(X_validation)[:, 1]

    calibration_selection = select_and_fit_calibrator(
        selection.champion_pipeline,
        X_train,
        y_train,
        outer_splits=config.calibration_outer_splits,
        inner_splits=config.calibration_inner_splits,
        random_state=config.random_state,
    )
    calibrated_probabilities = calibration_selection.fitted_estimator.predict_proba(
        X_validation
    )[:, 1]

    leaderboard = selection.leaderboard.replace({np.nan: None}).to_dict(orient="records")
    return {
        "status": "development_only_not_release_evidence",
        "artifact_persisted": False,
        "decision_policy": "not_selected_business_costs_unapproved",
        "partition_rows": {
            "training": len(X_train),
            "validation": len(X_validation),
        },
        "configuration": asdict(config),
        "model_selection": {
            "primary_metric": selection.primary_metric,
            "selected_candidate": selection.champion_name,
            "leaderboard": leaderboard,
        },
        "uncalibrated_validation": _score_summary(
            y_validation, uncalibrated_probabilities
        ),
        "calibrated_validation": {
            "selected_method": calibration_selection.method,
            "mean_outer_brier": calibration_selection.mean_outer_brier,
            "outer_fold_brier": {
                method: list(scores)
                for method, scores in calibration_selection.fold_brier_scores.items()
            },
            **_score_summary(y_validation, calibrated_probabilities),
        },
    }
