"""Leakage-safe model selection and validation-only threshold optimization.

This module deliberately has no final-test evaluation function. Candidate
selection receives only the model-training partition, and threshold selection
receives only validation labels and probabilities. Final-test evaluation must
be a separate, explicitly invoked release step after all choices are frozen.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import (
    CV_N_REPEATS,
    CV_N_SPLITS,
    FALSE_NEGATIVE_COST,
    FALSE_POSITIVE_COST,
    MODEL_SELECTION_METRIC,
    RANDOM_STATE,
)
from src.features.build_features import build_preprocessor, get_feature_lists
from src.data.make_dataset import final_test_sealed


SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "f1": "f1",
    "balanced_accuracy": "balanced_accuracy",
}


@dataclass(frozen=True)
class ModelSelectionResult:
    """Repeated-CV leaderboard and the unfitted selected pipeline."""

    champion_name: str
    champion_pipeline: Pipeline
    leaderboard: pd.DataFrame
    primary_metric: str


@dataclass(frozen=True)
class ThresholdResult:
    """Decision threshold chosen exclusively from validation predictions."""

    threshold: float
    total_cost: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    balanced_accuracy: float
    false_negative_cost: float
    false_positive_cost: float


def get_selection_candidates(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    """Return candidate estimators used in repeated cross-validation.

    SVC is explicitly calibrated within each outer CV training fold so that it
    exposes probabilities without fitting probability calibration on held-out
    outer-fold observations.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="uniform",
            metric="minkowski",
        ),
        "Calibrated SVM": CalibratedClassifierCV(
            estimator=SVC(C=1.0, kernel="rbf", random_state=random_state),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=1,
        ),
    }


def build_raw_feature_pipeline(
    X_train: pd.DataFrame,
    estimator: BaseEstimator,
) -> Pipeline:
    """Build a pipeline whose preprocessing is refit inside every CV fold."""
    if isinstance(estimator, CalibratedClassifierCV):
        calibrated = clone(estimator)
        calibrated.estimator = build_raw_feature_pipeline(X_train, estimator.estimator)
        return Pipeline([("calibrated_classifier", calibrated)])
    numerical_features, categorical_features = get_feature_lists(X_train)
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(numerical_features, categorical_features),
            ),
            ("classifier", clone(estimator)),
        ]
    )


@final_test_sealed()
def select_champion_repeated_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: Mapping[str, BaseEstimator] | None = None,
    n_splits: int = CV_N_SPLITS,
    n_repeats: int = CV_N_REPEATS,
    random_state: int = RANDOM_STATE,
    primary_metric: str = MODEL_SELECTION_METRIC,
    n_jobs: int = 1,
) -> ModelSelectionResult:
    """Select a model using repeated stratified CV on model-training data only."""
    if primary_metric not in SCORING:
        raise ValueError(
            f"Unsupported primary metric '{primary_metric}'. "
            f"Choose one of {sorted(SCORING)}."
        )
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be >= 2 and n_repeats must be >= 1.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    candidate_estimators = dict(get_selection_candidates(random_state) if candidates is None else candidates)
    if not candidate_estimators:
        raise ValueError("At least one candidate estimator is required.")

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    rows = []
    pipelines: Dict[str, Pipeline] = {}

    for name, estimator in candidate_estimators.items():
        pipeline = build_raw_feature_pipeline(X_train, estimator)
        pipelines[name] = pipeline
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=n_jobs,
            return_train_score=False,
            error_score="raise",
        )
        row: Dict[str, Any] = {"Model": name, "CV_Folds": n_splits * n_repeats}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        rows.append(row)

    primary_mean = f"{primary_metric}_mean"
    primary_std = f"{primary_metric}_std"
    leaderboard = (
        pd.DataFrame(rows)
        .sort_values(
            by=[primary_mean, primary_std, "Model"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    champion_name = str(leaderboard.loc[0, "Model"])

    return ModelSelectionResult(
        champion_name=champion_name,
        champion_pipeline=pipelines[champion_name],
        leaderboard=leaderboard,
        primary_metric=primary_metric,
    )


def _threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    """Return exact decision boundaries induced by validation probabilities."""
    unique = np.unique(probabilities)
    no_positive_threshold = np.nextafter(float(unique[-1]), np.inf)
    return np.concatenate(([0.0], unique, [no_positive_threshold]))


def optimize_validation_threshold(
    y_validation: Sequence[int],
    validation_probabilities: Sequence[float],
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> ThresholdResult:
    """Minimize stated classification cost using validation data only.

    Ties are resolved by higher recall and then by the higher threshold. The
    The caller must supply validation data; array names alone cannot establish
    provenance. The development runner enforces file access restrictions.
    """
    y_true = np.asarray(y_validation)
    probabilities = np.asarray(validation_probabilities, dtype=float)

    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("Validation labels and probabilities must be equal-length 1D arrays.")
    if len(y_true) == 0:
        raise ValueError("Validation data must not be empty.")
    if set(np.unique(y_true)) - {0, 1}:
        raise ValueError("Validation labels must be binary values 0 or 1.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Validation probabilities must be finite values in [0, 1].")
    if not np.isfinite([false_negative_cost, false_positive_cost]).all() or false_negative_cost < 0 or false_positive_cost < 0:
        raise ValueError("Misclassification costs must be non-negative.")

    evaluations: list[Tuple[float, float, int, int, int, int, float]] = []
    for threshold in _threshold_candidates(probabilities):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        total_cost = float(fn * false_negative_cost + fp * false_positive_cost)
        recall = float(recall_score(y_true, predictions, zero_division=0))
        evaluations.append(
            (total_cost, -recall, -float(threshold), int(tn), int(fp), int(fn), int(tp))
        )

    total_cost, _, negative_threshold, tn, fp, fn, tp = min(evaluations)
    threshold = -negative_threshold
    predictions = (probabilities >= threshold).astype(int)

    return ThresholdResult(
        threshold=float(threshold),
        total_cost=total_cost,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        true_negatives=tn,
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
        accuracy=float(accuracy_score(y_true, predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, predictions)),
        false_negative_cost=float(false_negative_cost),
        false_positive_cost=float(false_positive_cost),
    )


@final_test_sealed()
def fit_champion_and_select_threshold(
    selection: ModelSelectionResult,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
    false_positive_cost: float = FALSE_POSITIVE_COST,
) -> Tuple[Pipeline, ThresholdResult]:
    """Fit the selected pipeline on training data and tune only on validation."""
    fitted_pipeline = clone(selection.champion_pipeline)
    fitted_pipeline.fit(X_train, y_train)
    validation_probabilities = fitted_pipeline.predict_proba(X_validation)[:, 1]
    threshold = optimize_validation_threshold(
        y_validation,
        validation_probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    return fitted_pipeline, threshold
