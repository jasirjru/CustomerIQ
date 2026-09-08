"""
CustomerIQ — Baseline Model Training & Evaluation

Trains an interpretable Logistic Regression model as the benchmark.
All future candidate models (KNN, SVM, Decision Trees, Random Forests) must beat this baseline.
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE
from src.evaluation.metrics import compute_classification_metrics


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
    max_iter: int = 1000,
    class_weight: Any = None,
) -> LogisticRegression:
    """
    Train a Logistic Regression classification model.

    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        random_state: Deterministic random seed.
        max_iter: Maximum solver iterations for convergence.
        class_weight: Optional class weights (e.g. 'balanced' for imbalanced data).

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_baseline(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Tuple[Dict[str, float], pd.Series]:
    """
    Evaluate model on test dataset at a specified probability decision threshold.

    Args:
        model: Fitted LogisticRegression model.
        X_test: Test features.
        y_test: Test ground truth labels.
        threshold: Decision threshold for positive prediction (default: 0.5).

    Returns:
        (metrics_dict, y_pred)
    """
    # Predict probabilities for class 1 (churn)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Apply decision threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = compute_classification_metrics(y_test, y_pred, y_pred_proba)
    return metrics, y_pred
