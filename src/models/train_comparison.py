"""
CustomerIQ — Model Comparison Suite

Trains and evaluates 5 classical Machine Learning algorithms on identical splits:
1. Logistic Regression (Baseline)
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest

Provides structured metrics comparison and cross-model visualization.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import RANDOM_STATE
from src.evaluation.metrics import compute_classification_metrics


def get_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Initialize dictionary of the 5 comparison models with sensible default/tuned hyperparameters.

    Returns:
        Dict[str, Any]: Map of model_name -> un-fitted estimator.
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
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            probability=True,  # Required for predict_proba and ROC-AUC
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,       # Pruning to prevent severe overfitting
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
            n_jobs=-1,
        ),
    }


def train_and_evaluate_all(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Train all models and evaluate on test set.

    Args:
        models: Dictionary of model name -> estimator.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        threshold: Decision threshold for classification (default: 0.5).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
            - results_df: Comparison metrics DataFrame.
            - fitted_models: Dictionary of fitted models.
            - test_probas: Dictionary of predicted probabilities for class 1.
    """
    results = []
    fitted_models = {}
    test_probas = {}

    for name, model in models.items():
        # Fit on training data
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            # Fallback for models without predict_proba
            df_vals = model.decision_function(X_test)
            y_proba = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min())
        else:
            y_proba = None

        test_probas[name] = y_proba

        # Make binary predictions
        if y_proba is not None:
            y_pred = (y_proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)

        # Compute standardized metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        results.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1_Score": metrics["f1_score"],
            "ROC_AUC": metrics["roc_auc"],
        })

    results_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False).reset_index(drop=True)
    return results_df, fitted_models, test_probas
