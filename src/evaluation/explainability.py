"""
CustomerIQ — Explainable Machine Learning (XAI) Utilities

Provides Global and Local model interpretability:
1. Global: Mean Decrease in Impurity (MDI) feature importances.
2. Global: Permutation Feature Importance on held-out test data.
3. Local: Per-customer churn driver decomposition for actionable frontline retention.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier


def compute_mdi_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Extract Gini Importance / Mean Decrease in Impurity (MDI) from Random Forest.

    Args:
        model: Fitted RandomForestClassifier.
        feature_names: List of all input feature column names.

    Returns:
        pd.DataFrame: Sorted feature importances.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return df_imp


def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute Permutation Feature Importance on the test set.

    Why Permutation Importance?
    - Evaluated on HELD-OUT test data (not training data).
    - Measures actual performance drop when a feature's values are shuffled.
    - Not biased toward high-cardinality continuous features unlike MDI.

    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: Test ground-truth labels.
        scoring: Metric to evaluate performance degradation.
        n_repeats: Number of times to permute each feature.
        random_state: Random seed.

    Returns:
        pd.DataFrame: Sorted permutation importance scores with standard deviations.
    """
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std,
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    return df_perm


def explain_single_prediction(
    model: RandomForestClassifier,
    customer_vector: pd.Series,
    feature_names: List[str],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Provide local, per-customer churn risk drivers.
    Identifies which features pushed this specific customer toward churn.

    Args:
        model: Fitted RandomForestClassifier.
        customer_vector: 1D array/series of standardized customer features.
        feature_names: List of all feature names.
        top_n: Number of key factors to extract.

    Returns:
        Dict: Predicted churn probability, risk level, and top drivers.
    """
    x_input = customer_vector.values.reshape(1, -1)
    churn_proba = float(model.predict_proba(x_input)[0, 1])

    # Feature contribution approximation based on feature value * global importance
    importances = model.feature_importances_
    contributions = customer_vector.values * importances

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": customer_vector.values,
        "Weighted_Signal": contributions,
    }).sort_values(by="Weighted_Signal", ascending=False)

    top_risk_factors = contrib_df.head(top_n).to_dict(orient="records")

    risk_category = "HIGH" if churn_proba >= 0.5 else "MODERATE" if churn_proba >= 0.35 else "LOW"

    return {
        "churn_probability": round(churn_proba, 4),
        "risk_level": risk_category,
        "top_drivers": top_risk_factors,
    }
