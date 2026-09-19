"""Global model-importance utilities; local attribution is not implemented."""

from typing import Any, List
import pandas as pd
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
    X_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute global permutation importance on an explicitly supplied evaluation set.

    Why Permutation Importance?
    - Evaluated on caller-supplied data (normally development validation data).
    - Measures actual performance drop when a feature's values are shuffled.
    - Not biased toward high-cardinality continuous features unlike MDI.

    Args:
        model: Fitted classifier.
        X_evaluation: Evaluation feature matrix. Do not pass the reserved final test
            during development.
        y_evaluation: Evaluation labels.
        scoring: Metric to evaluate performance degradation.
        n_repeats: Number of times to permute each feature.
        random_state: Random seed.

    Returns:
        pd.DataFrame: Sorted permutation importance scores with standard deviations.
    """
    perm = permutation_importance(
        model,
        X_evaluation,
        y_evaluation,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "Feature": X_evaluation.columns,
        "Importance_Mean": perm.importances_mean,
        "Importance_Std": perm.importances_std,
    }).sort_values(by="Importance_Mean", ascending=False).reset_index(drop=True)

    return df_perm


def explain_single_prediction(*args, **kwargs):
    """Legacy API disabled: global MDI cannot yield local signed contributions."""
    raise NotImplementedError(
        "Local attribution is unavailable. Use global importance as a population "
        "summary only; no customer-specific or causal explanation is supported."
    )
