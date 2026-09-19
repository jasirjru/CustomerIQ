"""
CustomerIQ — Feature Engineering & Preprocessing Pipeline

Constructs reproducible Scikit-Learn ColumnTransformer and Pipelines.
Rules:
- FIT ONLY ON TRAINING DATA to eliminate Data Leakage.
- TRANSFORM test/unseen data using the statistics learned from training data.
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_feature_lists(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical column names from the feature DataFrame.
    
    Args:
        X: Feature DataFrame.
        
    Returns:
        (numerical_features, categorical_features)
    """
    # Numerical features are int or float
    numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Categorical features are object or category
    # Pandas 3 includes string dtype in object selection only as a temporary
    # compatibility behavior. Name it explicitly so pandas 4 keeps the schema.
    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # Note: SeniorCitizen is already 0/1 integer, so it might be treated as numerical or categorical.
    # If we want to treat SeniorCitizen as categorical:
    if "SeniorCitizen" in numerical_features:
        numerical_features.remove("SeniorCitizen")
        categorical_features.append("SeniorCitizen")

    return sorted(numerical_features), sorted(categorical_features)


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build a Scikit-Learn ColumnTransformer for automated, leakage-free preprocessing.

    - Numerical: StandardScaler (centers mean to 0, variance to 1)
    - Categorical: OneHotEncoder (handle_unknown='ignore', sparse_output=False)

    Why handle_unknown='ignore'?
    In a real production API, if a user submits an unprecedented category value,
    'ignore' encodes all one-hot columns as 0 rather than raising an unhandled exception.

    Args:
        numerical_features: List of continuous feature names.
        categorical_features: List of discrete/categorical feature names.

    Returns:
        ColumnTransformer: An un-fitted Scikit-Learn ColumnTransformer.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # Drop any unexpected leftover columns
    )

    return preprocessor


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    numerical_features: List[str],
    categorical_features: List[str]
) -> List[str]:
    """
    Retrieve human-readable column names after one-hot encoding.
    Useful for feature importance, correlation inspection, and model explainability.

    Args:
        preprocessor: Fitted ColumnTransformer.
        numerical_features: List of numerical columns.
        categorical_features: List of categorical columns.

    Returns:
        List[str]: Clean list of all output feature names.
    """
    try:
        # scikit-learn >= 1.0 supports get_feature_names_out()
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        return numerical_features + encoded_cat_names
