"""
CustomerIQ — Data Loading and Cleaning Pipeline

Responsible for:
1. Loading raw customer data.
2. Cleaning data quality anomalies (e.g., whitespace in TotalCharges).
3. Splitting into stratified train and test sets to prevent data leakage.
"""

from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import (
    RAW_DATASET_PATH,
    TARGET_COLUMN,
    COLUMNS_TO_DROP,
    TEST_SIZE,
    RANDOM_STATE,
    FINAL_TEST_SIZE,
    VALIDATION_SIZE_WITHIN_DEVELOPMENT,
    DATA_PROCESSED,
)


_sealed_data = ContextVar("sealed_customeriq_data", default=False)


def _deny_reserved_data(event, args):
    """Reject filesystem access before opening historical test or raw data."""
    if event == "open" and _sealed_data.get() and isinstance(args[0], (str, bytes)):
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        name = path.name.lower()
        if (name.startswith(("x_test", "y_test", "final_test"))
                or name == "telco_customer_churn.csv"):
            raise PermissionError("Reserved final-test/full-raw data access is sealed.")


sys.addaudithook(_deny_reserved_data)


@contextmanager
def final_test_sealed():
    """Guard development runs; the full raw CSV also contains reserved labels."""
    token = _sealed_data.set(True)
    try:
        yield
    finally:
        _sealed_data.reset(token)


def load_development_partitions():
    """Split the previously materialized 80% development pool, without test IO.

    Original CSV row order is preserved, so seed 42 reproduces the earlier
    4,225/1,409 development split. Do not regenerate it from the full raw CSV.
    """
    with final_test_sealed():
        X = pd.read_csv(DATA_PROCESSED / "X_train_raw.csv")
        y = pd.read_csv(DATA_PROCESSED / "y_train.csv").squeeze("columns")
        if len(X) != len(y) or set(y.unique()) != {0, 1}:
            raise ValueError("Development features/labels are not aligned binary data.")
        return train_test_split(
            X, y, test_size=VALIDATION_SIZE_WITHIN_DEVELOPMENT,
            random_state=RANDOM_STATE, stratify=y,
        )


@dataclass(frozen=True)
class EvaluationPartitions:
    """Isolated partitions for model selection and final evaluation."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_final_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_final_test: pd.Series


def load_raw_data(filepath: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """
    Load raw customer dataset from disk.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    return pd.read_csv(filepath)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data quality issues discovered during EDA:
    1. Convert 'TotalCharges' from object to numeric.
       - Replace whitespace with 0.0 (these rows correspond to tenure=0).
    2. Drop non-predictive identifiers (e.g. 'customerID').
    3. Convert target 'Churn' from ('Yes'/'No') to binary (1/0).

    Args:
        df: Raw DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with correct types.
    """
    df_clean = df.copy()

    # Drop identifiers if present
    drop_cols = [col for col in COLUMNS_TO_DROP if col in df_clean.columns]
    if drop_cols:
        df_clean = df_clean.drop(columns=drop_cols)

    # Clean TotalCharges: convert whitespace to NaN, then fill with 0.0 (tenure=0)
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        # For tenure == 0 customers, they haven't been billed yet; fill NaN with 0.0
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Encode Target: Yes -> 1, No -> 0
    if TARGET_COLUMN in df_clean.columns and not pd.api.types.is_numeric_dtype(df_clean[TARGET_COLUMN]):
        df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    return df_clean


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the cleaned dataset into training and testing partitions using stratified sampling.

    Why Stratified Sampling?
    Our target class is imbalanced (~26.5% churn). Stratification guarantees
    that both the train and test splits retain the identical proportion of churners,
    preventing sampling bias.

    Args:
        df: Cleaned dataframe containing features and target.
        target_column: Name of the target column.
        test_size: Proportion of data allocated to the test split.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # CRITICAL: preserves class ratio in both splits
    )

    return X_train, X_test, y_train, y_test


def split_train_validation_test(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    final_test_size: float = FINAL_TEST_SIZE,
    validation_size_within_development: float = VALIDATION_SIZE_WITHIN_DEVELOPMENT,
    random_state: int = RANDOM_STATE,
) -> EvaluationPartitions:
    """Create isolated train, validation, and final-test partitions.

    The final test set is separated first and must not be passed to model
    selection or threshold optimization. The validation fraction is expressed
    relative to the remaining development data. With the default values this
    produces a 60/20/20 allocation.

    All splits are stratified by the binary target and retain their original
    indices so callers and tests can verify that the partitions are disjoint.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")
    if not 0.0 < final_test_size < 1.0:
        raise ValueError("final_test_size must be strictly between 0 and 1.")
    if not 0.0 < validation_size_within_development < 1.0:
        raise ValueError(
            "validation_size_within_development must be strictly between 0 and 1."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_development, X_final_test, y_development, y_final_test = train_test_split(
        X,
        y,
        test_size=final_test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_development,
        y_development,
        test_size=validation_size_within_development,
        random_state=random_state,
        stratify=y_development,
    )

    return EvaluationPartitions(
        X_train=X_train,
        X_validation=X_validation,
        X_final_test=X_final_test,
        y_train=y_train,
        y_validation=y_validation,
        y_final_test=y_final_test,
    )
