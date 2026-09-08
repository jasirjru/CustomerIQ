"""
CustomerIQ — Data Loading and Cleaning Pipeline

Responsible for:
1. Loading raw customer data.
2. Cleaning data quality anomalies (e.g., whitespace in TotalCharges).
3. Splitting into stratified train and test sets to prevent data leakage.
"""

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
)


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
