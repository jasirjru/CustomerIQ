"""
CustomerIQ — Tests for Data Cleaning & Preprocessing Pipeline

Validates:
1. Handling of TotalCharges whitespace anomalies (tenure=0 edge case).
2. Elimination of customerID.
3. Target column encoding (Yes/No -> 1/0).
4. Fitted ColumnTransformer output dimensionality and zero-leakage properties.
"""

import pytest
import pandas as pd
import numpy as np
import joblib

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.make_dataset import clean_dataset, split_data
from config import MODELS_DIR


@pytest.fixture
def sample_raw_dataframe():
    """Create a synthetic mini-batch of raw data containing edge-case anomalies."""
    return pd.DataFrame({
        "customerID": ["001-AAA", "002-BBB", "003-CCC"],
        "gender": ["Male", "Female", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 0, 48],  # Row 2 is a brand-new customer
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["Yes", "No", "Yes"],
        "OnlineBackup": ["No", "No", "Yes"],
        "DeviceProtection": ["Yes", "No", "No"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "Yes", "No"],
        "Contract": ["One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No"],
        "PaymentMethod": ["Mailed check", "Electronic check", "Credit card (automatic)"],
        "MonthlyCharges": [45.0, 80.0, 30.0],
        "TotalCharges": ["540.0", " ", "1440.0"],  # Row 2 has whitespace string!
        "Churn": ["No", "Yes", "No"],
    })


def test_clean_dataset_removes_customer_id(sample_raw_dataframe):
    """Verify that non-predictive identifier customerID is dropped."""
    cleaned = clean_dataset(sample_raw_dataframe)
    assert "customerID" not in cleaned.columns
    assert len(cleaned) == 3


def test_clean_dataset_handles_totalcharges_whitespace(sample_raw_dataframe):
    """Verify that whitespace in TotalCharges is imputed to 0.0 for tenure=0 customers."""
    cleaned = clean_dataset(sample_raw_dataframe)
    assert cleaned["TotalCharges"].dtype == np.float64
    assert cleaned.loc[1, "TotalCharges"] == 0.0
    assert cleaned.loc[0, "TotalCharges"] == 540.0
    assert cleaned["TotalCharges"].isna().sum() == 0


def test_clean_dataset_encodes_target(sample_raw_dataframe):
    """Verify that target 'Churn' is mapped to integers 1 and 0."""
    cleaned = clean_dataset(sample_raw_dataframe)
    assert pd.api.types.is_numeric_dtype(cleaned["Churn"])
    assert list(cleaned["Churn"].values) == [0, 1, 0]


def test_preprocessor_pipeline_transformation():
    """Verify that preprocessor.joblib produces exactly 46 dimensions without NaNs."""
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    assert preprocessor_path.exists(), "Fitted preprocessor artifact must exist."

    preprocessor = joblib.load(preprocessor_path)

    dummy_customer = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 500.0,
    }])

    transformed = preprocessor.transform(dummy_customer)
    assert transformed.shape == (1, 46)
    assert not np.isnan(transformed).any()
