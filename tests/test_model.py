"""
CustomerIQ — Tests for Machine Learning Models

Validates:
1. Model artifacts existence and deserialization.
2. Probability calibration properties (probabilities bounded in [0, 1] and sum to 1).
3. Risk monotonicity (high-risk features yield higher probability than low-risk features).
"""

import pytest
import numpy as np
import pandas as pd
import joblib

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import MODELS_DIR, DATA_PROCESSED


@pytest.fixture
def champion_model():
    model_path = MODELS_DIR / "champion_model.joblib"
    assert model_path.exists(), "Champion model artifact must exist."
    return joblib.load(model_path)


@pytest.fixture
def preprocessor():
    prep_path = MODELS_DIR / "preprocessor.joblib"
    assert prep_path.exists(), "Preprocessor artifact must exist."
    return joblib.load(prep_path)


def test_champion_model_attributes(champion_model):
    """Ensure champion model has expected classification interface."""
    assert hasattr(champion_model, "predict")
    assert hasattr(champion_model, "predict_proba")
    assert hasattr(champion_model, "feature_importances_")


def test_prediction_probabilities_bounded(champion_model, preprocessor):
    """Verify that predicted probabilities fall strictly between 0 and 1."""
    X_test_proc = pd.read_csv(DATA_PROCESSED / "X_test_processed.csv").head(20)
    probas = champion_model.predict_proba(X_test_proc)

    # Must have 2 classes (0 and 1)
    assert probas.shape[1] == 2
    # Probabilities bounded in [0, 1]
    assert (probas >= 0.0).all() and (probas <= 1.0).all()
    # Row probabilities sum to 1.0
    row_sums = probas.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


def test_relative_risk_monotonicity(champion_model, preprocessor):
    """
    Test that a customer with high-risk attributes receives a higher
    predicted churn probability than a customer with low-risk attributes.
    """
    high_risk_customer = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 99.0, "TotalCharges": 99.0,
    }])

    low_risk_customer = pd.DataFrame([{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 70, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 50.0, "TotalCharges": 3500.0,
    }])

    high_proc = pd.DataFrame(preprocessor.transform(high_risk_customer), columns=preprocessor.get_feature_names_out())
    low_proc = pd.DataFrame(preprocessor.transform(low_risk_customer), columns=preprocessor.get_feature_names_out())

    p_high = champion_model.predict_proba(high_proc)[0, 1]
    p_low = champion_model.predict_proba(low_proc)[0, 1]

    assert p_high > p_low
    assert p_high >= 0.60, f"Expected high-risk churn >= 0.60, got {p_high}"
    assert p_low <= 0.15, f"Expected low-risk churn <= 0.15, got {p_low}"
