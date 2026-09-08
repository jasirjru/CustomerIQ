"""
CustomerIQ — Tests for FastAPI Endpoints & Pydantic Validation

Validates:
1. /health endpoint for monitoring probes.
2. /model-info endpoint for operational metadata.
3. /predict single-customer scoring.
4. /predict-batch bulk scoring.
5. HTTP 422 Unprocessable Entity for invalid or out-of-bound inputs.
"""

import pytest
from fastapi.testclient import TestClient

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.main import app


@pytest.fixture
def client():
    """Create a FastAPI test client using context manager to trigger lifespan startup."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.0,
    }


def test_health_endpoint(client):
    """Verify system health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True


def test_model_info_endpoint(client):
    """Verify operational model info endpoint."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "Random Forest" in data["model_name"]
    assert data["decision_threshold"] == 0.35
    assert data["engineered_features_count"] == 46


def test_predict_endpoint_valid_payload(client, valid_customer_payload):
    """Verify successful prediction contract and structure."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["churn_prediction"] in [0, 1]
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MODERATE", "HIGH"]
    assert data["decision_threshold"] == 0.35
    assert "customer_segment" in data
    assert len(data["top_risk_drivers"]) > 0
    assert "recommended_retention_action" in data


def test_predict_batch_endpoint(client, valid_customer_payload):
    """Verify bulk scoring on a list of payloads."""
    payload_batch = [valid_customer_payload, valid_customer_payload]
    response = client.post("/predict-batch", json=payload_batch)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


def test_input_validation_negative_tenure(client, valid_customer_payload):
    """Verify that negative tenure is rejected with HTTP 422."""
    bad_payload = valid_customer_payload.copy()
    bad_payload["tenure"] = -5  # Invalid constraint (ge=0)

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("tenure" in err["loc"] for err in errors)


def test_input_validation_missing_required_field(client, valid_customer_payload):
    """Verify that omitting a required field triggers HTTP 422."""
    bad_payload = valid_customer_payload.copy()
    del bad_payload["MonthlyCharges"]  # Required field omitted

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert any("MonthlyCharges" in err["loc"] for err in errors)
