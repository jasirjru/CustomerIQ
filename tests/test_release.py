"""Release regression coverage using synthetic requests, never reserved rows."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.manifest import load_verified_artifact, read_manifest
from src.api.schemas import CustomerPayload, INTERNET_ADDONS
from src.api.security import SecuritySettings
from src.api.service import ModelService


@pytest.fixture(scope="module")
def engine():
    return ModelService()


@pytest.fixture
def payload():
    # Synthetic fixture, not a customer record or final-test observation.
    return dict(gender="Female", SeniorCitizen=0, Partner="No", Dependents="No",
                tenure=2, PhoneService="Yes", MultipleLines="No",
                InternetService="Fiber optic", OnlineSecurity="No",
                OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                StreamingTV="Yes", StreamingMovies="Yes", Contract="Month-to-month",
                PaperlessBilling="Yes", PaymentMethod="Electronic check",
                MonthlyCharges=85.5, TotalCharges=171.0)


@pytest.fixture
def client(engine):
    with TestClient(create_app(service_factory=lambda: engine)) as value:
        yield value


def test_exact_model_api_batch_and_repeat_contract(client, engine, payload):
    customer = CustomerPayload(**payload)
    frame = pd.DataFrame([customer.model_dump(exclude={"customer_id"})])
    X = pd.DataFrame(engine.preprocessor.transform(frame), columns=engine.feature_names)
    direct = float(engine.champion_model.predict_proba(X)[0, 1])
    first = client.post("/predict", json=payload).json()
    repeat = client.post("/predict", json=payload).json()
    batch = client.post("/predict-batch", json=[payload, payload]).json()
    assert abs(first["churn_probability"] - direct) <= 1e-12
    assert first == repeat == batch[0] == batch[1]
    assert first["churn_prediction"] == int(direct >= engine.manifest.threshold)
    assert first["risk_level"] == ("REVIEW" if first["churn_prediction"] else "BELOW_THRESHOLD")
    assert first["top_risk_drivers"] == []
    assert first["recommended_retention_action"] is None


def test_model_info_is_manifest_not_hardcoded(client, engine):
    assert client.get("/model-info").json() == engine.manifest.model_dump()
    assert engine.manifest.calibration.status == "not_fitted"
    assert engine.manifest.final_test.metrics is None
    assert engine.manifest.threshold != pytest.approx(0.135039)


@pytest.mark.parametrize("field", [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    *INTERNET_ADDONS, "Contract", "PaperlessBilling", "PaymentMethod",
])
def test_each_unknown_category_is_rejected(client, payload, field):
    response = client.post("/predict", json={**payload, field: "unsupported"})
    assert response.status_code == 422
    assert any(field in item["loc"] for item in response.json()["detail"])


@pytest.mark.parametrize("change", [
    {"tenure": -1}, {"tenure": 121}, {"tenure": 1.5}, {"tenure": True},
    {"MonthlyCharges": -1}, {"MonthlyCharges": 1001}, {"MonthlyCharges": True},
    {"TotalCharges": -1}, {"TotalCharges": 120001}, {"TotalCharges": None},
    {"TotalCharges": " "}, {"SeniorCitizen": 2}, {"SeniorCitizen": -1},
    {"gender": ""}, {"gender": " "}, {"customer_id": "x" * 65},
    {"customer_id": ""}, {"admin": True}, {"PhoneService": "No"},
    {"MultipleLines": "No phone service"}, {"InternetService": "No"},
    {"tenure": 0, "TotalCharges": 1},
    {"tenure": 1, "TotalCharges": 1001},
])
def test_invalid_payloads(client, payload, change):
    assert client.post("/predict", json={**payload, **change}).status_code == 422


@pytest.mark.parametrize("field", INTERNET_ADDONS)
def test_internet_cross_field_rules_both_directions(client, payload, field):
    assert client.post("/predict", json={**payload, field: "No internet service"}).status_code == 422
    absent = {**payload, "InternetService": "No", **{f: "No internet service" for f in INTERNET_ADDONS}}
    absent[field] = "Yes"
    assert client.post("/predict", json=absent).status_code == 422


def test_valid_no_service_and_missing_charges_policy(client, payload):
    valid = {**payload, "InternetService": "No", **{f: "No internet service" for f in INTERNET_ADDONS},
             "PhoneService": "No", "MultipleLines": "No phone service",
             "tenure": 0, "TotalCharges": None}
    assert client.post("/predict", json=valid).status_code == 200
    valid.pop("TotalCharges")
    assert client.post("/predict", json=valid).status_code == 200
    assert client.post("/predict", json={**payload, "Contract": " Month-to-month "}).status_code == 200


def test_numeric_upper_boundaries_and_missing_required(client, payload):
    assert client.post("/predict", json={**payload, "tenure": 120,
                       "MonthlyCharges": 1000, "TotalCharges": 120000}).status_code == 200
    payload.pop("MonthlyCharges")
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["MonthlyCharges", "TotalCharges"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_is_safely_rejected(client, payload, field, value):
    body = json.dumps({**payload, field: value})
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "input" not in response.json()["detail"][0]


@pytest.mark.parametrize("size,code", [(0, 422), (100, 200), (101, 422)])
def test_batch_limits(client, payload, size, code):
    assert client.post("/predict-batch", json=[payload] * size).status_code == code


def test_identifiers_are_optional_unique_and_not_features(client, payload):
    with_id = client.post("/predict", json={**payload, "customer_id": "<img src=x onerror=alert(1)>"}).json()
    without_id = client.post("/predict", json=payload).json()
    assert with_id["churn_probability"] == without_id["churn_probability"]
    response = client.post("/predict-batch", json=[{**payload, "customer_id": "same"}] * 2)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", 1, "customer_id"]


def test_body_size_limit_with_and_without_content_length(engine):
    app = create_app(SecuritySettings(max_body_bytes=1024), service_factory=lambda: engine)
    with TestClient(app) as client:
        assert client.post("/predict", content=b"x" * 1025).status_code == 413
        response = client.post("/predict", content=iter([b"x" * 600, b"x" * 600]))
        assert response.status_code == 413


def test_rate_limit_is_real_and_does_not_limit_health(engine, payload):
    app = create_app(SecuritySettings(requests_per_minute=2), service_factory=lambda: engine)
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 200
        assert client.post("/predict", json=payload).status_code == 200
        assert client.post("/predict", json=payload).status_code == 429
        assert client.get("/health").status_code == 200


def test_api_key_authentication_is_fail_closed_and_redacted(engine, payload, caplog):
    key = "release-test-key-0123456789abcdef"
    settings = SecuritySettings(require_api_key=True, api_keys=(key,), environment="production")
    with TestClient(create_app(settings, service_factory=lambda: engine)) as client:
        assert client.get("/health").status_code == 200
        missing = client.post("/predict", json=payload)
        wrong = client.post("/predict", json=payload, headers={"X-API-Key": "x" * 32})
        valid = client.post("/predict", json=payload, headers={"X-API-Key": key})
        assert missing.status_code == wrong.status_code == 401
        assert missing.headers["www-authenticate"] == "ApiKey"
        assert valid.status_code == 200
        assert client.get("/model-info").status_code == 401
        assert client.get("/metrics").status_code == 401
        assert client.get("/metrics", headers={"X-API-Key": key}).status_code == 200
    assert key not in caplog.text


def test_production_settings_require_strong_api_key(monkeypatch):
    with pytest.raises(ValueError, match="Production"):
        SecuritySettings(environment=" prod ")
    direct = SecuritySettings(
        environment=" Production ",
        require_api_key=True,
        api_keys=("direct-production-key-0123456789abc",),
    )
    assert direct.environment == "production"
    monkeypatch.setenv("CUSTOMERIQ_ENVIRONMENT", "production")
    monkeypatch.delenv("CUSTOMERIQ_API_KEYS", raising=False)
    monkeypatch.delenv("CUSTOMERIQ_REQUIRE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        SecuritySettings.from_env()
    monkeypatch.setenv("CUSTOMERIQ_API_KEYS", "short")
    with pytest.raises(ValueError, match="32 to 256"):
        SecuritySettings.from_env()


def test_metrics_are_bounded_and_contain_no_payload_data(engine, payload):
    app = create_app(service_factory=lambda: engine)
    secret_identifier = "SECRET-CUSTOMER-ID"
    with TestClient(app) as client:
        assert client.post("/predict", json={**payload, "customer_id": secret_identifier}).status_code == 200
        assert client.post("/predict", json={**payload, "gender": "attacker-value"}).status_code == 422
        assert client.get("/attacker-controlled-path").status_code == 404
        body = client.get("/metrics").text
    assert "customeriq_churn_score_count 1" in body
    assert 'endpoint="/predict"' in body
    assert 'route="unmatched"' in body
    assert secret_identifier not in body
    assert "attacker-value" not in body
    assert "/attacker-controlled-path" not in body


def test_cors_allowlist_and_security_headers(engine):
    app = create_app(SecuritySettings(allowed_origins=("https://example.org",)), service_factory=lambda: engine)
    with TestClient(app) as client:
        allowed = client.get("/health", headers={"Origin": "https://example.org"})
        denied = client.get("/health", headers={"Origin": "https://evil.invalid"})
        assert allowed.headers["access-control-allow-origin"] == "https://example.org"
        assert "access-control-allow-origin" not in denied.headers
        assert "frame-ancestors 'none'" in denied.headers["content-security-policy"]
        assert denied.headers["x-content-type-options"] == "nosniff"
        assert "x-request-id" in denied.headers


def test_generic_500_and_redacted_structured_log(client, engine, payload, caplog):
    with patch.object(engine, "predict_customers", side_effect=RuntimeError("SECRET CUSTOMER PAYLOAD")):
        response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert "SECRET" not in response.text
    assert "SECRET" not in caplog.text
    assert "RuntimeError" in caplog.text
    assert '"traceback"' in caplog.text
    assert response.headers["x-request-id"] in caplog.text


def test_corrupt_artifact_rejected_before_deserialization(engine, tmp_path):
    (tmp_path / "champion_model.joblib").write_bytes(b"corrupt model")
    with patch("src.api.manifest.joblib.load") as loader:
        with pytest.raises(ValueError, match="integrity"):
            load_verified_artifact(tmp_path, "champion_model.joblib", engine.manifest)
        loader.assert_not_called()


@pytest.mark.parametrize("mutation", [
    lambda manifest: manifest["calibration"].update(status="fitted"),
    lambda manifest: manifest["final_test"].update(metrics={"roc_auc": 0.99}),
    lambda manifest: manifest.update(threshold=0.13503908770159245, threshold_status="approved"),
    lambda manifest: manifest["development_evaluation"].update(threshold_deployable=True),
    lambda manifest: manifest.update(model_id="unverified-model"),
    lambda manifest: manifest["package_versions"].pop("numpy"),
    lambda manifest: manifest["cv_metrics"][0].update(average_precision_mean=float("nan")),
    lambda manifest: manifest["global_importance"][0].update(importance=-0.1),
    lambda manifest: manifest["business_costs"].update(false_negative=500),
])
def test_manifest_rejects_unauthorized_release_claims(engine, tmp_path, mutation):
    manifest = engine.manifest.model_dump(mode="json")
    mutation(manifest)
    path = tmp_path / "manifest.v1.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        read_manifest(path)


def test_no_manifest_no_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelService(tmp_path)


def test_request_schema_matches_manifest(engine):
    assert engine.manifest.feature_schema["request"] == CustomerPayload.model_json_schema()


def test_release_runtime_contract_matches_manifest(engine):
    root = Path(__file__).resolve().parents[1]
    requirement_lines = {
        line.strip() for line in (root / "requirements-api.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    package_names = {"uvicorn": "uvicorn[standard]"}
    for package, version in engine.manifest.package_versions.items():
        if package == "python":
            continue
        requirement = f"{package_names.get(package, package)}=={version}"
        assert requirement in requirement_lines

    python_version = engine.manifest.package_versions["python"]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert f"FROM python:{python_version}-slim-bookworm" in dockerfile
    assert "requirements-api.lock" in dockerfile
    assert "--require-hashes -r requirements-api.lock" in dockerfile
    assert "COPY models/ /app/models/" not in dockerfile
    assert "COPY models/manifest.v1.json /app/models/manifest.v1.json" in dockerfile
    assert "COPY models/preprocessor.joblib /app/models/preprocessor.joblib" in dockerfile
    assert "COPY models/champion_model.joblib /app/models/champion_model.joblib" in dockerfile
    assert "COPY src/api/ /app/src/api/" in dockerfile
    assert "USER customeriq" in dockerfile


def test_legacy_explanation_fails_explicitly():
    from src.evaluation.explainability import explain_single_prediction
    with pytest.raises(NotImplementedError, match="unavailable"):
        explain_single_prediction(None, None, [])


def test_live_routes_do_not_expose_legacy_html(client):
    assert client.get("/assets/../static/index.html").status_code == 404


def test_browser_uses_exact_api_score_and_safe_dom(client):
    page = client.get("/")
    script = client.get("/assets/app.js")
    assert page.status_code == 200
    assert script.status_code == 200
    assert "Scores are not calibrated probabilities" in page.text
    assert "NPS" not in page.text and "Technical Tickets" not in page.text
    assert "String(probability)" in script.text
    assert "result.churn_probability" in script.text
    assert "innerHTML" not in script.text
    assert "npsAdjustment" not in script.text
    assert "ticketAdjustment" not in script.text
    assert "top_risk_drivers" not in script.text


def test_docs_redirects_to_generated_openapi(client):
    response = client.get("/docs", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/openapi.json"
