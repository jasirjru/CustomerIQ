"""One inference service shared by single and batch routes."""

from pathlib import Path
import numpy as np
import pandas as pd

from config import MODELS_DIR
from src.api.manifest import load_verified_artifact, read_manifest
from src.api.schemas import CustomerPayload, PredictionResponse


class ModelService:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.manifest = read_manifest(models_dir / "manifest.v1.json")
        self.preprocessor = load_verified_artifact(models_dir, "preprocessor.joblib", self.manifest)
        self.champion_model = load_verified_artifact(models_dir, "champion_model.joblib", self.manifest)
        if type(self.preprocessor).__name__ != self.manifest.preprocessor["class"]:
            raise ValueError("Preprocessor does not match manifest.")
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        if self.feature_names != self.manifest.feature_schema["transformed_features"]:
            raise ValueError("Preprocessor feature schema does not match manifest.")
        if list(self.champion_model.feature_names_in_) != self.feature_names:
            raise ValueError("Classifier feature schema does not match preprocessor.")
        if type(self.champion_model).__name__ != self.manifest.model["class"]:
            raise ValueError("Classifier does not match manifest.")
        actual_parameters = self.champion_model.get_params(deep=False)
        if any(actual_parameters.get(name) != value
               for name, value in self.manifest.model["hyperparameters"].items()):
            raise ValueError("Classifier hyperparameters do not match manifest.")
        recorded_importance = {item["feature"]: item["importance"]
                               for item in self.manifest.global_importance}
        artifact_importance = dict(zip(self.feature_names, self.champion_model.feature_importances_))
        if recorded_importance.keys() != artifact_importance.keys() or any(
                not np.isclose(recorded_importance[name], value, rtol=0, atol=1e-15)
                for name, value in artifact_importance.items()):
            raise ValueError("Global importance does not match the classifier artifact.")
        if self.manifest.feature_schema["request"] != CustomerPayload.model_json_schema():
            raise ValueError("Request schema does not match manifest.")
        self.threshold = self.manifest.threshold
        classes = list(self.champion_model.classes_)
        if classes != [0, 1]:
            raise ValueError("Expected binary class order [0, 1].")
        # Runtime thread policy, not a change to learned trees or artifacts.
        self.champion_model.n_jobs = 1

    def predict_customers(self, customers: list[CustomerPayload]) -> list[PredictionResponse]:
        frame = pd.DataFrame([c.model_dump(exclude={"customer_id"}) for c in customers])
        transformed = pd.DataFrame(self.preprocessor.transform(frame), columns=self.feature_names)
        probabilities = self.champion_model.predict_proba(transformed)[:, 1]
        if not np.isfinite(probabilities).all():
            raise ValueError("Estimator returned a non-finite probability.")
        result = []
        for customer, probability in zip(customers, probabilities):
            decision = int(probability >= self.threshold)
            result.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_probability=float(probability),
                churn_prediction=decision,
                risk_level="REVIEW" if decision else "BELOW_THRESHOLD",
                decision_threshold=self.threshold,
                threshold_status=self.manifest.threshold_status,
                model_version=self.manifest.model_id,
                calibration_status=self.manifest.calibration.status,
            ))
        return result

    def predict_customer(self, payload: CustomerPayload) -> PredictionResponse:
        return self.predict_customers([payload])[0]
