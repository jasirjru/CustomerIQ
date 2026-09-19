"""Manifest validation and hash verification before trusted model deserialization."""

import hashlib
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CalibrationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["not_fitted"]
    method: None
    validation_metrics: None


class FinalTestMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["not_evaluated_in_v1_1"]
    rows_reserved: int = Field(gt=0)
    metrics: None
    historical_exposure: str = Field(min_length=1)


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["1.1"]
    model_id: str
    created_at: str
    artifact_created_at: str | None
    git_commit: str | None
    dataset_hash: str | None
    development_data_hashes: dict[str, str]
    feature_schema: dict
    preprocessor: dict
    model: dict
    calibration: CalibrationMetadata
    cv_metrics: list[dict]
    development_evaluation: dict
    threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    threshold_status: str
    threshold_selection: str
    business_costs: dict
    package_versions: dict[str, str]
    package_versions_provenance: str
    artifact_hashes: dict[str, str]
    final_test: FinalTestMetadata
    global_importance: list[dict]
    limitations: list[str]

    @field_validator("created_at", "artifact_created_at")
    @classmethod
    def validate_timestamps(cls, value):
        if value is None:
            return value
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Manifest timestamps must use ISO 8601.") from exc
        if parsed.utcoffset() is None:
            raise ValueError("Manifest timestamps must include a timezone offset.")
        return value

    @field_validator("git_commit")
    @classmethod
    def validate_git_commit(cls, value):
        if value is not None and re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("git_commit must be a full lowercase Git SHA.")
        return value

    @field_validator("development_data_hashes")
    @classmethod
    def validate_development_hashes(cls, value):
        if set(value) != {"X_train_raw.csv", "y_train.csv"}:
            raise ValueError("Development hashes must identify the two sealed development inputs.")
        cls._validate_sha_values(value.values())
        return value

    @field_validator("feature_schema")
    @classmethod
    def validate_feature_schema(cls, value):
        expected = {"request", "raw_features", "transformed_features", "categorical_vocabularies"}
        if set(value) != expected:
            raise ValueError("Feature schema has an invalid shape.")
        raw = value["raw_features"]
        transformed = value["transformed_features"]
        if not isinstance(raw, list) or not raw or len(raw) != len(set(raw)):
            raise ValueError("Raw feature names must be a non-empty unique list.")
        if not isinstance(transformed, list) or not transformed or len(transformed) != len(set(transformed)):
            raise ValueError("Transformed feature names must be a non-empty unique list.")
        if value["request"].get("additionalProperties") is not False:
            raise ValueError("The recorded request schema must reject unknown fields.")
        if not isinstance(value["categorical_vocabularies"], dict):
            raise ValueError("Categorical vocabularies must be an object.")
        return value

    @field_validator("preprocessor")
    @classmethod
    def validate_preprocessor_metadata(cls, value):
        if set(value) != {"class", "training_package_version"}:
            raise ValueError("Preprocessor metadata has an invalid shape.")
        if value["class"] != "ColumnTransformer" or value["training_package_version"] is not None:
            raise ValueError("Legacy preprocessor provenance must remain explicit and unclaimed.")
        return value

    @field_validator("model")
    @classmethod
    def validate_model_metadata(cls, value):
        expected = {"class", "hyperparameters", "training_rows", "training_provenance"}
        if set(value) != expected or value["class"] != "RandomForestClassifier":
            raise ValueError("Model metadata has an invalid shape or class.")
        if not isinstance(value["hyperparameters"], dict) or not value["hyperparameters"]:
            raise ValueError("Model hyperparameters must be recorded.")
        if not isinstance(value["training_rows"], int) or value["training_rows"] <= 0:
            raise ValueError("Model training_rows must be a positive integer.")
        if not isinstance(value["training_provenance"], str) or not value["training_provenance"].strip():
            raise ValueError("Model training provenance must be explicit.")
        return value

    @field_validator("cv_metrics")
    @classmethod
    def validate_cv_metrics(cls, value):
        expected = {
            "model", "folds", "average_precision_mean", "average_precision_std",
            "roc_auc_mean", "f1_mean", "balanced_accuracy_mean",
        }
        names = []
        if not value:
            raise ValueError("Development CV metrics must not be empty.")
        for row in value:
            if set(row) != expected or not isinstance(row["model"], str):
                raise ValueError("A CV metric row has an invalid shape.")
            if not isinstance(row["folds"], int) or row["folds"] < 2:
                raise ValueError("CV fold counts must be integers of at least 2.")
            numbers = [row[key] for key in expected - {"model", "folds"}]
            if any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in numbers):
                raise ValueError("CV metrics must be finite numbers.")
            if not 0 <= row["average_precision_std"] <= 1 or any(
                not 0 <= row[key] <= 1
                for key in ("average_precision_mean", "roc_auc_mean", "f1_mean", "balanced_accuracy_mean")
            ):
                raise ValueError("CV metrics must be in [0, 1].")
            names.append(row["model"])
        if len(names) != len(set(names)):
            raise ValueError("CV model names must be unique.")
        return value

    @field_validator("development_evaluation")
    @classmethod
    def validate_development_evaluation(cls, value):
        expected = {
            "status", "protocol", "primary_metric", "selected_candidate", "ap_difference",
            "experimental_threshold", "threshold_deployable", "validation_rows",
            "validation_cost", "illustrative_costs", "artifact_replaced",
        }
        if set(value) != expected or value["status"] != "separate_candidate_experiment":
            raise ValueError("Development evaluation metadata has an invalid shape.")
        if value["primary_metric"] != "average_precision":
            raise ValueError("Development primary metric must remain explicit.")
        for key in ("protocol", "selected_candidate"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"Development evaluation {key} is required.")
        if not isinstance(value["validation_rows"], int) or value["validation_rows"] <= 0:
            raise ValueError("validation_rows must be a positive integer.")
        for key in ("ap_difference", "experimental_threshold", "validation_cost"):
            if not isinstance(value[key], (int, float)) or not math.isfinite(value[key]):
                raise ValueError(f"{key} must be finite.")
        if not 0 <= value["experimental_threshold"] <= 1 or value["validation_cost"] < 0:
            raise ValueError("Development threshold/cost values are out of range.")
        costs = value["illustrative_costs"]
        if set(costs) != {"false_negative", "false_positive"} or any(
            not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0
            for item in costs.values()
        ):
            raise ValueError("Illustrative costs must be finite, non-negative values.")
        return value

    @field_validator("business_costs")
    @classmethod
    def validate_business_costs(cls, value):
        if set(value) != {"status", "false_negative", "false_positive", "approval_record"}:
            raise ValueError("Business cost metadata has an invalid shape.")
        if value != {
            "status": "not_approved",
            "false_negative": None,
            "false_positive": None,
            "approval_record": "docs/business-costs.json",
        }:
            raise ValueError("Unapproved costs must not contain deployment assumptions.")
        return value

    @field_validator("package_versions")
    @classmethod
    def validate_package_versions(cls, value):
        expected = {
            "python", "numpy", "pandas", "scikit-learn", "joblib",
            "fastapi", "pydantic", "uvicorn", "starlette",
        }
        if set(value) != expected or any(
            re.fullmatch(r"\d+\.\d+\.\d+(?:[a-zA-Z0-9.+-]*)", version) is None
            for version in value.values()
        ):
            raise ValueError("Runtime package version metadata is incomplete or invalid.")
        return value

    @field_validator("artifact_hashes")
    @classmethod
    def validate_artifact_hashes(cls, value):
        expected = {"preprocessor.joblib", "champion_model.joblib"}
        if set(value) != expected:
            raise ValueError("Manifest must register exactly the serving artifacts.")
        cls._validate_sha_values(value.values())
        return value

    @staticmethod
    def _validate_sha_values(values):
        if any(len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) for digest in values):
            raise ValueError("Hashes must be lowercase SHA-256 digests.")

    @field_validator("global_importance")
    @classmethod
    def validate_global_importance(cls, value):
        for item in value:
            if set(item) != {"feature", "importance"} or not isinstance(item["feature"], str):
                raise ValueError("Global importance entries have an invalid shape.")
            if not isinstance(item["importance"], (int, float)) or not math.isfinite(item["importance"]):
                raise ValueError("Global importance values must be finite numbers.")
            if item["importance"] < 0:
                raise ValueError("Global importance values must be non-negative.")
        if not value or not math.isclose(sum(item["importance"] for item in value), 1.0, abs_tol=1e-9):
            raise ValueError("Global importance values must form a complete normalized vector.")
        return value

    @field_validator("limitations")
    @classmethod
    def validate_limitations(cls, value):
        if not value or len(value) != len(set(value)) or any(not item.strip() for item in value):
            raise ValueError("Manifest limitations must be non-empty and unique.")
        return value

    @model_validator(mode="after")
    def validate_release_policy(self):
        if self.threshold_status != "legacy_unapproved" or self.threshold != 0.35:
            raise ValueError("This release retains the legacy threshold; promotion is not authorized.")
        if self.business_costs.get("status") != "not_approved":
            raise ValueError("Serving costs have not been approved.")
        if self.development_evaluation.get("threshold_deployable") is not False:
            raise ValueError("The development threshold must remain non-deployable.")
        if self.development_evaluation.get("artifact_replaced") is not False:
            raise ValueError("The serving artifact must remain unchanged in this release.")
        candidate_names = {row["model"] for row in self.cv_metrics}
        if self.development_evaluation.get("selected_candidate") not in candidate_names:
            raise ValueError("Selected development candidate is absent from CV evidence.")
        expected_model_id = f"legacy-rf-{self.artifact_hashes['champion_model.joblib'][:12]}"
        if self.model_id != expected_model_id:
            raise ValueError("Model ID does not match the registered classifier artifact.")
        transformed = self.feature_schema.get("transformed_features", [])
        importance_features = [item["feature"] for item in self.global_importance]
        if len(set(importance_features)) != len(importance_features) or set(importance_features) != set(transformed):
            raise ValueError("Global importance features do not match the transformed schema.")
        return self


def read_manifest(path: Path) -> ArtifactManifest:
    manifest = ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8"))
    return manifest


def load_verified_artifact(directory: Path, name: str, manifest: ArtifactManifest):
    if Path(name).name != name or name not in manifest.artifact_hashes:
        raise ValueError("Artifact is not registered in the manifest.")
    content = (directory / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.artifact_hashes[name]:
        raise ValueError("Artifact integrity verification failed.")
    # Only reviewed repository artifacts are trusted. Hashes do not make an
    # attacker-controlled pickle safe; protect manifest and artifacts together.
    return joblib.load(io.BytesIO(content))
