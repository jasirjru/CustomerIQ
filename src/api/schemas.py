"""Validated prediction inputs and honest, versioned response contracts."""

from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]
INTERNET_ADDONS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies")


class CustomerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    customer_id: str | None = Field(None, max_length=64, description="Optional correlation ID, never a model feature.")
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(ge=0, le=1, strict=True)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, strict=True)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0, le=1000, strict=True)
    TotalCharges: float | None = Field(None, ge=0, le=120000, strict=True, description="Observed spend. Null allowed only when tenure is zero.")

    @field_validator("*", mode="before")
    @classmethod
    def normalize(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value or len(value) > 64:
                raise ValueError("Strings must contain 1 to 64 non-whitespace characters.")
        if isinstance(value, bool):
            raise ValueError("Boolean values are not valid customer fields.")
        return value

    @model_validator(mode="after")
    def services_and_charges(self):
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines must be 'No phone service' exactly when PhoneService is 'No'.")
        for name in INTERNET_ADDONS:
            if (self.InternetService == "No") != (getattr(self, name) == "No internet service"):
                raise ValueError(f"{name} must be 'No internet service' exactly when InternetService is 'No'.")
        if self.tenure == 0:
            if self.TotalCharges not in (None, 0):
                raise ValueError("TotalCharges must be zero for tenure=0.")
            self.TotalCharges = 0.0
        elif self.TotalCharges is None:
            raise ValueError("Observed TotalCharges is required when tenure is positive.")
        elif self.TotalCharges > self.tenure * 1000:
            raise ValueError("TotalCharges exceeds the maximum possible charges for tenure.")
        return self


CustomerBatch = Annotated[list[CustomerPayload], Field(min_length=1, max_length=100)]


class PredictionResponse(BaseModel):
    customer_id: str | None = None
    churn_prediction: Literal[0, 1]
    churn_probability: float = Field(ge=0, le=1, allow_inf_nan=False, description="Unrounded estimated churn probability; no fitted calibrator.")
    risk_level: Literal["REVIEW", "BELOW_THRESHOLD"]
    decision_threshold: float
    threshold_status: str
    model_version: str
    calibration_status: str
    explanation_status: Literal["not_available"] = "not_available"
    # Retained empty/null for clients migrating from v1.0. No invalid attribution.
    top_risk_drivers: list[dict] = Field(default_factory=list)
    recommended_retention_action: str | None = None
    customer_segment: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    preprocessor_loaded: bool
    manifest_available: bool
    schema_version: str
    environment: str
