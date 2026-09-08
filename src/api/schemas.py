"""
CustomerIQ — Pydantic Request & Response Schemas

Defines type validation, field ranges, and API request/response contracts.
Prevents invalid customer data payloads from reaching the model.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CustomerPayload(BaseModel):
    """
    Schema representing a single customer profile for inference.
    Matches the raw incoming customer format before any preprocessing.
    """
    gender: str = Field(..., examples=["Female"], description="Customer biological gender ('Male' or 'Female')")
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0], description="Whether customer is a senior citizen (1 or 0)")
    Partner: str = Field(..., examples=["No"], description="'Yes' or 'No'")
    Dependents: str = Field(..., examples=["No"], description="'Yes' or 'No'")
    tenure: int = Field(..., ge=0, le=120, examples=[2], description="Number of months customer has stayed with the company")
    PhoneService: str = Field(..., examples=["Yes"], description="'Yes' or 'No'")
    MultipleLines: str = Field(..., examples=["No"], description="'Yes', 'No', or 'No phone service'")
    InternetService: str = Field(..., examples=["Fiber optic"], description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field(..., examples=["No"], description="'Yes', 'No', or 'No internet service'")
    OnlineBackup: str = Field(..., examples=["No"], description="'Yes', 'No', or 'No internet service'")
    DeviceProtection: str = Field(..., examples=["No"], description="'Yes', 'No', or 'No internet service'")
    TechSupport: str = Field(..., examples=["No"], description="'Yes', 'No', or 'No internet service'")
    StreamingTV: str = Field(..., examples=["Yes"], description="'Yes', 'No', or 'No internet service'")
    StreamingMovies: str = Field(..., examples=["Yes"], description="'Yes', 'No', or 'No internet service'")
    Contract: str = Field(..., examples=["Month-to-month"], description="'Month-to-month', 'One year', 'Two year'")
    PaperlessBilling: str = Field(..., examples=["Yes"], description="'Yes' or 'No'")
    PaymentMethod: str = Field(..., examples=["Electronic check"], description="'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'")
    MonthlyCharges: float = Field(..., ge=0.0, examples=[85.50], description="Monthly subscription charge in USD")
    TotalCharges: Optional[float] = Field(None, ge=0.0, examples=[171.0], description="Total historical spend. If null, automatically estimated.")

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class RiskFactor(BaseModel):
    feature: str
    impact_weight: float


class PredictionResponse(BaseModel):
    """
    Standardized inference response contract returned to clients.
    """
    churn_prediction: int = Field(..., description="1 = Likely to Churn, 0 = Likely to Stay")
    churn_probability: float = Field(..., description="Calibrated churn probability [0.0, 1.0]")
    risk_level: str = Field(..., description="'HIGH', 'MODERATE', or 'LOW'")
    decision_threshold: float = Field(..., description="Classification threshold applied")
    customer_segment: Optional[str] = Field(None, description="Discovered unsupervised customer persona")
    top_risk_drivers: List[Dict[str, Any]] = Field(default_factory=list, description="Top positive drivers toward churn")
    recommended_retention_action: str = Field(..., description="Tailored business retention guidance")


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    preprocessor_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    decision_threshold: float
    test_roc_auc: float
    test_accuracy: float
    engineered_features_count: int
