"""
CustomerIQ — FastAPI Model Serving Application

Exposes production REST endpoints for customer churn prediction,
unsupervised segmentation, and explainable retention playbooks.
"""

from typing import List
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.api.schemas import (
    CustomerPayload,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from src.api.service import ModelService

# Global service instance
model_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load ML models on startup and clean up on shutdown."""
    global model_service
    print("[INFO] Initializing CustomerIQ Model Service...")
    model_service = ModelService(threshold=0.35)
    print("[SUCCESS] Model Service successfully initialized with Preprocessor and Champion Model.")
    yield
    print("[INFO] Shutting down CustomerIQ API.")


app = FastAPI(
    title="CustomerIQ — Churn Intelligence API",
    description="Production Machine Learning API for real-time customer churn prediction and segmentation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def root():
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint for Kubernetes liveness/readiness probes and monitoring.
    """
    if model_service is None or model_service.champion_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service is not fully initialized.",
        )
    return HealthResponse(
        status="healthy",
        service="CustomerIQ-Inference-Engine",
        model_loaded=model_service.champion_model is not None,
        preprocessor_loaded=model_service.preprocessor is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """
    Return operational metadata about the currently deployed champion model.
    """
    if model_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized.")

    return ModelInfoResponse(
        model_name="Random Forest Classifier (Champion)",
        model_type="Ensemble / Bagging",
        decision_threshold=model_service.threshold,
        test_roc_auc=0.8429,
        test_accuracy=0.8070,
        engineered_features_count=len(model_service.feature_names),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_churn(customer: CustomerPayload):
    """
    Predict churn probability and customer segment for an individual customer.
    Applies real-time preprocessing, thresholding, and returns local explainability drivers.
    """
    try:
        response = model_service.predict_customer(customer)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post("/predict-batch", response_model=List[PredictionResponse], tags=["Inference"])
async def predict_batch(customers: List[CustomerPayload]):
    """
    Batch scoring endpoint for processing multiple customer profiles in a single request.
    """
    try:
        return [model_service.predict_customer(c) for c in customers]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )
