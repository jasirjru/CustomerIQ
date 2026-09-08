"""
CustomerIQ — Model Serving Service

Handles pipeline loading, input transformation, inference, and explainability.
Guarantees zero code duplication by using the exact preprocessor and model saved during training.
"""

from typing import Dict, Any, Tuple, List
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import MODELS_DIR
from src.api.schemas import CustomerPayload, PredictionResponse


class ModelService:
    """
    Singleton-style service managing machine learning artifacts in memory.
    """

    def __init__(self, models_dir: Path = MODELS_DIR, threshold: float = 0.35):
        self.models_dir = models_dir
        self.threshold = threshold
        self.preprocessor = None
        self.champion_model = None
        self.cluster_model = None
        self.feature_names = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load fitted pipeline, champion classifier, and clustering models."""
        preprocessor_path = self.models_dir / "preprocessor.joblib"
        champion_path = self.models_dir / "champion_model.joblib"
        cluster_path = self.models_dir / "kmeans_customer_segmentation.joblib"

        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
        if not champion_path.exists():
            raise FileNotFoundError(f"Champion model not found at {champion_path}")

        self.preprocessor = joblib.load(preprocessor_path)
        self.champion_model = joblib.load(champion_path)

        if cluster_path.exists():
            self.cluster_model = joblib.load(cluster_path)

        # Cache engineered output feature names
        try:
            self.feature_names = list(self.preprocessor.get_feature_names_out())
        except Exception:
            self.feature_names = [f"f_{i}" for i in range(46)]

    def predict_customer(self, payload: CustomerPayload) -> PredictionResponse:
        """
        Execute full end-to-end inference on a single customer payload.

        Steps:
        1. Parse Pydantic model into DataFrame.
        2. Clean and impute missing TotalCharges if needed.
        3. Transform via preprocessor.joblib (StandardScaler + OneHotEncoder).
        4. Predict probability using champion_model.joblib.
        5. Assign customer segment using kmeans_customer_segmentation.joblib.
        6. Compute local risk drivers.
        7. Generate actionable retention playbook.
        """
        data_dict = payload.model_dump()

        # Handle TotalCharges if missing
        if data_dict["TotalCharges"] is None:
            if data_dict["tenure"] == 0:
                data_dict["TotalCharges"] = 0.0
            else:
                data_dict["TotalCharges"] = round(data_dict["tenure"] * data_dict["MonthlyCharges"], 2)

        df_single = pd.DataFrame([data_dict])

        # Step 3: Transform using fitted pipeline
        X_trans_arr = self.preprocessor.transform(df_single)
        X_trans = pd.DataFrame(X_trans_arr, columns=self.feature_names)

        # Step 4: Predict probabilities
        churn_proba = float(self.champion_model.predict_proba(X_trans)[0, 1])
        churn_pred = 1 if churn_proba >= self.threshold else 0

        # Risk Category
        if churn_proba >= 0.50:
            risk_level = "HIGH"
        elif churn_proba >= self.threshold:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Step 5: Unsupervised Customer Segment
        segment_name = "Standard Subscriber"
        if self.cluster_model is not None:
            cluster_id = int(self.cluster_model.predict(X_trans_arr)[0])
            segment_map = {
                0: "Budget Phone / Basic Loyalist (Low Churn)",
                1: "Mid-Tier DSL Subscriber (Moderate Churn)",
                2: "High-Value Multi-Service Loyalist (Core Revenue)",
                3: "New High-Spend Flight Risk (Severe Churn - Urgent)",
            }
            segment_name = segment_map.get(cluster_id, f"Cluster {cluster_id}")

        # Step 6: Top local risk factors
        importances = self.champion_model.feature_importances_
        contributions = X_trans_arr[0] * importances
        top_indices = np.argsort(contributions)[::-1][:3]

        top_drivers = []
        for idx in top_indices:
            if contributions[idx] > 0:
                feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                top_drivers.append({
                    "feature": feat_name,
                    "impact_score": round(float(contributions[idx]), 4),
                })

        # Step 7: Prescriptive business recommendation
        action = self._determine_retention_action(payload, risk_level, churn_proba)

        return PredictionResponse(
            churn_prediction=churn_pred,
            churn_probability=round(churn_proba, 4),
            risk_level=risk_level,
            decision_threshold=self.threshold,
            customer_segment=segment_name,
            top_risk_drivers=top_drivers,
            recommended_retention_action=action,
        )

    def _determine_retention_action(self, payload: CustomerPayload, risk_level: str, proba: float) -> str:
        """Prescribe targeted retention strategy based on customer attributes."""
        if risk_level == "LOW":
            return "Account healthy. Enroll in loyalty rewards program to encourage referrals."

        reasons = []
        if payload.Contract == "Month-to-month":
            reasons.append("Offer 15% discount for upgrading to a 1-year annual contract.")
        if payload.InternetService == "Fiber optic" and payload.TechSupport == "No":
            reasons.append("Provide 3 months complimentary VIP Tech Support to address technical frustrations.")
        if payload.PaymentMethod == "Electronic check":
            reasons.append("Offer a one-time $10 credit to enroll in automated credit card auto-pay.")

        if reasons:
            return " | ".join(reasons)
        return "High flight risk detected. Contact via customer success team within 24 hours."
