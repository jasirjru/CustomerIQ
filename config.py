"""
CustomerIQ — Central Configuration

All project-wide constants and paths are defined here.
This avoids hard-coded paths scattered across notebooks and source files.

Why this matters:
- If the project structure changes, you update ONE file.
- Reproducibility: fixed random seed ensures consistent results.
- Clean code: no magic numbers or strings buried in logic.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory (where this file lives)
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"

# Reports and figures
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Raw dataset filename
RAW_DATASET_FILENAME = "telco_customer_churn.csv"
RAW_DATASET_PATH = DATA_RAW / RAW_DATASET_FILENAME

# ============================================================================
# ML CONFIGURATION
# ============================================================================

# Random seed for reproducibility
# Why 42? Convention. Any fixed integer works.
# The important thing is to use the SAME seed everywhere.
RANDOM_STATE = 42

# Target column
TARGET_COLUMN = "Churn"

# Columns to drop before training
# customerID: unique identifier, no predictive value
COLUMNS_TO_DROP = ["customerID"]

# Test set size (20% held out for evaluation)
TEST_SIZE = 0.2

# Evaluation protocol
#
# The final test partition is reserved for a single evaluation after model,
# hyperparameter, preprocessing, and decision-threshold choices are frozen.
# The remaining 80% development partition is split 75/25, which yields an
# overall 60/20/20 train/validation/final-test allocation.
FINAL_TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.25

# Repeated cross-validation is performed on the model-training partition only.
CV_N_SPLITS = 5
CV_N_REPEATS = 3
MODEL_SELECTION_METRIC = "average_precision"

# Illustrative validation-time decision costs. These values are not considered
# production-approved until their business provenance is documented.
FALSE_NEGATIVE_COST = 500.0
FALSE_POSITIVE_COST = 100.0

# ============================================================================
# FEATURE GROUPS (will be refined during EDA)
# ============================================================================

# These are initial groupings based on dataset documentation.
# We will verify and potentially update these during Phase 1 (EDA).

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
