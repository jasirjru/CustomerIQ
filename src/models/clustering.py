"""
CustomerIQ — Customer Segmentation & Unsupervised Clustering

Provides reusable K-Means clustering, WCSS/Inertia calculations,
Silhouette analysis, and cluster profiling utilities.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import RANDOM_STATE


def evaluate_kmeans_clusters(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = RANDOM_STATE,
) -> Tuple[List[float], List[float]]:
    """
    Compute WCSS (Inertia) and Silhouette Scores across a range of candidate cluster counts k.

    Args:
        X: Preprocessed numerical feature matrix.
        k_range: Range of k values to evaluate (e.g. 2 to 10).
        random_state: Random seed for deterministic centroid initialization.

    Returns:
        (wcss_scores, silhouette_scores)
    """
    wcss_scores = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=random_state,
        )
        labels = kmeans.fit_predict(X)
        wcss_scores.append(kmeans.inertia_)
        sil_score = silhouette_score(X, labels)
        silhouette_scores.append(round(sil_score, 4))

    return wcss_scores, silhouette_scores


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_STATE,
) -> Tuple[KMeans, np.ndarray]:
    """
    Fit final K-Means model with chosen k.

    Args:
        X: Preprocessed feature matrix.
        n_clusters: Selected optimal k.
        random_state: Random seed.

    Returns:
        (fitted_kmeans, cluster_labels)
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def profile_clusters(
    raw_df: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    Profile customer clusters using raw, interpretable business attributes.

    Args:
        raw_df: Cleaned dataframe with human-readable original columns
                (tenure, MonthlyCharges, Contract, Churn, etc.).
        cluster_labels: Integer cluster assignment for each row.

    Returns:
        pd.DataFrame: Profile summary grouped by cluster.
    """
    df_profile = raw_df.copy()
    df_profile["Cluster"] = cluster_labels

    # Compute key business statistics per cluster
    summary = df_profile.groupby("Cluster").agg(
        Count=("tenure", "count"),
        Pct_of_Total=("tenure", lambda x: round(len(x) / len(df_profile) * 100, 1)),
        Avg_Tenure_Months=("tenure", "mean"),
        Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        Avg_Total_Charges=("TotalCharges", "mean"),
        Churn_Rate=("Churn", lambda x: round((x == 1).mean() * 100, 1) if (x == 1).sum() > 0 else round((x == "Yes").mean() * 100, 1)),
    ).round(2)

    return summary
