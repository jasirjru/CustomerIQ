"""Distribution-drift diagnostics for approved reference and monitoring samples."""

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NumericDriftReport:
    reference_count: int
    current_count: int
    population_stability_index: float
    severity: str
    bin_edges: tuple[float, ...]
    reference_proportions: tuple[float, ...]
    current_proportions: tuple[float, ...]


@dataclass(frozen=True)
class CategoricalDriftReport:
    reference_count: int
    current_count: int
    total_variation_distance: float
    unseen_current_rate: float
    severity: str
    categories: tuple[str, ...]


def _severity(value: float, warning: float, critical: float) -> str:
    if value >= critical:
        return "critical"
    if value >= warning:
        return "warning"
    return "stable"


def numeric_population_stability(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.25,
) -> NumericDriftReport:
    """Compute PSI using quantile bins learned only from the reference sample."""
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    if reference_values.ndim != 1 or current_values.ndim != 1:
        raise ValueError("Reference and current samples must be 1D.")
    if len(reference_values) < 20 or len(current_values) < 20:
        raise ValueError("Each drift sample needs at least 20 observations.")
    if not np.isfinite(reference_values).all() or not np.isfinite(current_values).all():
        raise ValueError("Drift samples must contain only finite values.")
    if not isinstance(bins, int) or isinstance(bins, bool) or not 2 <= bins <= 50:
        raise ValueError("bins must be an integer between 2 and 50.")
    if not 0 < warning_threshold < critical_threshold:
        raise ValueError("Drift thresholds must be positive and ordered.")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if len(edges) < 3:
        raise ValueError("Reference sample has insufficient numeric variation.")
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    epsilon = 1e-6
    reference_rates = np.maximum(reference_counts / len(reference_values), epsilon)
    current_rates = np.maximum(current_counts / len(current_values), epsilon)
    psi = float(np.sum((current_rates - reference_rates) * np.log(current_rates / reference_rates)))

    return NumericDriftReport(
        reference_count=len(reference_values),
        current_count=len(current_values),
        population_stability_index=psi,
        severity=_severity(psi, warning_threshold, critical_threshold),
        bin_edges=tuple(float(value) for value in edges),
        reference_proportions=tuple(float(value) for value in reference_rates),
        current_proportions=tuple(float(value) for value in current_rates),
    )


def categorical_distribution_drift(
    reference: Sequence[Hashable],
    current: Sequence[Hashable],
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.2,
) -> CategoricalDriftReport:
    """Measure categorical drift without silently merging unseen categories."""
    reference_values = np.asarray(reference, dtype=object)
    current_values = np.asarray(current, dtype=object)
    if reference_values.ndim != 1 or current_values.ndim != 1:
        raise ValueError("Reference and current samples must be 1D.")
    if len(reference_values) < 20 or len(current_values) < 20:
        raise ValueError("Each drift sample needs at least 20 observations.")
    if pd.isna(reference_values).any() or pd.isna(current_values).any():
        raise ValueError("Categorical drift samples must not contain missing values.")
    if not 0 < warning_threshold < critical_threshold <= 1:
        raise ValueError("Drift thresholds must be in (0, 1] and ordered.")

    reference_categories = set(reference_values.tolist())
    current_categories = set(current_values.tolist())
    categories = sorted(reference_categories | current_categories, key=str)
    reference_rates = np.array([(reference_values == item).mean() for item in categories])
    current_rates = np.array([(current_values == item).mean() for item in categories])
    distance = float(0.5 * np.abs(reference_rates - current_rates).sum())
    unseen_rate = float(np.isin(current_values, list(current_categories - reference_categories)).mean())
    signal = max(distance, unseen_rate)
    return CategoricalDriftReport(
        reference_count=len(reference_values),
        current_count=len(current_values),
        total_variation_distance=distance,
        unseen_current_rate=unseen_rate,
        severity=_severity(signal, warning_threshold, critical_threshold),
        categories=tuple(str(item) for item in categories),
    )
