"""Descriptive cohort diagnostics for binary churn decisions.

These metrics expose differences for human review; they do not establish legal
fairness, causality, or the absence of discrimination.
"""

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CohortMetrics:
    cohort: str
    count: int
    outcome_rate: float
    mean_probability: float
    review_rate: float
    true_positive_rate: float | None
    false_positive_rate: float | None
    precision: float | None


@dataclass(frozen=True)
class CohortReport:
    threshold: float
    minimum_group_size: int
    excluded_small_cohorts: tuple[str, ...]
    cohorts: tuple[CohortMetrics, ...]
    max_review_rate_gap: float
    max_true_positive_rate_gap: float | None
    max_false_positive_rate_gap: float | None


def _gap(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return float(max(present) - min(present)) if len(present) >= 2 else None


def evaluate_binary_cohorts(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    cohorts: Sequence[Hashable],
    threshold: float,
    minimum_group_size: int = 20,
) -> CohortReport:
    """Compute review and error-rate diagnostics for explicitly supplied cohorts."""
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    groups = np.asarray(cohorts, dtype=object)
    if any(array.ndim != 1 for array in (labels, scores, groups)):
        raise ValueError("Labels, probabilities, and cohorts must be 1D arrays.")
    if len(labels) == 0 or len(labels) != len(scores) or len(labels) != len(groups):
        raise ValueError("Labels, probabilities, and cohorts must be non-empty and aligned.")
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Labels must contain only binary values 0 and 1.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1].")
    if not isinstance(minimum_group_size, int) or isinstance(minimum_group_size, bool) or minimum_group_size < 1:
        raise ValueError("minimum_group_size must be a positive integer.")
    if pd.isna(groups).any():
        raise ValueError("Cohorts must not contain missing values.")

    decisions = scores >= threshold
    rows: list[CohortMetrics] = []
    excluded: list[str] = []
    for value in sorted(set(groups.tolist()), key=str):
        mask = groups == value
        count = int(mask.sum())
        name = str(value)
        if count < minimum_group_size:
            excluded.append(name)
            continue
        group_labels = labels[mask].astype(int)
        group_decisions = decisions[mask]
        positives = group_labels == 1
        negatives = ~positives
        predicted_positive = group_decisions
        rows.append(
            CohortMetrics(
                cohort=name,
                count=count,
                outcome_rate=float(group_labels.mean()),
                mean_probability=float(scores[mask].mean()),
                review_rate=float(group_decisions.mean()),
                true_positive_rate=(
                    float(group_decisions[positives].mean()) if positives.any() else None
                ),
                false_positive_rate=(
                    float(group_decisions[negatives].mean()) if negatives.any() else None
                ),
                precision=(
                    float(group_labels[predicted_positive].mean())
                    if predicted_positive.any()
                    else None
                ),
            )
        )
    if not rows:
        raise ValueError("No cohort meets minimum_group_size.")

    return CohortReport(
        threshold=float(threshold),
        minimum_group_size=minimum_group_size,
        excluded_small_cohorts=tuple(excluded),
        cohorts=tuple(rows),
        max_review_rate_gap=float(_gap([row.review_rate for row in rows]) or 0.0),
        max_true_positive_rate_gap=_gap([row.true_positive_rate for row in rows]),
        max_false_positive_rate_gap=_gap([row.false_positive_rate for row in rows]),
    )
