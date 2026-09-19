"""Validity gates for candidate local attributions; not a serving explainer."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AdditivityCheck:
    output_space: str
    explained_output: float
    reconstructed_output: float
    absolute_error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class ExplanationStability:
    runs: int
    top_k: int
    mean_pairwise_jaccard: float
    passed: bool


def aggregate_transformed_attributions(
    contributions: Sequence[float],
    transformed_feature_names: Sequence[str],
    raw_feature_names: Sequence[str],
) -> dict[str, float]:
    """Aggregate one-hot/transformed contributions back to raw input features."""
    values = np.asarray(contributions, dtype=float)
    transformed = list(transformed_feature_names)
    raw = list(raw_feature_names)
    if values.ndim != 1 or len(values) != len(transformed):
        raise ValueError("Contributions and transformed feature names must be aligned 1D values.")
    if not np.isfinite(values).all():
        raise ValueError("Contributions must be finite.")
    if not raw or len(raw) != len(set(raw)):
        raise ValueError("Raw feature names must be non-empty and unique.")

    result = {name: 0.0 for name in raw}
    for name, value in zip(transformed, values):
        suffix = name.split("__", 1)[-1]
        matches = [candidate for candidate in raw if suffix == candidate or suffix.startswith(candidate + "_")]
        if len(matches) != 1:
            raise ValueError(f"Cannot map transformed feature '{name}' to exactly one raw feature.")
        result[matches[0]] += float(value)
    if not np.isclose(sum(result.values()), float(values.sum()), atol=1e-12):
        raise ValueError("Attribution aggregation failed conservation.")
    return result


def validate_additivity(
    base_value: float,
    contributions: Sequence[float],
    explained_output: float,
    output_space: str,
    tolerance: float = 1e-6,
) -> AdditivityCheck:
    """Verify that candidate local attributions reconstruct their stated output."""
    if output_space not in {"probability", "log_odds", "raw"}:
        raise ValueError("output_space must be probability, log_odds, or raw.")
    numbers = np.asarray([base_value, explained_output, tolerance, *contributions], dtype=float)
    if not np.isfinite(numbers).all() or tolerance <= 0:
        raise ValueError("Explanation values must be finite and tolerance must be positive.")
    reconstructed = float(base_value + np.sum(contributions))
    error = abs(float(explained_output) - reconstructed)
    return AdditivityCheck(
        output_space=output_space,
        explained_output=float(explained_output),
        reconstructed_output=reconstructed,
        absolute_error=error,
        tolerance=float(tolerance),
        passed=error <= tolerance,
    )


def evaluate_top_feature_stability(
    attribution_runs: Sequence[Mapping[str, float]],
    top_k: int = 3,
    minimum_jaccard: float = 0.6,
) -> ExplanationStability:
    """Check whether top absolute attributions survive repeated perturbation runs."""
    if len(attribution_runs) < 2:
        raise ValueError("At least two attribution runs are required.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if not 0 <= minimum_jaccard <= 1:
        raise ValueError("minimum_jaccard must be in [0, 1].")
    feature_sets = []
    for run in attribution_runs:
        if len(run) < top_k or any(not np.isfinite(value) for value in run.values()):
            raise ValueError("Every run needs enough finite feature attributions.")
        ranked = sorted(run, key=lambda name: (-abs(run[name]), name))[:top_k]
        feature_sets.append(set(ranked))
    similarities = []
    for left in range(len(feature_sets)):
        for right in range(left + 1, len(feature_sets)):
            union = feature_sets[left] | feature_sets[right]
            similarities.append(len(feature_sets[left] & feature_sets[right]) / len(union))
    score = float(np.mean(similarities))
    return ExplanationStability(
        runs=len(attribution_runs),
        top_k=top_k,
        mean_pairwise_jaccard=score,
        passed=score >= minimum_jaccard,
    )
