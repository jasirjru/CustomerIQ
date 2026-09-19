from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from scripts.load_test import percentile
from src.evaluation.calibration import evaluate_calibration, select_and_fit_calibrator
from src.evaluation.fairness import evaluate_binary_cohorts
from src.models.decision_governance import require_approved_business_costs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_calibration_metrics_are_numerically_correct():
    report = evaluate_calibration([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], n_bins=2)
    assert report.sample_count == 4
    assert report.brier_score == pytest.approx(0.025)
    assert report.expected_calibration_error == pytest.approx(0.15)
    assert sum(item.count for item in report.bins) == 4


@pytest.mark.parametrize("scores", [[-0.1], [1.1], [float("nan")], [float("inf")]])
def test_calibration_rejects_invalid_probabilities(scores):
    with pytest.raises(ValueError):
        evaluate_calibration([1], scores)


def test_calibrator_selection_is_training_only_and_returns_probabilities():
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        random_state=7,
    )
    result = select_and_fit_calibrator(
        LogisticRegression(max_iter=500),
        X,
        y,
        outer_splits=3,
        inner_splits=3,
    )
    assert result.method in {"sigmoid", "isotonic"}
    assert set(result.fold_brier_scores) == {"sigmoid", "isotonic"}
    assert all(len(scores) == 3 for scores in result.fold_brier_scores.values())
    probabilities = result.fitted_estimator.predict_proba(X[:5])[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((0 <= probabilities) & (probabilities <= 1)).all()


def test_cohort_report_exposes_rate_gaps_and_small_groups():
    report = evaluate_binary_cohorts(
        [1, 1, 0, 0, 1, 0, 1],
        [0.9, 0.7, 0.6, 0.1, 0.8, 0.2, 0.4],
        ["A", "A", "A", "A", "B", "B", "tiny"],
        threshold=0.5,
        minimum_group_size=2,
    )
    assert [row.cohort for row in report.cohorts] == ["A", "B"]
    assert report.excluded_small_cohorts == ("tiny",)
    assert report.max_review_rate_gap == pytest.approx(0.25)
    assert report.max_true_positive_rate_gap == pytest.approx(0.0)
    assert report.max_false_positive_rate_gap == pytest.approx(0.5)


def test_unapproved_business_costs_fail_closed():
    with pytest.raises(PermissionError, match="not approved"):
        require_approved_business_costs(PROJECT_ROOT / "docs" / "business-costs.json")


def test_load_probe_percentiles():
    assert percentile([4, 1, 3, 2], 0.5) == 2
    assert percentile([4, 1, 3, 2], 0.95) == 4
