"""Tests for isolated evaluation, repeated CV, and validation-only thresholding."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.data.make_dataset import split_train_validation_test, final_test_sealed, load_development_partitions
from src.models.model_selection import (
    fit_champion_and_select_threshold,
    optimize_validation_threshold,
    select_champion_repeated_cv,
)


@pytest.fixture
def evaluation_frame():
    rows = []
    for index in range(200):
        target = index % 4 == 0
        rows.append(
            {
                "row_id": index,
                "tenure": index % 73,
                "MonthlyCharges": 20.0 + (index % 90),
                "Contract": "Month-to-month" if target else "Two year",
                "Churn": int(target),
            }
        )
    return pd.DataFrame(rows).set_index("row_id")


def test_three_way_split_is_deterministic_disjoint_and_60_20_20(evaluation_frame):
    first = split_train_validation_test(evaluation_frame)
    second = split_train_validation_test(evaluation_frame)

    assert len(first.X_train) == 120
    assert len(first.X_validation) == 40
    assert len(first.X_final_test) == 40

    train_ids = set(first.X_train.index)
    validation_ids = set(first.X_validation.index)
    final_test_ids = set(first.X_final_test.index)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(final_test_ids)
    assert validation_ids.isdisjoint(final_test_ids)
    assert train_ids | validation_ids | final_test_ids == set(evaluation_frame.index)

    assert first.X_train.index.equals(second.X_train.index)
    assert first.X_validation.index.equals(second.X_validation.index)
    assert first.X_final_test.index.equals(second.X_final_test.index)
    assert first.y_train.mean() == pytest.approx(0.25)
    assert first.y_validation.mean() == pytest.approx(0.25)
    assert first.y_final_test.mean() == pytest.approx(0.25)


def test_repeated_cv_is_deterministic_and_refits_raw_feature_pipeline(evaluation_frame):
    partitions = split_train_validation_test(evaluation_frame)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    }

    first = select_champion_repeated_cv(
        partitions.X_train,
        partitions.y_train,
        candidates=candidates,
        n_splits=3,
        n_repeats=2,
        n_jobs=1,
    )
    second = select_champion_repeated_cv(
        partitions.X_train,
        partitions.y_train,
        candidates=candidates,
        n_splits=3,
        n_repeats=2,
        n_jobs=1,
    )

    assert first.champion_name == second.champion_name
    pd.testing.assert_frame_equal(first.leaderboard, second.leaderboard)
    assert first.leaderboard["CV_Folds"].eq(6).all()
    assert list(first.champion_pipeline.named_steps) == ["preprocessor", "classifier"]
    assert not hasattr(first.champion_pipeline.named_steps["preprocessor"], "transformers_")


def test_threshold_optimizer_uses_stated_validation_costs():
    y_validation = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.10, 0.20, 0.80, 0.30, 0.90])

    result = optimize_validation_threshold(
        y_validation,
        probabilities,
        false_negative_cost=5.0,
        false_positive_cost=1.0,
    )

    assert result.threshold == pytest.approx(0.30)
    assert result.total_cost == pytest.approx(1.0)
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.recall == pytest.approx(1.0)


def test_fit_and_threshold_path_never_requires_final_test(evaluation_frame):
    partitions = split_train_validation_test(evaluation_frame)
    selection = select_champion_repeated_cv(
        partitions.X_train,
        partitions.y_train,
        candidates={"Logistic Regression": LogisticRegression(max_iter=500)},
        n_splits=3,
        n_repeats=1,
        n_jobs=1,
    )

    fitted, threshold = fit_champion_and_select_threshold(
        selection,
        partitions.X_train,
        partitions.y_train,
        partitions.X_validation,
        partitions.y_validation,
        false_negative_cost=5.0,
        false_positive_cost=1.0,
    )

    assert hasattr(fitted, "predict_proba")
    assert 0.0 <= threshold.threshold <= 1.0
    assert len(partitions.X_final_test) == 40


def test_development_loader_counts_disjointness_and_access_guard(monkeypatch):
    opened = []
    original = pd.read_csv
    def record(path, *args, **kwargs):
        opened.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", record)
    train, validation, y_train, y_validation = load_development_partitions()
    assert (len(train), len(validation)) == (4225, 1409)
    assert set(train.index).isdisjoint(validation.index)
    assert abs(y_train.mean() - y_validation.mean()) < 0.001
    assert set(opened) == {"X_train_raw.csv", "y_train.csv"}
    with final_test_sealed(), pytest.raises(PermissionError):
        open("data/processed/y_test.csv")


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf")])
def test_threshold_rejects_invalid_costs(cost):
    with pytest.raises(ValueError):
        optimize_validation_threshold([0, 1], [.1, .9], false_negative_cost=cost)
