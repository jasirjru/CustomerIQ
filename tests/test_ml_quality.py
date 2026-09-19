import json
from pathlib import Path

import numpy as np
import pytest

from src.api.manifest import read_manifest
from src.evaluation.drift import categorical_distribution_drift, numeric_population_stability
from src.evaluation.local_explanations import (
    aggregate_transformed_attributions,
    evaluate_top_feature_stability,
    validate_additivity,
)
from src.models.decision_governance import load_business_cost_approval
from src.release.readiness import ReleaseEvidence, assess_release_readiness, load_release_evidence


ROOT = Path(__file__).resolve().parents[1]


def test_numeric_drift_is_reference_binned_and_detects_shift():
    reference = np.linspace(0, 1, 200)
    stable = numeric_population_stability(reference, reference + 0.001)
    shifted = numeric_population_stability(reference, reference + 2)
    assert stable.severity == "stable"
    assert shifted.severity == "critical"
    assert shifted.population_stability_index > stable.population_stability_index


def test_categorical_drift_reports_unseen_rate():
    reference = ["DSL"] * 50 + ["Fiber"] * 50
    current = ["DSL"] * 20 + ["Fiber"] * 20 + ["Satellite"] * 60
    report = categorical_distribution_drift(reference, current)
    assert report.unseen_current_rate == pytest.approx(0.6)
    assert report.severity == "critical"
    assert "Satellite" in report.categories


def test_local_attribution_aggregation_conserves_sum():
    result = aggregate_transformed_attributions(
        [0.2, -0.1, 0.05],
        ["num__tenure", "cat__Contract_Month-to-month", "cat__Contract_Two year"],
        ["tenure", "Contract"],
    )
    assert result == pytest.approx({"tenure": 0.2, "Contract": -0.05})
    assert sum(result.values()) == pytest.approx(0.15)


def test_additivity_and_stability_are_explicit_release_gates():
    assert validate_additivity(0.4, [0.1, -0.05], 0.45, "probability").passed
    assert not validate_additivity(0.4, [0.1], 0.8, "probability").passed
    stable = evaluate_top_feature_stability([
        {"tenure": 0.5, "Contract": 0.4, "Charges": 0.1},
        {"Contract": 0.6, "tenure": 0.3, "Charges": 0.1},
    ], top_k=2)
    assert stable.passed
    assert stable.mean_pairwise_jaccard == 1


def test_release_readiness_fails_closed_without_touching_data():
    manifest = read_manifest(ROOT / "models" / "manifest.v1.json")
    costs = load_business_cost_approval(ROOT / "docs" / "business-costs.json")
    evidence = load_release_evidence(ROOT / "docs" / "release-evidence.template.json")
    report = assess_release_readiness(manifest, costs, evidence)
    assert report.ready is False
    assert report.passed_gates == ()
    assert any("calibrator" in blocker for blocker in report.blockers)
    assert any("fresh_external_evaluation" in blocker for blocker in report.blockers)


def test_passed_release_gate_requires_evidence_and_approver():
    payload = json.loads((ROOT / "docs" / "release-evidence.template.json").read_text(encoding="utf-8"))
    payload["privacy_security_review"]["status"] = "passed"
    with pytest.raises(ValueError, match="evidence_ref and approved_by"):
        ReleaseEvidence.model_validate(payload)


@pytest.mark.parametrize("sample", [[], [1] * 19, [1] * 19 + [float("nan")]])
def test_numeric_drift_rejects_unsafe_samples(sample):
    with pytest.raises(ValueError):
        numeric_population_stability(sample, [1.0] * 20)
