"""Machine-readable release gate assessment with no deployment side effects."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.api.manifest import ArtifactManifest, read_manifest
from src.models.decision_governance import BusinessCostApproval, load_business_cost_approval


class GateEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["pending", "passed", "failed"] = "pending"
    evidence_ref: str | None = Field(default=None, min_length=1, max_length=500)
    approved_by: str | None = Field(default=None, min_length=1, max_length=200)
    note: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def passed_gate_requires_evidence(self):
        if self.status == "passed" and (self.evidence_ref is None or self.approved_by is None):
            raise ValueError("A passed release gate requires evidence_ref and approved_by.")
        return self


class ReleaseEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal["1.0"]
    candidate_model_id: str = Field(min_length=1, max_length=200)
    frozen_git_commit: str | None = None
    fresh_external_evaluation: GateEvidence
    calibration: GateEvidence
    cohort_risk_review: GateEvidence
    privacy_security_review: GateEvidence
    load_soak_test: GateEvidence
    rollback_rehearsal: GateEvidence
    shadow_observation: GateEvidence
    release_approval: GateEvidence

    @model_validator(mode="after")
    def validate_commit(self):
        if self.frozen_git_commit is not None and re.fullmatch(r"[0-9a-f]{40}", self.frozen_git_commit) is None:
            raise ValueError("frozen_git_commit must be a full lowercase Git SHA.")
        return self


@dataclass(frozen=True)
class ReleaseReadinessReport:
    ready: bool
    candidate_model_id: str
    passed_gates: tuple[str, ...]
    blockers: tuple[str, ...]


def load_release_evidence(path: Path) -> ReleaseEvidence:
    try:
        return ReleaseEvidence.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Unable to load release evidence: {exc}") from exc


def assess_release_readiness(
    manifest: ArtifactManifest,
    business_costs: BusinessCostApproval,
    evidence: ReleaseEvidence,
) -> ReleaseReadinessReport:
    """Return explicit blockers; never promote, persist, evaluate, or deploy anything."""
    blockers: list[str] = []
    passed: list[str] = []

    if evidence.candidate_model_id != manifest.model_id:
        blockers.append("Candidate ID does not match the verified artifact manifest.")
    if evidence.frozen_git_commit is None:
        blockers.append("Candidate Git commit has not been frozen.")
    if manifest.calibration.status != "not_fitted":
        blockers.append("Unexpected legacy calibration metadata.")
    else:
        blockers.append("Serving scores have no fitted, evaluated calibrator.")
    if manifest.threshold_status != "legacy_unapproved":
        blockers.append("Unexpected legacy threshold policy metadata.")
    elif business_costs.status != "approved":
        blockers.append("Business costs and contact capacity are not approved.")
    if manifest.final_test.metrics is not None:
        blockers.append("Legacy reserved-final-test metrics must not authorize this release.")

    gate_names = (
        "fresh_external_evaluation", "calibration", "cohort_risk_review",
        "privacy_security_review", "load_soak_test", "rollback_rehearsal",
        "shadow_observation", "release_approval",
    )
    for name in gate_names:
        gate = getattr(evidence, name)
        if gate.status == "passed":
            passed.append(name)
        else:
            blockers.append(f"Release gate '{name}' is {gate.status}.")

    return ReleaseReadinessReport(
        ready=not blockers,
        candidate_model_id=evidence.candidate_model_id,
        passed_gates=tuple(passed),
        blockers=tuple(blockers),
    )


def assess_from_files(manifest_path: Path, costs_path: Path, evidence_path: Path) -> ReleaseReadinessReport:
    return assess_release_readiness(
        read_manifest(manifest_path),
        load_business_cost_approval(costs_path),
        load_release_evidence(evidence_path),
    )


def report_as_json(report: ReleaseReadinessReport) -> str:
    return json.dumps({
        "ready": report.ready,
        "candidate_model_id": report.candidate_model_id,
        "passed_gates": report.passed_gates,
        "blockers": report.blockers,
    }, indent=2)
