"""Fail-closed loading of business assumptions used by decision policies."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BusinessCostApproval(BaseModel):
    """Versionable approval record; incomplete drafts are valid but unusable."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["not_approved", "approved"]
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    false_negative: float | None = Field(default=None, ge=0)
    false_positive: float | None = Field(default=None, ge=0)
    contact_capacity: float | None = Field(default=None, gt=0, le=1)
    intervention_success_assumption: float | None = Field(default=None, ge=0, le=1)
    approved_by: str | None = Field(default=None, min_length=1)
    approved_at: datetime | None = None
    evidence_source: str | None = Field(default=None, min_length=1)
    note: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def require_complete_approval(self):
        if self.status == "approved":
            required = (
                "currency",
                "false_negative",
                "false_positive",
                "contact_capacity",
                "intervention_success_assumption",
                "approved_by",
                "approved_at",
                "evidence_source",
            )
            missing = [name for name in required if getattr(self, name) is None]
            if missing:
                raise ValueError(f"Approved business costs are incomplete: {', '.join(missing)}")
            if self.approved_at.utcoffset() is None:
                raise ValueError("approved_at must include a timezone offset.")
        return self


def load_business_cost_approval(path: Path) -> BusinessCostApproval:
    """Load and validate an approval record without making it deployable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load business cost approval: {exc}") from exc
    return BusinessCostApproval.model_validate(payload)


def require_approved_business_costs(path: Path) -> BusinessCostApproval:
    """Refuse threshold work until an accountable approval is complete."""
    approval = load_business_cost_approval(path)
    if approval.status != "approved":
        raise PermissionError("Business cost assumptions are not approved.")
    return approval
