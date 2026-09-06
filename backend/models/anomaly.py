"""
backend/models/anomaly.py
==========================
Pydantic v2 data model for anomaly reports.

AnomalyReport
    Represents one detected anomaly linked to a specific experiment.
    Created by Phase 3 Anomaly_Detector, stored by StateManager,
    and used by Phase 4 Recommender_Agent as JSON input.

Detection rules (from DESIGN_REVIEW_CHANGES.md, and its "Architecture
Revision" entry for the validation_collapse update)
--------------------------------------------------------------------------
- ``outlier_detection``    : val_loss > 3 leave-one-out std devs from the
                             mean of the rest of its (dataset, model_type,
                             hyperparameters) group
- ``loss_divergence``      : training loss increases over final 20% of epochs
- ``validation_collapse``  : classification accuracy is not statistically
                             distinguishable from random guessing for the
                             dataset's number of classes (class-count-aware
                             z-test, not a flat threshold)

Explanations are template-based (no LLM calls) — see Phase 3
backend/tools/anomaly_templates.py for template strings.

Requirements
------------
4.1  Anomaly_Detector identifies outliers
4.6  Anomaly_Detector provides natural language explanation
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AnomalyReport(BaseModel):
    """One detected anomaly linked to a specific experiment.

    Stored in the ``anomalies`` table by StateManager.
    Severity mapping:
    - ``validation_collapse`` -> ``critical``
    - ``outlier_detection``, ``loss_divergence`` -> ``warning``

    ORM round-trip
    --------------
    ``ConfigDict(from_attributes=True)`` enables construction from
    a SQLAlchemy AnomalyModel row via::

        AnomalyReport.model_validate(orm_row, from_attributes=True)

    Requirements covered: 4.1, 4.6
    """

    anomaly_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID primary key",
    )
    experiment_id: str = Field(description="FK to experiments table")
    rule: Literal[
        "outlier_detection", "loss_divergence", "validation_collapse"
    ] = Field(description="Detection rule that triggered this report")
    explanation: str = Field(
        description="Template-generated natural language explanation (no LLM call)"
    )
    severity: Literal["warning", "critical"] = Field(
        description="'critical' for validation_collapse, 'warning' for others"
    )
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Anomaly detection timestamp (UTC)",
    )

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "anomaly_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "experiment_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                "rule": "outlier_detection",
                "explanation": (
                    "Validation val_loss of 2.301 is 3.7 std devs from mean "
                    "(0.182). This suggests numerical instability or incorrect "
                    "configuration."
                ),
                "severity": "warning",
                "detected_at": "2024-01-15T10:35:00Z",
            }
        },
    )
