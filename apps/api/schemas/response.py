"""
apps/api/schemas/response.py
Pydantic response schemas for all API endpoints.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Risk driver ──────────────────────────────────────────────
class RiskDriver(BaseModel):
    """A single SHAP driver for a prediction."""

    feature:        str   = Field(description="Internal feature name")
    label:          str   = Field(description="Business-friendly display label")
    shap_value:     float = Field(description="SHAP value (magnitude = importance)")
    direction:      Literal["increases_risk", "decreases_risk"]
    customer_value: float | str | bool | None = Field(
        default=None, description="Raw feature value for this customer"
    )


# ── Core prediction payload ──────────────────────────────────
class PredictionResponse(BaseModel):
    """Scored prediction for a single customer."""

    customer_id:              UUID
    churn_probability:        float  = Field(ge=0.0, le=1.0)
    risk_band:                Literal["low", "medium", "high", "critical"]
    model_version:            str
    feature_set_version:      str
    observation_date:         date
    prediction_horizon_days:  int
    scored_at:                datetime
    top_drivers:              List[RiskDriver] = Field(default_factory=list, max_length=10)


# ── Batch wrapper ────────────────────────────────────────────
class BatchPredictError(BaseModel):
    customer_id: UUID
    error:       str


class BatchPredictResponse(BaseModel):
    results: List[PredictionResponse]
    total:   int
    failed:  int
    errors:  List[BatchPredictError] = Field(default_factory=list)


# ── Health / Ready ───────────────────────────────────────────
class HealthResponse(BaseModel):
    status:    Literal["ok"]
    timestamp: datetime


class ReadyResponse(BaseModel):
    status:        Literal["ready", "not_ready"]
    database:      Literal["connected", "disconnected"]
    model_loaded:  bool
    model_version: Optional[str] = None
    timestamp:     datetime


# ── Model metadata ───────────────────────────────────────────
class ModelInfoResponse(BaseModel):
    model_name:                str
    model_version:             str
    algorithm:                 str
    stage:                     Literal["staging", "champion", "archived"]
    feature_set_version:       str
    pr_auc:                    Optional[float] = None
    roc_auc:                   Optional[float] = None
    brier_score:               Optional[float] = None
    recall_at_threshold:       Optional[float] = None
    precision_at_threshold:    Optional[float] = None
    f1_at_threshold:           Optional[float] = None
    threshold:                 Optional[float] = None
    promoted_at:               Optional[datetime] = None
    promoted_by:               Optional[str] = None


# ── Error ────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    error:      str
    detail:     Optional[str] = None
    customer_id: Optional[UUID] = None
