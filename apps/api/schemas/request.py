"""
apps/api/schemas/request.py
Pydantic request schemas with strict validation.
"""
from __future__ import annotations

from datetime import date
from typing import Annotated, List
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Shared constraint types ──────────────────────────────────
CustomerId  = Annotated[UUID, Field(description="Customer UUID (internal)")]
ObsDate     = Annotated[date, Field(description="Observation date (features cut-off)")]
HorizonDays = Annotated[int,  Field(ge=1, le=365, description="Prediction horizon in days")]


class PredictRequest(BaseModel):
    """Single customer prediction request."""

    customer_id:              CustomerId
    observation_date:         ObsDate
    prediction_horizon_days:  HorizonDays = 30

    @field_validator("observation_date")
    @classmethod
    def date_not_future(cls, v: date) -> date:
        if v > date.today():
            raise ValueError("observation_date cannot be in the future")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "customer_id":             "a1b2c3d4-0001-0001-0001-000000000001",
                "observation_date":        "2024-01-14",
                "prediction_horizon_days": 30,
            }
        }
    }


class BatchPredictRequest(BaseModel):
    """Batch prediction request (max 1 000 records)."""

    requests: List[PredictRequest] = Field(min_length=1, max_length=1000)

    model_config = {
        "json_schema_extra": {
            "example": {
                "requests": [
                    {
                        "customer_id":             "a1b2c3d4-0001-0001-0001-000000000001",
                        "observation_date":        "2024-01-14",
                        "prediction_horizon_days": 30,
                    }
                ]
            }
        }
    }
