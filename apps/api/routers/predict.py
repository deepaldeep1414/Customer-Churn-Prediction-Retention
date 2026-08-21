"""
apps/api/routers/predict.py
Prediction endpoints:
  GET  /v1/customers/{customer_id}/risk
  POST /v1/predict
  POST /v1/predict/batch
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, status

from schemas.request import BatchPredictRequest, PredictRequest
from schemas.response import (
    BatchPredictError,
    BatchPredictResponse,
    PredictionResponse,
)
from services.prediction_service import PredictionService

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_service(request: Request) -> PredictionService:
    loader = getattr(request.app.state, "model_loader", None)
    if loader is None or not loader.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "model_unavailable"},
        )
    return PredictionService(model_loader=loader)


# ── GET /v1/customers/{customer_id}/risk ────────────────────
@router.get(
    "/customers/{customer_id}/risk",
    response_model=PredictionResponse,
    summary="Get latest risk score for a customer",
    responses={
        404: {"description": "Customer not found or no snapshot available"},
        503: {"description": "Model not loaded"},
    },
)
async def get_customer_risk(
    customer_id: UUID,
    request: Request,
    observation_date: date | None = None,
    prediction_horizon_days: int = 30,
) -> PredictionResponse:
    """
    Returns the most recent churn probability, risk band, and top SHAP drivers
    for a given customer. If observation_date is omitted, uses today.
    """
    obs_date = observation_date or date.today()
    service = _get_service(request)

    result = await service.score_single(
        customer_id=customer_id,
        observation_date=obs_date,
        prediction_horizon_days=prediction_horizon_days,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "customer_not_found", "customer_id": str(customer_id)},
        )
    return result


# ── POST /v1/predict ─────────────────────────────────────────
@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single customer",
    status_code=status.HTTP_200_OK,
)
async def predict(
    body: PredictRequest,
    request: Request,
) -> PredictionResponse:
    service = _get_service(request)
    result = await service.score_single(
        customer_id=body.customer_id,
        observation_date=body.observation_date,
        prediction_horizon_days=body.prediction_horizon_days,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "customer_not_found", "customer_id": str(body.customer_id)},
        )
    return result


# ── POST /v1/predict/batch ───────────────────────────────────
@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Score up to 1 000 customers in one request",
    status_code=status.HTTP_200_OK,
)
async def predict_batch(
    body: BatchPredictRequest,
    request: Request,
) -> BatchPredictResponse:
    service = _get_service(request)
    results: list[PredictionResponse] = []
    errors:  list[BatchPredictError]  = []

    # Fan-out concurrent scoring with app-level bounded concurrency.
    # Semaphore is allocated once at startup (app.state.batch_semaphore)
    # to prevent 1000-item batches from exhausting the DB connection pool
    # across both replicas in a multi-instance deployment.
    sem = getattr(request.app.state, "batch_semaphore", asyncio.Semaphore(50))

    async def _score(req: PredictRequest):
        async with sem:
            try:
                res = await service.score_single(
                    customer_id=req.customer_id,
                    observation_date=req.observation_date,
                    prediction_horizon_days=req.prediction_horizon_days,
                )
                if res is None:
                    errors.append(
                        BatchPredictError(
                            customer_id=req.customer_id,
                            error="customer_not_found",
                        )
                    )
                else:
                    results.append(res)
            except Exception as exc:
                logger.warning("Batch score failed for %s: %s", req.customer_id, exc)
                errors.append(
                    BatchPredictError(customer_id=req.customer_id, error=str(exc))
                )

    await asyncio.gather(*[_score(r) for r in body.requests])

    return BatchPredictResponse(
        results=results,
        total=len(body.requests),
        failed=len(errors),
        errors=errors,
    )
