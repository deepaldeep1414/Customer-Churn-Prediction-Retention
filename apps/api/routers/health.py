"""
apps/api/routers/health.py
Liveness (/health) and readiness (/ready) endpoints.
"""
from datetime import datetime, timezone
from fastapi import APIRouter, Request
from schemas.response import HealthResponse, ReadyResponse
import asyncpg
import os

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 OK if the process is alive. Used by Docker healthcheck.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=datetime.now(timezone.utc))


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    description="Returns 200 if DB is reachable and champion model is loaded.",
    responses={503: {"description": "Service not ready"}},
)
async def ready(request: Request) -> ReadyResponse:
    db_status = "disconnected"
    model_loaded = False
    model_version = None

    # Check DB
    try:
        conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
        await conn.fetchval("SELECT 1")
        await conn.close()
        db_status = "connected"
    except Exception:
        pass

    # Check model loader
    loader = getattr(request.app.state, "model_loader", None)
    if loader and loader.is_loaded:
        model_loaded = True
        model_version = loader.model_version

    status = "ready" if (db_status == "connected" and model_loaded) else "not_ready"
    return ReadyResponse(
        status=status,
        database=db_status,
        model_loaded=model_loaded,
        model_version=model_version,
        timestamp=datetime.now(timezone.utc),
    )
