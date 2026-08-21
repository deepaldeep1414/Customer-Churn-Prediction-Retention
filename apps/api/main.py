"""
apps/api/main.py
FastAPI application entry point — production configuration.

Middleware stack (outer → inner):
  1. StructuredLoggingMiddleware  — request_id, duration, model_version
  2. MetricsMiddleware            — p50/p95/p99 latency sampling
  3. RateLimitMiddleware          — token-bucket per client
  4. AuthMiddleware               — Bearer token validation
  5. CORSMiddleware               — browser CORS headers
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from middleware.auth import AuthMiddleware
from middleware.logging_middleware import StructuredLoggingMiddleware
from middleware.metrics import MetricsMiddleware, prometheus_metrics_text, get_metrics_summary
from middleware.rate_limit import RateLimitMiddleware
from routers import health, predict, models
from services.model_loader import ModelLoader

# ── Logging setup ────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(message)s",          # Structured middleware emits JSON; keep root simple
)
# Access log uses JSON; suppress uvicorn's default access log to avoid duplication
logging.getLogger("uvicorn.access").propagate = False

logger = logging.getLogger(__name__)


# ── Model loader singleton ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load champion model on startup; graceful degraded mode if MLflow unavailable."""
    logger.info(json.dumps({"event": "startup", "phase": "loading_model"}))
    loader = ModelLoader()
    await loader.load()
    app.state.model_loader = loader

    # Shared semaphore for batch endpoint — prevents 1000-item batches from
    # saturating the DB connection pool. Bound = min(BATCH_CONCURRENCY_LIMIT, 50).
    batch_concurrency = int(os.getenv("BATCH_CONCURRENCY_LIMIT", "50"))
    app.state.batch_semaphore = asyncio.Semaphore(batch_concurrency)
    logger.info(json.dumps({"event": "startup", "batch_concurrency": batch_concurrency}))

    if loader.is_loaded:
        logger.info(json.dumps({
            "event":         "startup",
            "phase":         "model_ready",
            "model_version": loader.model_version,
            "algorithm":     loader.model_metadata.get("algorithm"),
        }))
    else:
        logger.warning(json.dumps({
            "event": "startup",
            "phase": "model_degraded",
            "detail": "No champion model loaded — API starts in degraded mode",
        }))

    yield

    logger.info(json.dumps({"event": "shutdown"}))


# ── Application ──────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Decision-support service scoring customers on churn probability, "
        "risk band, and top SHAP drivers. Not for autonomous retention actions."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── Middleware (registered in reverse order — last registered = outermost) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8088").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
    expose_headers=["X-Request-Id"],
)
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(StructuredLoggingMiddleware)

# ── Routers ──────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Predictions"])
app.include_router(models.router, prefix="/v1", tags=["Models"])


# ── Metrics endpoint ─────────────────────────────────────────
@app.get(
    "/metrics",
    include_in_schema=False,   # Hidden from public Swagger UI
    response_class=PlainTextResponse,
)
async def metrics_endpoint(request: Request) -> PlainTextResponse:
    """
    Prometheus-compatible metrics endpoint.
    Returns p50/p95/p99 latency and error counts per route.
    Restrict access to internal network in production (nginx allow/deny).
    """
    return PlainTextResponse(
        content=prometheus_metrics_text(),
        media_type="text/plain; version=0.0.4",
    )


@app.get(
    "/metrics/json",
    include_in_schema=False,
    summary="Metrics in JSON format",
)
async def metrics_json(request: Request) -> dict:
    """JSON version of the metrics endpoint for human inspection."""
    return get_metrics_summary()
