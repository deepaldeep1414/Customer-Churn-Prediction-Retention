"""
tests/conftest.py
Shared pytest fixtures for unit and integration tests.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


# ── Env setup (must happen before app imports) ───────────────
os.environ.setdefault("DATABASE_URL",        "postgresql://test:test@localhost:5432/churn_test")
os.environ.setdefault("SECRET_KEY",          "test-secret-key-do-not-use-in-prod")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("MODEL_NAME",          "churn_predictor")


# ── Canonical test customer ───────────────────────────────────
TEST_CUSTOMER_ID  = UUID("a1b2c3d4-0001-0001-0001-000000000001")
TEST_OBS_DATE     = date(2024, 1, 14)
TEST_MODEL_VER    = "20240114.xgboost"
TEST_FEATURE_VER  = "v1.0.0"


# ── Minimal feature dict ─────────────────────────────────────
@pytest.fixture
def sample_features() -> dict:
    return {
        "tenure_months":            24,
        "monthly_charges":          79.99,
        "total_charges":            1919.76,
        "avg_monthly_charges_3m":   78.50,
        "charge_change_pct_3m":     0.02,
        "days_overdue_last":        0,
        "contract_type":            "month-to-month",
        "payment_method":           "electronic_check",
        "paperless_billing":        True,
        "phone_service":            True,
        "internet_service":         "fiber_optic",
        "addon_count":              2,
        "tickets_30d":              1,
        "tickets_90d":              2,
        "tickets_total":            5,
        "days_since_last_ticket":   8,
        "pct_billing_tickets_90d":  0.5,
        "pct_cancel_tickets_90d":   0.0,
        "avg_csat_90d":             3.5,
        "charge_x_tenure":          1919.76,
        "overdue_x_cancel_intent":  0.0,
    }


# ── Mock model loader ─────────────────────────────────────────
@pytest.fixture
def mock_model_loader(sample_features):
    loader = MagicMock()
    loader.is_loaded           = True
    loader.model_version       = TEST_MODEL_VER
    loader.model_metadata      = {
        "algorithm":          "xgboost",
        "feature_set_version": TEST_FEATURE_VER,
        "frozen_threshold":    0.5,
        "pr_auc":              0.82,
    }

    # predict returns a realistic probability
    async def mock_predict(features: dict) -> dict:
        return {
            "churn_probability": 0.73,
            "risk_band":         "high",
            "model_version":     TEST_MODEL_VER,
            "feature_set_version": TEST_FEATURE_VER,
            "top_drivers": [
                {
                    "feature":        "pct_cancel_tickets_90d",
                    "label":          "% Cancellation-Intent Tickets (90 days)",
                    "shap_value":     0.41,
                    "direction":      "increases_risk",
                    "customer_value": 0.0,
                },
            ],
        }

    loader.predict = mock_predict
    return loader


# ── Mock feature service ──────────────────────────────────────
@pytest.fixture
def mock_feature_service(sample_features):
    svc = AsyncMock()
    svc.get_features.return_value = sample_features
    return svc


# ── Mock DB pool ──────────────────────────────────────────────
@pytest.fixture
def mock_db_pool():
    pool = AsyncMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value  = None
    conn.fetchval.return_value = True   # customer exists
    conn.fetchrow.return_value  = None  # no cached prediction
    conn.execute.return_value   = None
    return pool, conn


# ── FastAPI test client (sync) ────────────────────────────────
@pytest.fixture
def test_client(mock_model_loader, mock_feature_service):
    import sys
    sys.path.insert(0, "apps/api")

    with patch("services.model_loader.ModelLoader") as MockLoader:
        MockLoader.return_value = mock_model_loader
        from main import app
        app.state.model_loader = mock_model_loader

        with patch("services.feature_service.FeatureService") as MockFS:
            MockFS.return_value = mock_feature_service
            client = TestClient(app, raise_server_exceptions=False)
            yield client


# ── Async HTTPX client ────────────────────────────────────────
@pytest.fixture
async def async_client(mock_model_loader, mock_feature_service):
    import sys
    sys.path.insert(0, "apps/api")

    with patch("services.model_loader.ModelLoader") as MockLoader:
        MockLoader.return_value = mock_model_loader
        from main import app
        app.state.model_loader = mock_model_loader

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {os.environ['SECRET_KEY']}"},
        ) as client:
            yield client


# ── Sample prediction response ────────────────────────────────
@pytest.fixture
def sample_prediction_row():
    return {
        "prediction_id":          uuid4(),
        "customer_id":            TEST_CUSTOMER_ID,
        "model_version":          TEST_MODEL_VER,
        "observation_date":       TEST_OBS_DATE,
        "churn_probability":      0.73,
        "risk_band":              "high",
        "threshold_used":         0.50,
        "scored_at":              datetime.now(timezone.utc),
        "feature_set_version":    TEST_FEATURE_VER,
        "top_drivers":            [],
    }
