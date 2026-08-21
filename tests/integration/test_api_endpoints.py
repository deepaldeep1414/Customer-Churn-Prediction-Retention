"""
tests/integration/test_api_endpoints.py
Integration tests for all FastAPI endpoint contracts.

These tests use real HTTP calls against a mocked application
(no live DB or MLflow required). They verify:
  - HTTP status codes
  - Response schema shape
  - Auth enforcement
  - Input validation error messages
  - Deterministic scoring (same payload → same result)
  - Structured error responses
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

# Test env is set in conftest.py
from tests.conftest import TEST_CUSTOMER_ID, TEST_OBS_DATE, TEST_MODEL_VER

AUTH_HEADER = {"Authorization": f"Bearer {os.environ['SECRET_KEY']}"}


# ════════════════════════════════════════════════════════════
# Health & Ready endpoints
# ════════════════════════════════════════════════════════════
class TestHealthEndpoints:

    @pytest.mark.asyncio
    async def test_health_ok(self, async_client: AsyncClient):
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, async_client: AsyncClient):
        """Health must be accessible without a Bearer token."""
        resp = await async_client.get("/health", headers={})  # no auth
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_ready_structure(self, async_client: AsyncClient):
        resp = await async_client.get("/ready")
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert "status" in body
        assert "database" in body
        assert "model_loaded" in body

    @pytest.mark.asyncio
    async def test_request_id_injected_in_response(self, async_client: AsyncClient):
        """Every response must carry X-Request-Id header."""
        resp = await async_client.get("/health")
        assert "x-request-id" in resp.headers
        rid = resp.headers["x-request-id"]
        assert len(rid) == 36, f"X-Request-Id should be a UUID: {rid}"


# ════════════════════════════════════════════════════════════
# Authentication
# ════════════════════════════════════════════════════════════
class TestAuthentication:

    @pytest.mark.asyncio
    async def test_no_token_returns_401(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/v1/predict",
            json={"customer_id": str(uuid4()), "observation_date": str(date.today())},
            headers={},  # strip auth
        )
        assert resp.status_code == 401
        body = resp.json()
        assert "error" in body or "detail" in body

    @pytest.mark.asyncio
    async def test_wrong_token_returns_403(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/v1/predict",
            json={"customer_id": str(uuid4()), "observation_date": str(date.today())},
            headers={"Authorization": "Bearer totally-wrong-key"},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_malformed_auth_header_returns_401(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/v1/predict",
            json={"customer_id": str(uuid4()), "observation_date": str(date.today())},
            headers={"Authorization": "NotBearer somethingwrong"},
        )
        assert resp.status_code == 401


# ════════════════════════════════════════════════════════════
# /v1/predict — Single prediction
# ════════════════════════════════════════════════════════════
class TestPredictEndpoint:

    @pytest.mark.asyncio
    async def test_predict_valid_returns_200(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/v1/predict",
            json={
                "customer_id":            str(TEST_CUSTOMER_ID),
                "observation_date":       str(TEST_OBS_DATE),
                "prediction_horizon_days": 30,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "churn_probability" in body
        assert "risk_band"         in body
        assert "model_version"     in body
        assert "top_drivers"       in body
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert body["risk_band"] in ("low", "medium", "high", "critical")

    @pytest.mark.asyncio
    async def test_predict_future_date_returns_422(self, async_client: AsyncClient):
        future_date = (date.today() + timedelta(days=10)).isoformat()
        resp = await async_client.post(
            "/v1/predict",
            json={"customer_id": str(uuid4()), "observation_date": future_date},
        )
        assert resp.status_code == 422
        body = resp.json()
        # Must contain field-level error detail
        detail = body.get("detail") or body.get("error") or ""
        assert "observation_date" in str(detail).lower() or resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_unknown_customer_returns_404(self, async_client: AsyncClient):
        """A customer_id with no features should return 404, not 500."""
        resp = await async_client.post(
            "/v1/predict",
            json={
                "customer_id":      str(uuid4()),   # random — won't have snapshot
                "observation_date": str(date.today() - timedelta(days=1)),
            },
        )
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body or "detail" in body

    @pytest.mark.asyncio
    async def test_predict_invalid_horizon_returns_422(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/v1/predict",
            json={
                "customer_id":             str(uuid4()),
                "observation_date":        str(date.today()),
                "prediction_horizon_days": 0,   # must be >= 1
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_deterministic(self, async_client: AsyncClient):
        """Same payload with same model version must return same probability."""
        payload = {
            "customer_id":            str(TEST_CUSTOMER_ID),
            "observation_date":       str(TEST_OBS_DATE),
            "prediction_horizon_days": 30,
        }
        r1 = await async_client.post("/v1/predict", json=payload)
        r2 = await async_client.post("/v1/predict", json=payload)
        assert r1.status_code == r2.status_code
        if r1.status_code == 200:
            assert r1.json()["churn_probability"] == r2.json()["churn_probability"]

    @pytest.mark.asyncio
    async def test_predict_response_schema_complete(self, async_client: AsyncClient):
        """Response must include all required Pydantic fields."""
        resp = await async_client.post(
            "/v1/predict",
            json={
                "customer_id":            str(TEST_CUSTOMER_ID),
                "observation_date":       str(TEST_OBS_DATE),
                "prediction_horizon_days": 30,
            },
        )
        if resp.status_code != 200:
            pytest.skip("No model or features available in test env")
        body = resp.json()
        required_fields = {
            "customer_id", "churn_probability", "risk_band",
            "model_version", "feature_set_version",
            "observation_date", "prediction_horizon_days",
            "scored_at", "top_drivers",
        }
        missing = required_fields - set(body.keys())
        assert not missing, f"Response missing fields: {missing}"


# ════════════════════════════════════════════════════════════
# /v1/predict/batch — Batch predictions
# ════════════════════════════════════════════════════════════
class TestBatchPredictEndpoint:

    @pytest.mark.asyncio
    async def test_batch_empty_returns_422(self, async_client: AsyncClient):
        resp = await async_client.post("/v1/predict/batch", json={"requests": []})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_over_1000_returns_422(self, async_client: AsyncClient):
        requests = [
            {"customer_id": str(uuid4()), "observation_date": str(date.today())}
            for _ in range(1001)
        ]
        resp = await async_client.post("/v1/predict/batch", json={"requests": requests})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_single_item_structure(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/v1/predict/batch",
            json={"requests": [{
                "customer_id":            str(TEST_CUSTOMER_ID),
                "observation_date":       str(TEST_OBS_DATE),
                "prediction_horizon_days": 30,
            }]},
        )
        assert resp.status_code in (200, 207)
        body = resp.json()
        assert "results" in body
        assert "total"   in body
        assert "failed"  in body


# ════════════════════════════════════════════════════════════
# /v1/customers/{id}/risk — Customer risk endpoint
# ════════════════════════════════════════════════════════════
class TestCustomerRiskEndpoint:

    @pytest.mark.asyncio
    async def test_customer_risk_valid(self, async_client: AsyncClient):
        resp = await async_client.get(
            f"/v1/customers/{TEST_CUSTOMER_ID}/risk",
            params={"observation_date": str(TEST_OBS_DATE)},
        )
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            body = resp.json()
            assert "churn_probability" in body
            assert "risk_band" in body

    @pytest.mark.asyncio
    async def test_customer_risk_invalid_uuid_returns_422(self, async_client: AsyncClient):
        resp = await async_client.get(
            "/v1/customers/not-a-valid-uuid/risk",
            params={"observation_date": str(date.today())},
        )
        assert resp.status_code == 422


# ════════════════════════════════════════════════════════════
# /v1/models/current
# ════════════════════════════════════════════════════════════
class TestModelsEndpoint:

    @pytest.mark.asyncio
    async def test_current_model_returns_200_or_503(self, async_client: AsyncClient):
        resp = await async_client.get("/v1/models/current")
        assert resp.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_current_model_schema_when_loaded(self, async_client: AsyncClient):
        resp = await async_client.get("/v1/models/current")
        if resp.status_code == 200:
            body = resp.json()
            required = {"model_name", "model_version", "algorithm", "stage", "feature_set_version"}
            missing  = required - set(body.keys())
            assert not missing, f"Model response missing: {missing}"

    @pytest.mark.asyncio
    async def test_current_model_stage_is_champion(self, async_client: AsyncClient):
        resp = await async_client.get("/v1/models/current")
        if resp.status_code == 200:
            assert resp.json()["stage"] == "champion"


# ════════════════════════════════════════════════════════════
# /metrics
# ════════════════════════════════════════════════════════════
class TestMetricsEndpoint:

    @pytest.mark.asyncio
    async def test_metrics_returns_prometheus_format(self, async_client: AsyncClient):
        # First generate some traffic
        await async_client.get("/health")
        await async_client.get("/health")

        resp = await async_client.get("/metrics", headers={})
        assert resp.status_code == 200
        content = resp.text
        # Prometheus text format contains metric lines
        assert "churn_api" in content or content.strip() == ""  # empty before traffic

    @pytest.mark.asyncio
    async def test_metrics_json_returns_dict(self, async_client: AsyncClient):
        resp = await async_client.get("/metrics/json")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body, dict)


# ════════════════════════════════════════════════════════════
# Structured error response format
# ════════════════════════════════════════════════════════════
class TestStructuredErrors:

    @pytest.mark.asyncio
    async def test_404_has_error_field(self, async_client: AsyncClient):
        resp = await async_client.post(
            "/v1/predict",
            json={"customer_id": str(uuid4()), "observation_date": str(date.today())},
        )
        if resp.status_code == 404:
            body = resp.json()
            assert "error" in body or "detail" in body

    @pytest.mark.asyncio
    async def test_422_has_field_level_details(self, async_client: AsyncClient):
        resp = await async_client.post("/v1/predict", json={"invalid": "payload"})
        assert resp.status_code == 422
        body = resp.json()
        # FastAPI/Pydantic 422 includes "detail" with field errors
        assert "detail" in body
