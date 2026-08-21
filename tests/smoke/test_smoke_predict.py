"""
tests/smoke/test_smoke_predict.py
Post-deployment smoke tests.
Runs against a live API endpoint after every deployment.
Requires env var BASE_URL and API_SECRET_KEY.

Usage:
    BASE_URL=https://api.churn.internal API_SECRET_KEY=... pytest tests/smoke/
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from uuid import uuid4

import httpx
import pytest

BASE_URL   = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY    = os.getenv("API_SECRET_KEY", "smoke-test-key")
HEADERS    = {"Authorization": f"Bearer {API_KEY}"}
TIMEOUT    = 15.0


@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=TIMEOUT) as c:
        yield c


class TestHealthSmoke:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "timestamp" in body

    def test_ready_returns_200(self, client):
        resp = client.get("/ready")
        # Accept 200 (ready) or 503 (not ready in degraded mode)
        assert resp.status_code in (200, 503)
        body = resp.json()
        assert "status" in body
        assert "database" in body

    def test_health_no_auth_required(self):
        """Health endpoint must not require Bearer token."""
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
            resp = c.get("/health")
        assert resp.status_code == 200


class TestAuthSmoke:

    def test_predict_without_token_returns_401(self):
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
            resp = c.post("/v1/predict", json={
                "customer_id": str(uuid4()),
                "observation_date": date.today().isoformat(),
            })
        assert resp.status_code == 401

    def test_predict_with_wrong_token_returns_403(self):
        with httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": "Bearer wrong-token"},
            timeout=TIMEOUT,
        ) as c:
            resp = c.post("/v1/predict", json={
                "customer_id": str(uuid4()),
                "observation_date": date.today().isoformat(),
            })
        assert resp.status_code == 403


class TestPredictSmoke:

    def test_predict_unknown_customer_returns_404(self, client):
        resp = client.post("/v1/predict", json={
            "customer_id": str(uuid4()),   # random UUID — won't exist
            "observation_date": (date.today() - timedelta(days=1)).isoformat(),
        })
        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body or "detail" in body

    def test_predict_future_date_returns_422(self, client):
        resp = client.post("/v1/predict", json={
            "customer_id": str(uuid4()),
            "observation_date": (date.today() + timedelta(days=10)).isoformat(),
        })
        assert resp.status_code == 422

    def test_batch_empty_returns_422(self, client):
        resp = client.post("/v1/predict/batch", json={"requests": []})
        assert resp.status_code == 422

    def test_batch_over_1000_returns_422(self, client):
        requests = [
            {"customer_id": str(uuid4()), "observation_date": date.today().isoformat()}
            for _ in range(1001)
        ]
        resp = client.post("/v1/predict/batch", json={"requests": requests})
        assert resp.status_code == 422


class TestModelsSmoke:

    def test_current_model_endpoint(self, client):
        resp = client.get("/v1/models/current")
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            body = resp.json()
            assert "model_version" in body
            assert "algorithm" in body
            assert "stage" in body
            assert body["stage"] == "champion"

    def test_openapi_schema_accessible(self):
        """OpenAPI schema should be publicly accessible."""
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
            resp = c.get("/api/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "openapi" in schema
