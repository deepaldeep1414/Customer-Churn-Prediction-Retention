"""
tests/unit/test_api_schemas.py
Unit tests for Pydantic request/response schema validation.
"""
from __future__ import annotations

from datetime import date, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../apps/api"))

from schemas.request import PredictRequest, BatchPredictRequest
from schemas.response import PredictionResponse, RiskDriver


class TestPredictRequest:

    def test_valid_request(self):
        req = PredictRequest(
            customer_id=uuid4(),
            observation_date=date.today() - timedelta(days=1),
            prediction_horizon_days=30,
        )
        assert req.prediction_horizon_days == 30

    def test_future_observation_date_rejected(self):
        with pytest.raises(ValidationError, match="observation_date cannot be in the future"):
            PredictRequest(
                customer_id=uuid4(),
                observation_date=date.today() + timedelta(days=1),
            )

    def test_invalid_horizon_too_large(self):
        with pytest.raises(ValidationError):
            PredictRequest(
                customer_id=uuid4(),
                observation_date=date.today() - timedelta(days=1),
                prediction_horizon_days=366,
            )

    def test_invalid_horizon_zero(self):
        with pytest.raises(ValidationError):
            PredictRequest(
                customer_id=uuid4(),
                observation_date=date.today() - timedelta(days=1),
                prediction_horizon_days=0,
            )

    def test_default_horizon_is_30(self):
        req = PredictRequest(
            customer_id=uuid4(),
            observation_date=date.today(),
        )
        assert req.prediction_horizon_days == 30

    def test_today_is_valid_observation_date(self):
        req = PredictRequest(customer_id=uuid4(), observation_date=date.today())
        assert req.observation_date == date.today()


class TestBatchPredictRequest:

    def _make_req(self):
        return {
            "customer_id": str(uuid4()),
            "observation_date": (date.today() - timedelta(days=1)).isoformat(),
        }

    def test_valid_batch(self):
        batch = BatchPredictRequest(requests=[self._make_req() for _ in range(10)])
        assert len(batch.requests) == 10

    def test_empty_batch_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(requests=[])

    def test_batch_exceeding_1000_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(requests=[self._make_req() for _ in range(1001)])


class TestPredictionResponse:

    def test_valid_response(self):
        from datetime import datetime, timezone
        resp = PredictionResponse(
            customer_id=uuid4(),
            churn_probability=0.73,
            risk_band="high",
            model_version="1.3.2",
            feature_set_version="v1.0.0",
            observation_date=date.today(),
            prediction_horizon_days=30,
            scored_at=datetime.now(timezone.utc),
            top_drivers=[],
        )
        assert resp.risk_band == "high"
        assert 0.0 <= resp.churn_probability <= 1.0

    def test_churn_probability_out_of_range(self):
        from datetime import datetime, timezone
        with pytest.raises(ValidationError):
            PredictionResponse(
                customer_id=uuid4(),
                churn_probability=1.5,   # invalid
                risk_band="high",
                model_version="1.3.2",
                feature_set_version="v1.0.0",
                observation_date=date.today(),
                prediction_horizon_days=30,
                scored_at=datetime.now(timezone.utc),
            )

    def test_invalid_risk_band_rejected(self):
        from datetime import datetime, timezone
        with pytest.raises(ValidationError):
            PredictionResponse(
                customer_id=uuid4(),
                churn_probability=0.5,
                risk_band="extreme",    # invalid — not in Literal
                model_version="1.0.0",
                feature_set_version="v1.0.0",
                observation_date=date.today(),
                prediction_horizon_days=30,
                scored_at=datetime.now(timezone.utc),
            )
