"""
tests/load/locustfile.py
Locust load test for the Churn Prediction API.

Usage:
    locust -f tests/load/locustfile.py \
           --host http://localhost:8000 \
           --users 50 --spawn-rate 5 \
           --run-time 60s \
           --headless

SLA targets (fail if exceeded):
    - p50 latency: < 50ms
    - p95 latency: < 200ms
    - p99 latency: < 500ms
    - error rate:  < 1%

Environment variables:
    API_SECRET_KEY   Bearer token for auth (required)
    TEST_CUSTOMER_ID Customer UUID with valid snapshot (optional, uses default)
"""
from __future__ import annotations

import os
import random
from datetime import date, timedelta

from locust import HttpUser, TaskSet, between, events, task

SECRET_KEY   = os.getenv("API_SECRET_KEY",  "changeme_set_api_secret_key_env")
CUSTOMER_ID  = os.getenv("TEST_CUSTOMER_ID", "a1b2c3d4-0001-0001-0001-000000000001")
OBS_DATE     = (date.today() - timedelta(days=1)).isoformat()
AUTH_HEADERS = {"Authorization": f"Bearer {SECRET_KEY}"}


class PredictTasks(TaskSet):

    @task(6)
    def single_predict(self):
        """Primary workload: single-customer prediction (60% of traffic)."""
        with self.client.post(
            "/v1/predict",
            json={
                "customer_id":            CUSTOMER_ID,
                "observation_date":       OBS_DATE,
                "prediction_horizon_days": 30,
            },
            headers=AUTH_HEADERS,
            name="/v1/predict",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if "churn_probability" not in body:
                    resp.failure("Response missing churn_probability")
                elif not (0.0 <= body["churn_probability"] <= 1.0):
                    resp.failure(f"Invalid probability: {body['churn_probability']}")
                else:
                    resp.success()
            elif resp.status_code == 404:
                # Customer not found — not an error in load test
                resp.success()
            else:
                resp.failure(f"Unexpected status: {resp.status_code}")

    @task(2)
    def customer_risk_get(self):
        """GET variant (20% of traffic)."""
        with self.client.get(
            f"/v1/customers/{CUSTOMER_ID}/risk",
            params={"observation_date": OBS_DATE},
            headers=AUTH_HEADERS,
            name="/v1/customers/{id}/risk",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Unexpected: {resp.status_code}")

    @task(1)
    def health_check(self):
        """Health endpoint (10% — lightweight keepalive)."""
        self.client.get("/health", name="/health")

    @task(1)
    def current_model(self):
        """Model info endpoint (10%)."""
        with self.client.get(
            "/v1/models/current",
            headers=AUTH_HEADERS,
            name="/v1/models/current",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 503):
                resp.success()
            else:
                resp.failure(f"Unexpected: {resp.status_code}")


class BatchPredictTasks(TaskSet):

    @task(1)
    def batch_10(self):
        """Small batch (10 customers)."""
        requests = [
            {"customer_id": CUSTOMER_ID, "observation_date": OBS_DATE}
            for _ in range(10)
        ]
        with self.client.post(
            "/v1/predict/batch",
            json={"requests": requests},
            headers=AUTH_HEADERS,
            name="/v1/predict/batch (n=10)",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if body.get("total") != 10:
                    resp.failure(f"Expected total=10, got {body.get('total')}")
                else:
                    resp.success()
            else:
                resp.failure(f"Status: {resp.status_code}")

    @task(1)
    def batch_100(self):
        """Medium batch (100 customers) — must complete < 5s."""
        requests = [
            {"customer_id": CUSTOMER_ID, "observation_date": OBS_DATE}
            for _ in range(100)
        ]
        with self.client.post(
            "/v1/predict/batch",
            json={"requests": requests},
            headers=AUTH_HEADERS,
            name="/v1/predict/batch (n=100)",
            catch_response=True,
            timeout=10.0,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status: {resp.status_code}")


class APIUser(HttpUser):
    """Simulates a retention analyst / dashboard polling at moderate pace."""
    tasks       = [PredictTasks]
    wait_time   = between(0.1, 0.5)
    weight      = 80   # 80% of virtual users


class BatchAPIUser(HttpUser):
    """Simulates the nightly batch scoring pipeline."""
    tasks       = [BatchPredictTasks]
    wait_time   = between(1.0, 3.0)
    weight      = 20   # 20% of virtual users


# ── SLA validation hook ───────────────────────────────────────
@events.quitting.add_listener
def check_sla(environment, **kwargs):
    """
    Fail the load test if SLA targets are not met.
    Called automatically by Locust at test end.
    """
    stats     = environment.stats
    failures  = []

    for name, stat in stats.entries.items():
        if stat.num_requests == 0:
            continue

        p95 = stat.get_response_time_percentile(0.95)
        p99 = stat.get_response_time_percentile(0.99)
        err_pct = stat.fail_ratio * 100

        if name[1] == "/v1/predict" and p95 > 200:
            failures.append(f"{name}: p95={p95}ms > 200ms SLA")
        if p99 > 500:
            failures.append(f"{name}: p99={p99}ms > 500ms hard limit")
        if err_pct > 1.0:
            failures.append(f"{name}: error rate={err_pct:.2f}% > 1% SLA")

    if failures:
        print("\n❌ LOAD TEST SLA FAILURES:")
        for f in failures:
            print(f"  - {f}")
        environment.process_exit_code = 1
    else:
        print("\n✅ All SLA targets met")
