"""
pipelines/scoring_pipeline.py
Daily batch scoring: reads all active customers, retrieves pre-computed snapshots,
runs champion model, and writes predictions to ml.predictions.

Usage:
    python scoring_pipeline.py --observation-date 2024-01-15
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone
from uuid import UUID

import asyncpg
import mlflow.pyfunc
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL  = os.getenv("DATABASE_URL")
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME    = os.getenv("MODEL_NAME", "churn_predictor")
BATCH_SIZE    = 200

RISK_BAND_LOW  = float(os.getenv("RISK_BAND_LOW",  "0.30"))
RISK_BAND_MED  = float(os.getenv("RISK_BAND_MED",  "0.55"))
RISK_BAND_HIGH = float(os.getenv("RISK_BAND_HIGH", "0.75"))

FEATURE_COLUMNS = [
    "tenure_months", "monthly_charges", "total_charges",
    "avg_monthly_charges_3m", "charge_change_pct_3m", "days_overdue_last",
    "contract_type", "payment_method", "paperless_billing",
    "phone_service", "internet_service", "addon_count",
    "tickets_30d", "tickets_90d", "tickets_total",
    "days_since_last_ticket", "pct_billing_tickets_90d",
    "pct_cancel_tickets_90d", "avg_csat_90d",
    "charge_x_tenure", "overdue_x_cancel_intent",
]


def _risk_band(prob: float) -> str:
    if prob < RISK_BAND_LOW:  return "low"
    if prob < RISK_BAND_MED:  return "medium"
    if prob < RISK_BAND_HIGH: return "high"
    return "critical"


async def run_batch_scoring(observation_date: date, horizon_days: int = 30) -> dict:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)

    # Load champion model
    client   = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError("No Production model found in MLflow registry.")
    mv          = versions[0]
    model_uri   = f"models:/{MODEL_NAME}/Production"
    model       = mlflow.pyfunc.load_model(model_uri)
    model_version_str = mv.version
    run         = client.get_run(mv.run_id)
    threshold   = float(run.data.params.get("frozen_threshold", 0.5))
    feat_ver    = run.data.params.get("feature_set_version", "v1.0.0")

    logger.info(
        "Loaded model version=%s | threshold=%.4f | feature_version=%s",
        model_version_str, threshold, feat_ver,
    )

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=4, max_size=12)
    scored = failed = 0
    scored_at = datetime.now(timezone.utc)

    async with pool.acquire() as conn:
        # Fetch all customers with a snapshot for this observation_date
        rows = await conn.fetch(
            """
            SELECT fs.*
            FROM ml.feature_snapshots fs
            WHERE fs.observation_date        = $1
              AND fs.prediction_horizon_days = $2
              AND fs.feature_set_version     = $3
            ORDER BY fs.customer_id
            """,
            observation_date, horizon_days, feat_ver,
        )

    logger.info("Snapshots found: %d | observation_date=%s", len(rows), observation_date)

    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]

        # Build feature matrix
        records = []
        cids    = []
        snaps   = []
        for r in batch:
            records.append({col: r[col] for col in FEATURE_COLUMNS if col in r.keys()})
            cids.append(r["customer_id"])
            snaps.append(r["snapshot_id"])

        X = pd.DataFrame(records, columns=FEATURE_COLUMNS)

        try:
            probs_raw = model.predict(X)
            if hasattr(probs_raw, "ndim") and probs_raw.ndim == 2:
                probs = probs_raw[:, 1]
            else:
                probs = np.asarray(probs_raw)

            async with pool.acquire() as conn:
                async with conn.transaction():
                    for cid, snap_id, prob in zip(cids, snaps, probs):
                        band = _risk_band(float(prob))
                        await conn.execute(
                            """
                            INSERT INTO ml.predictions
                                (customer_id, model_version, snapshot_id, observation_date,
                                 churn_probability, risk_band, threshold_used,
                                 scored_at, request_source)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'batch')
                            ON CONFLICT DO NOTHING
                            """,
                            cid, model_version_str, snap_id, observation_date,
                            round(float(prob), 5), band, threshold, scored_at,
                        )
                    scored += len(batch)

        except Exception as exc:
            logger.error("Batch scoring failed (start=%d): %s", batch_start, exc)
            failed += len(batch)

        logger.info(
            "Progress: scored=%d failed=%d / total=%d",
            scored, failed, len(rows),
        )

    await pool.close()
    summary = {
        "observation_date": observation_date.isoformat(),
        "model_version":    model_version_str,
        "total_snapshots":  len(rows),
        "scored":           scored,
        "failed":           failed,
        "scored_at":        scored_at.isoformat(),
    }
    logger.info("Batch scoring complete: %s", summary)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Daily batch scoring pipeline")
    p.add_argument("--observation-date", default=date.today().isoformat())
    p.add_argument("--horizon-days", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_batch_scoring(
            observation_date=date.fromisoformat(args.observation_date),
            horizon_days=args.horizon_days,
        )
    )
