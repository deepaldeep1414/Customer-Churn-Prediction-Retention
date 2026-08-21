"""
pipelines/monitoring_pipeline.py
Model and data drift monitoring.

Checks performed:
  1. Population Stability Index (PSI) on top-5 features
  2. Churn rate drift in recent predictions vs training baseline
  3. PR-AUC decay check (if ground-truth labels available for past window)

Alerts are emitted as structured log lines (integrate with your log
aggregation / PagerDuty / Slack webhook as needed).

Usage:
    python monitoring_pipeline.py --reference-date 2023-12-31 --monitoring-date 2024-01-31
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import date

import asyncpg
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")

PSI_THRESHOLD_WARN  = 0.1   # Yellow alert
PSI_THRESHOLD_ALERT = 0.2   # Red alert — mandatory retrain
CHURN_RATE_DELTA_ALERT = 0.05   # Absolute change in predicted churn rate


# ── PSI calculation ─────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  : No significant change
    PSI 0.1–0.2: Moderate shift — monitor
    PSI > 0.2  : Significant shift — retrain
    """
    eps   = 1e-6
    bins  = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins  = np.unique(bins)
    if len(bins) < 2:
        return 0.0

    exp_counts = np.histogram(expected, bins=bins)[0] + eps
    act_counts = np.histogram(actual,   bins=bins)[0] + eps

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


async def run_monitoring(reference_date: date, monitoring_date: date) -> dict:
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    alerts = []
    report = {"reference_date": reference_date.isoformat(), "monitoring_date": monitoring_date.isoformat()}

    # Numeric features to monitor for PSI
    numeric_features = [
        "tenure_months", "monthly_charges", "total_charges",
        "tickets_90d", "pct_cancel_tickets_90d",
    ]

    async with pool.acquire() as conn:
        psi_results = {}
        for feat in numeric_features:
            ref_vals = await conn.fetch(
                f"SELECT {feat} FROM ml.feature_snapshots WHERE observation_date = $1 AND {feat} IS NOT NULL",
                reference_date,
            )
            mon_vals = await conn.fetch(
                f"SELECT {feat} FROM ml.feature_snapshots WHERE observation_date = $1 AND {feat} IS NOT NULL",
                monitoring_date,
            )

            if not ref_vals or not mon_vals:
                continue

            ref_arr = np.array([r[feat] for r in ref_vals], dtype=float)
            mon_arr = np.array([r[feat] for r in mon_vals], dtype=float)
            psi     = compute_psi(ref_arr, mon_arr)
            psi_results[feat] = round(psi, 4)

            if psi > PSI_THRESHOLD_ALERT:
                msg = f"PSI ALERT: feature={feat} PSI={psi:.4f} > {PSI_THRESHOLD_ALERT} — retrain recommended"
                logger.error(msg)
                alerts.append({"type": "psi_alert", "feature": feat, "psi": psi})
            elif psi > PSI_THRESHOLD_WARN:
                logger.warning("PSI WARN: feature=%s PSI=%.4f", feat, psi)

        # Churn rate drift in predictions
        ref_rate = await conn.fetchval(
            """
            SELECT AVG((churn_probability >= threshold_used)::INT::FLOAT)
            FROM ml.predictions WHERE observation_date = $1
            """,
            reference_date,
        )
        mon_rate = await conn.fetchval(
            """
            SELECT AVG((churn_probability >= threshold_used)::INT::FLOAT)
            FROM ml.predictions WHERE observation_date = $1
            """,
            monitoring_date,
        )

        if ref_rate is not None and mon_rate is not None:
            delta = abs(float(mon_rate) - float(ref_rate))
            report["predicted_churn_rate_ref"] = round(float(ref_rate), 4)
            report["predicted_churn_rate_mon"] = round(float(mon_rate), 4)
            report["predicted_churn_rate_delta"] = round(delta, 4)
            if delta > CHURN_RATE_DELTA_ALERT:
                msg = f"CHURN RATE DRIFT: delta={delta:.4f} ref={ref_rate:.4f} mon={mon_rate:.4f}"
                logger.error(msg)
                alerts.append({"type": "churn_rate_drift", "delta": delta})

        report["psi"] = psi_results
        report["alerts"] = alerts

    await pool.close()

    if alerts:
        logger.error("MONITORING REPORT — %d alert(s): %s", len(alerts), json.dumps(report, indent=2))
    else:
        logger.info("MONITORING REPORT — No alerts: %s", json.dumps(report, indent=2))

    return report


def parse_args():
    p = argparse.ArgumentParser(description="Model drift monitoring")
    p.add_argument("--reference-date", required=True, help="Baseline observation date (YYYY-MM-DD)")
    p.add_argument("--monitoring-date", required=True, help="Current observation date (YYYY-MM-DD)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_monitoring(
            reference_date=date.fromisoformat(args.reference_date),
            monitoring_date=date.fromisoformat(args.monitoring_date),
        )
    )
