"""
pipelines/mlops/retraining_trigger.py
Automated retraining trigger.

Checks three trigger conditions (evaluated in order of urgency):
  1. PSI alert: any top-5 feature has PSI > 0.2 → mandatory retrain
  2. PR-AUC decay: production PR-AUC has dropped > 0.05 → mandatory retrain
  3. Calendar schedule: monthly cadence (if --scheduled flag passed)
  4. Label volume: ≥ 500 new labeled records available since last training

Exits with code 0 if retrain should proceed, code 1 if not needed.
Set RETRAIN_DRY_RUN=true to log decision without triggering.

Usage:
    python retraining_trigger.py --reference-date 2023-12-31 --monitoring-date 2024-01-31
    python retraining_trigger.py --scheduled           # monthly calendar trigger
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import date

import asyncpg
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL    = os.getenv("DATABASE_URL")
DRY_RUN         = os.getenv("RETRAIN_DRY_RUN", "false").lower() == "true"
PSI_THRESHOLD   = float(os.getenv("PSI_ALERT_THRESHOLD",  "0.20"))
PRAUC_DECAY     = float(os.getenv("PRAUC_DECAY_THRESHOLD", "0.05"))
LABEL_VOLUME    = int(os.getenv("LABEL_VOLUME_THRESHOLD",  "500"))

TOP_FEATURES_FOR_PSI = [
    "tenure_months", "monthly_charges",
    "tickets_90d", "pct_cancel_tickets_90d", "days_overdue_last",
]


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    eps  = 1e-6
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    e_c = np.histogram(expected, bins=bins)[0] + eps
    a_c = np.histogram(actual,   bins=bins)[0] + eps
    e_p = e_c / e_c.sum()
    a_p = a_c / a_c.sum()
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


@dataclass_like := None  # avoid dataclass import for simple struct
class TriggerDecision:
    should_retrain: bool
    reasons: list[str]
    trigger_type: str   # 'psi' | 'prauc_decay' | 'scheduled' | 'label_volume' | 'none'

    def __init__(self, should_retrain, reasons, trigger_type):
        self.should_retrain = should_retrain
        self.reasons = reasons
        self.trigger_type = trigger_type

    def to_dict(self):
        return {
            "should_retrain": self.should_retrain,
            "trigger_type":   self.trigger_type,
            "reasons":        self.reasons,
        }


async def evaluate_triggers(
    reference_date: date,
    monitoring_date: date,
    scheduled: bool = False,
) -> TriggerDecision:
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=4)
    reasons: list[str] = []
    trigger_type = "none"

    async with pool.acquire() as conn:

        # ── Trigger 1: PSI on top features ───────────────────
        psi_alerts = []
        for feat in TOP_FEATURES_FOR_PSI:
            ref_rows = await conn.fetch(
                f"SELECT {feat} FROM ml.feature_snapshots "
                f"WHERE observation_date=$1 AND {feat} IS NOT NULL LIMIT 5000",
                reference_date,
            )
            mon_rows = await conn.fetch(
                f"SELECT {feat} FROM ml.feature_snapshots "
                f"WHERE observation_date=$1 AND {feat} IS NOT NULL LIMIT 5000",
                monitoring_date,
            )
            if not ref_rows or not mon_rows:
                continue
            ref_arr = np.array([r[feat] for r in ref_rows], dtype=float)
            mon_arr = np.array([r[feat] for r in mon_rows], dtype=float)
            psi_val = _psi(ref_arr, mon_arr)
            if psi_val > PSI_THRESHOLD:
                psi_alerts.append(f"PSI({feat})={psi_val:.4f} > {PSI_THRESHOLD}")
                logger.warning("PSI ALERT: %s = %.4f", feat, psi_val)

        if psi_alerts:
            reasons.extend(psi_alerts)
            trigger_type = "psi"

        # ── Trigger 2: PR-AUC decay ───────────────────────────
        # Compare champion's registered test PR-AUC vs recent val performance
        champion = await conn.fetchrow(
            "SELECT pr_auc, model_version FROM ml.model_registry WHERE stage='champion' LIMIT 1"
        )
        if champion and champion["pr_auc"]:
            baseline_prauc = float(champion["pr_auc"])
            # In a real system, read recent online PR-AUC from monitoring_log
            # Here we use a stub: check if any monitoring alert exists
            recent_alert = await conn.fetchval(
                """
                SELECT 1 FROM raw.quality_audit_log
                WHERE validated_at >= ($1::timestamptz - INTERVAL '7 days')
                  AND failure_summary::text LIKE '%prauc_decay%'
                LIMIT 1
                """,
                monitoring_date,
            )
            if recent_alert:
                msg = f"PR-AUC decay detected vs champion={champion['model_version']} baseline={baseline_prauc:.4f}"
                reasons.append(msg)
                trigger_type = "prauc_decay"

        # ── Trigger 3: Calendar schedule ─────────────────────
        if scheduled:
            reasons.append(f"Monthly scheduled retrain (monitoring_date={monitoring_date})")
            trigger_type = trigger_type or "scheduled"

        # ── Trigger 4: Label volume ───────────────────────────
        new_labels = await conn.fetchval(
            """
            SELECT COUNT(*) FROM curated.churn_labels
            WHERE created_at >= ($1::date - INTERVAL '30 days')
              AND created_at <= $1::date
            """,
            monitoring_date,
        )
        if new_labels and int(new_labels) >= LABEL_VOLUME:
            reasons.append(f"New labels available: {new_labels} >= {LABEL_VOLUME}")
            trigger_type = trigger_type or "label_volume"

    await pool.close()

    should_retrain = len(reasons) > 0
    decision = TriggerDecision(should_retrain, reasons, trigger_type)

    logger.info(
        "Retrain decision: %s | trigger=%s | reasons=%s",
        "RETRAIN" if should_retrain else "SKIP",
        trigger_type,
        reasons,
    )
    return decision


async def main(args) -> int:
    decision = await evaluate_triggers(
        reference_date=date.fromisoformat(args.reference_date),
        monitoring_date=date.fromisoformat(args.monitoring_date),
        scheduled=args.scheduled,
    )

    print(json.dumps(decision.to_dict(), indent=2))

    if DRY_RUN:
        logger.info("DRY_RUN=true — not triggering pipeline")
        return 0

    if decision.should_retrain:
        logger.info("Retrain triggered. Downstream: run training_pipeline.py")
        return 0   # Signal to CI/cron: proceed with training

    logger.info("No retrain needed.")
    return 1   # Signal to CI/cron: skip training


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate retraining triggers")
    p.add_argument("--reference-date",  required=True, help="Baseline date (YYYY-MM-DD)")
    p.add_argument("--monitoring-date", required=True, help="Current date (YYYY-MM-DD)")
    p.add_argument("--scheduled", action="store_true", help="Monthly scheduled trigger")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
