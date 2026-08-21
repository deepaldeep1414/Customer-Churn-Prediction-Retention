"""
ml/features/feature_store.py
Populates ml.feature_snapshots from curated tables using
point-in-time safe SQL (feature_query.sql).

Usage:
    python feature_store.py \
        --observation-date 2024-01-01 \
        --horizon-days 30 \
        --feature-version v1.0.0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import date
from pathlib import Path
from uuid import UUID

import asyncpg
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

DATABASE_URL   = os.getenv("DATABASE_URL")
FEATURE_SQL    = Path(__file__).parent / "feature_query.sql"
FEATURE_CONFIG = Path(__file__).parent / "feature_config.yaml"

BATCH_SIZE = 500   # customers per batch


async def build_snapshots(
    observation_date: date,
    prediction_horizon_days: int,
    feature_set_version: str,
    customer_ids: list[UUID] | None = None,
) -> tuple[int, int]:
    """
    Build feature snapshots for all active customers (or a subset).
    Returns (inserted, skipped) counts.
    """
    sql = FEATURE_SQL.read_text()
    config = yaml.safe_load(FEATURE_CONFIG.read_text())

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=4, max_size=16)
    inserted = skipped = 0

    async with pool.acquire() as conn:
        # Fetch active customer IDs if not provided
        if customer_ids is None:
            rows = await conn.fetch(
                """
                SELECT DISTINCT c.customer_id
                FROM curated.customers c
                JOIN curated.subscriptions s ON s.customer_id = c.customer_id
                WHERE s.start_date <= $1
                  AND (s.end_date IS NULL OR s.end_date > $1)
                ORDER BY c.customer_id
                """,
                observation_date,
            )
            customer_ids = [r["customer_id"] for r in rows]

    logger.info(
        "Building snapshots for %d customers | obs=%s | horizon=%dd | version=%s",
        len(customer_ids), observation_date, prediction_horizon_days, feature_set_version,
    )

    # Process in batches
    for batch_start in range(0, len(customer_ids), BATCH_SIZE):
        batch = customer_ids[batch_start : batch_start + BATCH_SIZE]

        async with pool.acquire() as conn:
            for cid in batch:
                try:
                    row = await conn.fetchrow(sql, observation_date, cid)
                    if row is None:
                        skipped += 1
                        continue

                    await conn.execute(
                        """
                        INSERT INTO ml.feature_snapshots (
                            customer_id, observation_date, prediction_horizon_days,
                            feature_set_version,
                            tenure_months, monthly_charges, total_charges,
                            avg_monthly_charges_3m, charge_change_pct_3m, days_overdue_last,
                            contract_type, payment_method, paperless_billing,
                            phone_service, internet_service, addon_count,
                            tickets_30d, tickets_90d, tickets_total,
                            days_since_last_ticket, pct_billing_tickets_90d,
                            pct_cancel_tickets_90d, avg_csat_90d,
                            charge_x_tenure, overdue_x_cancel_intent
                        )
                        VALUES (
                            $1, $2, $3, $4,
                            $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16,
                            $17, $18, $19, $20, $21, $22, $23, $24, $25
                        )
                        ON CONFLICT (customer_id, observation_date, prediction_horizon_days, feature_set_version)
                        DO NOTHING
                        """,
                        cid, observation_date, prediction_horizon_days, feature_set_version,
                        row["tenure_months"],
                        row["monthly_charges"],
                        row["total_charges"],
                        row["avg_monthly_charges_3m"],
                        row["charge_change_pct_3m"],
                        row["days_overdue_last"],
                        row["contract_type"],
                        row["payment_method"],
                        row["paperless_billing"],
                        row["phone_service"],
                        row["internet_service"],
                        row["addon_count"],
                        row["tickets_30d"],
                        row["tickets_90d"],
                        row["tickets_total"],
                        row["days_since_last_ticket"],
                        row["pct_billing_tickets_90d"],
                        row["pct_cancel_tickets_90d"],
                        row["avg_csat_90d"],
                        row["charge_x_tenure"],
                        row["overdue_x_cancel_intent"],
                    )
                    inserted += 1

                except Exception as exc:
                    logger.warning("Failed for customer %s: %s", cid, exc)
                    skipped += 1

        logger.info(
            "Batch %d/%d done | inserted=%d skipped=%d",
            batch_start // BATCH_SIZE + 1,
            (len(customer_ids) + BATCH_SIZE - 1) // BATCH_SIZE,
            inserted, skipped,
        )

    await pool.close()
    logger.info("Feature snapshot build complete | inserted=%d skipped=%d", inserted, skipped)
    return inserted, skipped


def parse_args():
    parser = argparse.ArgumentParser(description="Build ML feature snapshots")
    parser.add_argument("--observation-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--feature-version", default="v1.0.0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        build_snapshots(
            observation_date=date.fromisoformat(args.observation_date),
            prediction_horizon_days=args.horizon_days,
            feature_set_version=args.feature_version,
        )
    )
