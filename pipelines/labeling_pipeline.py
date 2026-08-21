"""
pipelines/labeling_pipeline.py
Generate point-in-time churn labels for a given observation_date and horizon.

Label definition:
  churned = TRUE  iff the customer's subscription ended within
            (observation_date, observation_date + prediction_horizon_days]
  churned = FALSE iff the subscription was still active at
            observation_date + prediction_horizon_days

Run for each training observation_date in your dataset.

Usage:
    python labeling_pipeline.py \
        --observation-date 2023-06-30 \
        --horizon-days 30 \
        --label-version 1
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import date, timedelta

import asyncpg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL = os.getenv("DATABASE_URL")


async def generate_labels(
    observation_date: date,
    horizon_days: int,
    label_version: int,
    labeling_method: str = "subscription_end",
) -> tuple[int, int]:
    """
    Generate churn labels for all customers active at observation_date.

    Returns (churned_count, not_churned_count).
    """
    window_end = observation_date + timedelta(days=horizon_days)
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)

    async with pool.acquire() as conn:
        # Find all customers with an active subscription at observation_date
        active_customers = await conn.fetch(
            """
            SELECT DISTINCT c.customer_id
            FROM curated.customers c
            JOIN curated.subscriptions s ON s.customer_id = c.customer_id
            WHERE s.start_date <= $1
              AND (s.end_date IS NULL OR s.end_date > $1)
            """,
            observation_date,
        )

        churned = not_churned = 0

        async with conn.transaction():
            for row in active_customers:
                cid = row["customer_id"]

                # Check: did ANY subscription for this customer end in (obs_date, window_end]?
                churn_event = await conn.fetchrow(
                    """
                    SELECT end_date
                    FROM curated.subscriptions
                    WHERE customer_id = $1
                      AND end_date >  $2          -- strictly after observation_date
                      AND end_date <= $3          -- within prediction window
                    ORDER BY end_date ASC
                    LIMIT 1
                    """,
                    cid, observation_date, window_end,
                )

                did_churn = churn_event is not None
                churn_date = churn_event["end_date"] if did_churn else None

                await conn.execute(
                    """
                    INSERT INTO curated.churn_labels
                        (customer_id, observation_date, prediction_horizon,
                         churned, churn_date, labeling_method, label_version)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (customer_id, observation_date, prediction_horizon, label_version)
                    DO NOTHING
                    """,
                    cid, observation_date, horizon_days,
                    did_churn, churn_date, labeling_method, label_version,
                )
                if did_churn:
                    churned += 1
                else:
                    not_churned += 1

    await pool.close()

    total = churned + not_churned
    churn_rate = churned / max(total, 1)
    logger.info(
        "Labels generated | obs=%s | horizon=%dd | total=%d | churned=%d (%.1f%%)",
        observation_date, horizon_days, total, churned, churn_rate * 100,
    )
    return churned, not_churned


def parse_args():
    p = argparse.ArgumentParser(description="Generate churn labels")
    p.add_argument("--observation-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--horizon-days", type=int, default=30)
    p.add_argument("--label-version", type=int, default=1)
    p.add_argument("--labeling-method", default="subscription_end")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        generate_labels(
            observation_date=date.fromisoformat(args.observation_date),
            horizon_days=args.horizon_days,
            label_version=args.label_version,
            labeling_method=args.labeling_method,
        )
    )
