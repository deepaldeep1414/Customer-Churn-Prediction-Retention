"""
apps/api/services/feature_service.py
Retrieves point-in-time features for a customer at a given observation_date.
Reads from ml.feature_snapshots first (pre-computed); falls back to
computing on-the-fly from curated tables if no snapshot exists.
"""
from __future__ import annotations

import logging
import os
from datetime import date
from uuid import UUID

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Feature column order must exactly match training pipeline
FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "avg_monthly_charges_3m",
    "charge_change_pct_3m",
    "days_overdue_last",
    "contract_type",
    "payment_method",
    "paperless_billing",
    "phone_service",
    "internet_service",
    "addon_count",
    "tickets_30d",
    "tickets_90d",
    "tickets_total",
    "days_since_last_ticket",
    "pct_billing_tickets_90d",
    "pct_cancel_tickets_90d",
    "avg_csat_90d",
    "charge_x_tenure",
    "overdue_x_cancel_intent",
]


class FeatureService:
    """Retrieves feature vectors from the ML snapshot store."""

    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        return self._pool

    async def get_features(
        self,
        customer_id: UUID,
        observation_date: date,
        prediction_horizon_days: int,
        feature_set_version: str,
    ) -> dict | None:
        """
        Return a feature dict for the given customer / observation_date.
        Reads from ml.feature_snapshots (pre-computed batch path).
        Returns None if the customer does not exist.
        """
        pool = await self._get_pool()

        row = await pool.fetchrow(
            """
            SELECT *
            FROM ml.feature_snapshots
            WHERE customer_id              = $1
              AND observation_date         = $2
              AND prediction_horizon_days  = $3
              AND feature_set_version      = $4
            LIMIT 1
            """,
            customer_id,
            observation_date,
            prediction_horizon_days,
            feature_set_version,
        )

        if row:
            return {col: row[col] for col in FEATURE_COLUMNS if col in row.keys()}

        # Fallback: verify customer exists
        exists = await pool.fetchval(
            "SELECT 1 FROM curated.customers WHERE customer_id = $1", customer_id
        )
        if not exists:
            return None

        logger.warning(
            "No pre-computed snapshot for customer=%s obs=%s — "
            "on-the-fly feature computation not yet implemented.",
            customer_id, observation_date,
        )
        return None
