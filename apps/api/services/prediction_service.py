"""
apps/api/services/prediction_service.py
Orchestrates: feature retrieval → model inference → risk band → SHAP drivers
→ persistence to ml.predictions.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg
import pandas as pd

from schemas.response import PredictionResponse, RiskDriver
from services.feature_service import FeatureService, FEATURE_COLUMNS
from services.model_loader import ModelLoader

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

# Risk band boundaries (configurable via env)
BAND_LOW  = float(os.getenv("RISK_BAND_LOW",  "0.30"))
BAND_MED  = float(os.getenv("RISK_BAND_MED",  "0.55"))
BAND_HIGH = float(os.getenv("RISK_BAND_HIGH", "0.75"))

# Business-friendly feature labels (from explainability plan)
FEATURE_LABELS: dict[str, str] = {
    "tenure_months":            "Customer Tenure (months)",
    "monthly_charges":          "Current Monthly Charge ($)",
    "total_charges":            "Lifetime Spend ($)",
    "avg_monthly_charges_3m":   "Avg Monthly Charge (last 3 mo.)",
    "charge_change_pct_3m":     "Charge Change % (3 mo.)",
    "days_overdue_last":        "Days Overdue on Last Bill",
    "contract_type":            "Contract Type",
    "payment_method":           "Payment Method",
    "paperless_billing":        "Paperless Billing",
    "phone_service":            "Has Phone Service",
    "internet_service":         "Internet Service Type",
    "addon_count":              "Number of Add-On Services",
    "tickets_30d":              "Support Tickets (last 30 days)",
    "tickets_90d":              "Support Tickets (last 90 days)",
    "tickets_total":            "Total Support Tickets (all time)",
    "days_since_last_ticket":   "Days Since Last Support Contact",
    "pct_billing_tickets_90d":  "% Billing Tickets (last 90 days)",
    "pct_cancel_tickets_90d":   "% Cancellation-Intent Tickets (90 days)",
    "avg_csat_90d":             "Avg Satisfaction Score (last 90 days)",
    "charge_x_tenure":          "Charge × Tenure (interaction)",
    "overdue_x_cancel_intent":  "Overdue Bill × Cancellation Intent (interaction)",
}


def _risk_band(prob: float) -> str:
    if prob < BAND_LOW:
        return "low"
    elif prob < BAND_MED:
        return "medium"
    elif prob < BAND_HIGH:
        return "high"
    return "critical"


class PredictionService:
    """End-to-end scoring: features → inference → SHAP → persist."""

    def __init__(self, model_loader: ModelLoader) -> None:
        self._loader  = model_loader
        self._features = FeatureService()
        self._pool: asyncpg.Pool | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
        return self._pool

    async def score_single(
        self,
        customer_id: UUID,
        observation_date: date,
        prediction_horizon_days: int,
    ) -> PredictionResponse | None:
        """Score one customer. Returns None if customer does not exist."""

        feature_dict = await self._features.get_features(
            customer_id=customer_id,
            observation_date=observation_date,
            prediction_horizon_days=prediction_horizon_days,
            feature_set_version=self._loader.feature_set_version,
        )
        if feature_dict is None:
            return None

        # ── Inference ──────────────────────────────────────────
        X = pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)
        model = self._loader.model
        raw_probs = model.predict(X)
        # pyfunc may return 1-D or 2-D array depending on flavour
        if hasattr(raw_probs, "ndim") and raw_probs.ndim == 2:
            churn_prob = float(raw_probs[0][1])
        else:
            churn_prob = float(raw_probs[0])

        band = _risk_band(churn_prob)

        # ── SHAP drivers (best-effort; cached in snapshot) ─────
        top_drivers = await self._get_cached_drivers(
            customer_id, observation_date, self._loader.model_version
        )
        if not top_drivers:
            top_drivers = self._compute_shap_drivers(model, X, feature_dict)

        # ── Persist prediction ─────────────────────────────────
        scored_at = datetime.now(timezone.utc)
        await self._persist(
            customer_id=customer_id,
            observation_date=observation_date,
            churn_prob=churn_prob,
            band=band,
            model_version=self._loader.model_version,
            top_drivers=top_drivers,
            scored_at=scored_at,
            source="api",
        )

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(churn_prob, 5),
            risk_band=band,
            model_version=self._loader.model_version or "unknown",
            feature_set_version=self._loader.feature_set_version,
            observation_date=observation_date,
            prediction_horizon_days=prediction_horizon_days,
            scored_at=scored_at,
            top_drivers=[RiskDriver(**d) for d in top_drivers[:5]],
        )

    def _compute_shap_drivers(
        self, model: Any, X: pd.DataFrame, feature_dict: dict
    ) -> list[dict]:
        """
        Compute SHAP values using ChurnExplainer, which correctly handles:
          - sklearn Pipeline wrapping
          - CalibratedClassifierCV base estimator extraction
          - One-hot SHAP aggregation back to original feature names
          - model_version + explained_at traceability fields

        Falls back to empty list on any error (graceful degradation).
        """
        try:
            import sys
            sys.path.insert(0, "/app/ml")
            from explainability.shap_explainer import ChurnExplainer

            algorithm = "unknown"
            meta = getattr(self._loader, "model_metadata", None)
            if meta and isinstance(meta, dict):
                algorithm = meta.get("algorithm", "unknown")

            explainer = ChurnExplainer(
                pipeline=model,
                algorithm=algorithm,
                X_background=X,  # single-row background — acceptable for API path
            )
            drivers = explainer.explain_single(
                X_row=X,
                customer_raw_values=feature_dict,
                top_n=5,
            )
            # Inject provenance for traceability (attribution, not causation)
            explained_at = datetime.now(timezone.utc).isoformat()
            model_ver    = self._loader.model_version or "unknown"
            for d in drivers:
                d["model_version"] = model_ver
                d["explained_at"]  = explained_at
            return drivers

        except Exception as exc:
            logger.warning(
                "SHAP computation failed (returning empty drivers): %s", exc,
                exc_info=True,
            )
            return []

    async def _get_cached_drivers(
        self, customer_id: UUID, observation_date: date, model_version: str | None
    ) -> list[dict]:
        """Read pre-computed top_drivers from ml.predictions (batch-scored path)."""
        if not model_version:
            return []
        try:
            pool = await self._get_pool()
            row = await pool.fetchrow(
                """
                SELECT top_drivers
                FROM ml.predictions
                WHERE customer_id    = $1
                  AND observation_date = $2
                  AND model_version  = $3
                ORDER BY scored_at DESC
                LIMIT 1
                """,
                customer_id, observation_date, model_version,
            )
            if row and row["top_drivers"]:
                return json.loads(row["top_drivers"])
        except Exception as exc:
            logger.warning("Failed to read cached drivers: %s", exc)
        return []

    async def _persist(
        self,
        customer_id: UUID,
        observation_date: date,
        churn_prob: float,
        band: str,
        model_version: str | None,
        top_drivers: list[dict],
        scored_at: datetime,
        source: str,
    ) -> None:
        try:
            pool = await self._get_pool()
            await pool.execute(
                """
                INSERT INTO ml.predictions
                    (customer_id, model_version, observation_date, churn_probability,
                     risk_band, threshold_used, scored_at, top_drivers, request_source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9)
                """,
                customer_id,
                model_version or "unknown",
                observation_date,
                round(churn_prob, 5),
                band,
                self._loader.threshold,
                scored_at,
                json.dumps(top_drivers[:5]),
                source,
            )
        except Exception as exc:
            logger.warning("Failed to persist prediction: %s", exc)
