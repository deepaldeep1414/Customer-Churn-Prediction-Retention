"""
apps/api/services/model_loader.py
Loads the champion model from MLflow on startup and caches it in memory.
Supports hot-reload when the champion model version changes.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.pyfunc

logger = logging.getLogger(__name__)

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME    = os.getenv("MODEL_NAME", "churn_predictor")
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")  # 'latest' resolves to champion


class ModelLoader:
    """
    Singleton model loader.

    On startup, queries ml.model_registry for the current champion version,
    then loads the artifact from MLflow. Exposes the model and metadata
    to the FastAPI request lifecycle.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._metadata: dict = {}
        self._loaded: bool = False

    async def load(self) -> None:
        """Load champion model from MLflow (called once at startup)."""
        mlflow.set_tracking_uri(MLFLOW_URI)

        try:
            run_id = None
            version_str = None
            model_uri = None

            # 1. Query PostgreSQL ml.model_registry for champion stage
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                try:
                    import asyncpg
                    conn = await asyncpg.connect(db_url)
                    row = await conn.fetchrow(
                        "SELECT mlflow_run_id, model_version, algorithm, threshold, feature_set_version "
                        "FROM ml.model_registry WHERE stage = 'champion' ORDER BY created_at DESC LIMIT 1"
                    )
                    await conn.close()
                    if row and row["mlflow_run_id"]:
                        run_id = row["mlflow_run_id"]
                        version_str = row["model_version"]
                        model_uri = f"runs:/{run_id}/model"
                except Exception as db_err:
                    logger.warning("Could not query DB model_registry: %s", db_err)

            # 2. Fallback to MLflow Model Registry Production stage
            if not model_uri:
                client = mlflow.tracking.MlflowClient()
                try:
                    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
                    if versions:
                        mv = versions[0]
                        model_uri = f"models:/{MODEL_NAME}/Production"
                        version_str = mv.version
                        run_id = mv.run_id
                except Exception:
                    pass

            if not model_uri:
                logger.warning("No champion model found. API will start in degraded mode.")
                return

            self._model = mlflow.pyfunc.load_model(model_uri)

            # Fetch run metadata for the model endpoint
            if run_id:
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(run_id)
                params = run.data.params
                metrics = run.data.metrics
                self._metadata = {
                    "model_name":             MODEL_NAME,
                    "model_version":          version_str,
                    "algorithm":              params.get("algorithm", "unknown"),
                    "stage":                  "champion",
                    "feature_set_version":    params.get("feature_set_version", "unknown"),
                    "pr_auc":                 metrics.get("test_pr_auc"),
                    "roc_auc":                metrics.get("test_roc_auc"),
                    "brier_score":            metrics.get("test_brier_score"),
                    "recall_at_threshold":    metrics.get("test_recall"),
                    "precision_at_threshold": metrics.get("test_precision"),
                    "f1_at_threshold":        metrics.get("test_f1"),
                    "threshold":              float(params.get("frozen_threshold", 0.5)),
                    "promoted_by":            params.get("promoted_by"),
                    "promoted_at":            None,
                }
            else:
                self._metadata = {
                    "model_name":          MODEL_NAME,
                    "model_version":       version_str,
                    "algorithm":           "unknown",
                    "stage":               "champion",
                    "feature_set_version": "unknown",
                }

            self._loaded = True
            logger.info("Loaded model '%s' version '%s'", MODEL_NAME, version_str)

        except Exception as exc:
            logger.error("Failed to load model from MLflow: %s", exc, exc_info=True)
            # Do not re-raise — allow API to start in degraded mode

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model(self) -> Any:
        return self._model

    @property
    def model_version(self) -> str | None:
        return self._metadata.get("model_version")

    @property
    def threshold(self) -> float:
        return float(self._metadata.get("threshold", 0.5))

    @property
    def feature_set_version(self) -> str:
        return self._metadata.get("feature_set_version", "unknown")

    @property
    def model_metadata(self) -> dict:
        return self._metadata
