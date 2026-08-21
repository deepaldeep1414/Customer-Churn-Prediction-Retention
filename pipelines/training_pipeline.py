"""
pipelines/training_pipeline.py
End-to-end training orchestrator.

Runs in sequence:
  1. Validate data quality (quarantine check)
  2. Build feature snapshots (if not pre-built)
  3. Generate labels (if not pre-built)
  4. Load train/val/test splits
  5. Train all 4 algorithms with Optuna tuning
  6. Select champion (best val PR-AUC with calibration)
  7. Evaluate on test set (ONCE, after champion selection)
  8. Generate SHAP summary plots
  9. Run promotion gates
 10. Register in MLflow (Staging) and ml.model_registry

Usage:
    python training_pipeline.py \
        --observation-date 2023-12-31 \
        --horizon-days 30 \
        --feature-version v1.0.0 \
        --run-tuning
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
from datetime import date
from pathlib import Path
from uuid import uuid4

import asyncpg
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd

# Adjust PYTHONPATH so sibling ml/ modules resolve
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml" / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml" / "evaluation"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ml" / "explainability"))

from training.splitter import temporal_split
from training.trainer import train, _build_preprocessor, _build_model
from training.tuner import tune
from evaluation.evaluator import evaluate_model, check_promotion_gates, EvalResults
from evaluation.calibration import calibrate_model
from evaluation.threshold_selector import select_threshold
from explainability.shap_explainer import ChurnExplainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATABASE_URL        = os.getenv("DATABASE_URL")
MLFLOW_URI          = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME          = os.getenv("MODEL_NAME", "churn_predictor")
RETENTION_CAPACITY  = int(os.getenv("RETENTION_CAPACITY", "200"))

ALGORITHMS = ["logistic_regression", "xgboost", "lightgbm", "catboost"]

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


async def _load_dataset(
    observation_date: date,
    horizon_days: int,
    feature_version: str,
) -> pd.DataFrame:
    """Load merged feature_snapshots + churn_labels from DB."""
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                fs.*,
                cl.churned,
                cl.churn_date
            FROM ml.feature_snapshots fs
            JOIN curated.churn_labels cl
              ON cl.customer_id      = fs.customer_id
             AND cl.observation_date = fs.observation_date
             AND cl.prediction_horizon = fs.prediction_horizon_days
            WHERE fs.feature_set_version     = $1
              AND fs.observation_date        <= $2
              AND fs.prediction_horizon_days  = $3
            ORDER BY fs.observation_date
            """,
            feature_version, observation_date, horizon_days,
        )
    await pool.close()
    df = pd.DataFrame([dict(r) for r in rows])
    logger.info("Dataset loaded: %d rows | churn_rate=%.3f", len(df), df["churned"].mean())
    return df


async def _check_leakage(
    conn: asyncpg.Connection,
    feature_version: str,
    observation_date: date,
) -> int:
    """Call the DB leakage audit function. Returns count of leaking rows."""
    rows = await conn.fetch(
        "SELECT * FROM ml.validate_snapshot_leakage($1, $2)",
        feature_version, observation_date,
    )
    return len(rows)


async def _register_in_db(
    run_id: str,
    version: str,
    algorithm: str,
    val_pr_auc: float,
    val_recall: float,
    threshold: float,
    feature_version: str,
    train_hash: str = "unknown",
    val_hash: str = "unknown",
    test_hash: str = "unknown",
) -> None:
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ml.model_registry
                (model_name, model_version, algorithm, feature_set_version,
                 mlflow_run_id, mlflow_model_uri, stage,
                 train_dataset_hash, val_dataset_hash, test_dataset_hash,
                 pr_auc, recall_at_threshold, threshold)
            VALUES ($1, $2, $3, $4, $5, $6, 'staging', $7, $8, $9, $10, $11, $12)
            ON CONFLICT (model_version) DO UPDATE SET
                stage = 'staging', pr_auc = EXCLUDED.pr_auc,
                mlflow_run_id = EXCLUDED.mlflow_run_id,
                mlflow_model_uri = EXCLUDED.mlflow_model_uri
            """,
            MODEL_NAME, version, algorithm, feature_version,
            run_id, f"runs:/{run_id}/model", train_hash, val_hash, test_hash,
            val_pr_auc, val_recall, threshold,
        )
    await pool.close()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


async def run_training_pipeline(
    observation_date: date,
    horizon_days: int,
    feature_version: str,
    run_tuning: bool = False,
) -> dict:
    """
    Full training pipeline. Returns summary dict with best run_id and metrics.
    """
    job_id  = str(uuid4())
    git_sha = _git_sha()
    logger.info(json.dumps({
        "event":            "training_pipeline_start",
        "job_id":           job_id,
        "observation_date": observation_date.isoformat(),
        "feature_version":  feature_version,
        "git_commit":       git_sha,
    }))

    # ── Step 1: Leakage audit ─────────────────────────────────
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=8)
    async with pool.acquire() as conn:
        leaking_count = await _check_leakage(conn, feature_version, observation_date)
    await pool.close()

    if leaking_count > 0:
        raise RuntimeError(
            f"TRAINING ABORTED: {leaking_count} feature snapshot rows have label leakage. "
            "Fix the feature SQL before training."
        )
    logger.info("Leakage audit passed: 0 leaking rows.")

    # ── Step 2: Load dataset ──────────────────────────────────
    min_rows = int(os.getenv("MIN_DATASET_ROWS", "30"))
    df = await _load_dataset(observation_date, horizon_days, feature_version)
    if len(df) < min_rows:
        raise ValueError(
            f"Dataset has only {len(df)} rows — minimum required is {min_rows}. "
            "Expand observation dates or check ingestion."
        )

    # ── Step 3: Temporal split ────────────────────────────────
    splits = temporal_split(df, label_col="churned")
    logger.info(
        "Splits | train=%d val=%d test=%d",
        len(splits.X_train), len(splits.X_val), len(splits.X_test),
    )

    # ── Step 4: Train all algorithms ──────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("churn_prediction")

    run_results: list[dict] = []

    for algorithm in ALGORITHMS:
        logger.info("Training: %s", algorithm)
        config_path = Path(__file__).parent.parent / "ml" / "training" / "configs" / f"{algorithm}.yaml"

        if run_tuning:
            from training.trainer import _build_preprocessor, _build_model
            from sklearn.pipeline import Pipeline
            import yaml

            def build_fn(algo, params, y_tr):
                pp = _build_preprocessor()
                m  = _build_model(algo, params, splits)
                p  = Pipeline([("preprocessor", pp), ("model", m)])
                p.fit(splits.X_train, splits.y_train)
                return p

            from training.tuner import tune
            import yaml
            best_params = tune(algorithm, splits, build_fn)
        else:
            import yaml
            with open(config_path) as cf:
                best_params = yaml.safe_load(cf)

        try:
            run_id = train(
                splits=splits,
                algorithm=algorithm,
                config=best_params,
                feature_set_version=feature_version,
                retention_capacity=RETENTION_CAPACITY,
            )
            run     = mlflow.get_run(run_id)
            metrics = run.data.metrics
            run_results.append({
                "run_id":     run_id,
                "algorithm":  algorithm,
                "val_pr_auc": metrics.get("val_pr_auc", 0.0),
                "val_ece":    metrics.get("val_cal_ece"),
                "threshold":  float(run.data.params.get("frozen_threshold", 0.5)),
            })
            logger.info("Trained %s: val_pr_auc=%.4f", algorithm, metrics.get("val_pr_auc", 0))
        except Exception as exc:
            logger.error("Training FAILED for %s: %s", algorithm, exc)

    if not run_results:
        raise RuntimeError("All algorithm training runs failed.")

    # ── Step 5: Champion selection (best val PR-AUC on val set) ──
    best = max(run_results, key=lambda r: r["val_pr_auc"])
    logger.info(
        "Champion selected: algorithm=%s val_pr_auc=%.4f run_id=%s",
        best["algorithm"], best["val_pr_auc"], best["run_id"],
    )

    # ── Step 6: Test set evaluation (ONCE, after selection) ───
    best_run    = mlflow.get_run(best["run_id"])
    model_uri   = f"runs:/{best['run_id']}/model"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    y_prob_test = loaded_model.predict(splits.X_test)
    if hasattr(y_prob_test, "ndim") and y_prob_test.ndim == 2:
        import numpy as np
        y_prob_test = y_prob_test[:, 1]

    test_res = evaluate_model(
        splits.y_test, y_prob_test,
        threshold=best["threshold"], prefix="test",
    )
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics(test_res.as_dict())
    logger.info("Test set evaluation: %s", test_res.summary())

    # ── Step 7: SHAP summary (test set sample) ────────────────
    try:
        loaded_sklearn = mlflow.sklearn.load_model(model_uri)
        X_background   = splits.X_train.sample(min(200, len(splits.X_train)), random_state=42)
        explainer      = ChurnExplainer(loaded_sklearn, best["algorithm"], X_background)
        shap_plot_path = Path("/tmp/shap_summary.png")
        explainer.save_summary_plot(splits.X_test.head(500), shap_plot_path)
        with mlflow.start_run(run_id=best["run_id"]):
            mlflow.log_artifact(str(shap_plot_path), "explainability")
        logger.info("SHAP summary plot logged to MLflow")
    except Exception as exc:
        logger.warning("SHAP plot generation failed (non-fatal): %s", exc)

    # ── Step 8: Register best model in DB ─────────────────────
    candidate_version = f"{observation_date.isoformat().replace('-', '')}.{best['algorithm']}"
    await _register_in_db(
        run_id=best["run_id"],
        version=candidate_version,
        algorithm=best["algorithm"],
        val_pr_auc=best["val_pr_auc"],
        val_recall=float(best_run.data.metrics.get("val_thresh_recall", 0.0)),
        threshold=best["threshold"],
        feature_version=feature_version,
        train_hash=splits.train_dataset_hash,
        val_hash=splits.val_dataset_hash,
        test_hash=splits.test_dataset_hash,
    )

    summary = {
        "job_id":               job_id,
        "best_run_id":          best["run_id"],
        "best_algorithm":       best["algorithm"],
        "candidate_version":    candidate_version,
        "val_pr_auc":           round(best["val_pr_auc"], 4),
        "test_pr_auc":          round(test_res.pr_auc, 4),
        "test_recall":          round(test_res.recall, 4),
        "test_precision":       round(test_res.precision, 4),
        "all_run_results":      run_results,
    }
    logger.info(json.dumps({"event": "training_pipeline_complete", **summary}))
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="End-to-end training pipeline")
    p.add_argument("--observation-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--horizon-days",     type=int, default=30)
    p.add_argument("--feature-version",  default="v1.0.0")
    p.add_argument("--run-tuning",       action="store_true",
                   help="Run Optuna hyperparameter tuning (slow)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = asyncio.run(
        run_training_pipeline(
            observation_date=date.fromisoformat(args.observation_date),
            horizon_days=args.horizon_days,
            feature_version=args.feature_version,
            run_tuning=args.run_tuning,
        )
    )
    print(json.dumps(result, indent=2))
