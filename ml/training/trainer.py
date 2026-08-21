"""
ml/training/trainer.py
End-to-end model training with MLflow logging.

Usage:
    python trainer.py --algorithm xgboost --config configs/xgboost.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from splitter import DataSplits
from evaluation.evaluator import evaluate_model, EvalResults
from evaluation.calibration import calibrate_model
from evaluation.threshold_selector import select_threshold

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

MLFLOW_URI         = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME    = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn_prediction")
FEATURE_CONFIG     = Path(__file__).parent.parent / "features" / "feature_config.yaml"

CATEGORICAL_FEATURES = ["contract_type", "payment_method", "internet_service"]
BOOLEAN_FEATURES     = ["paperless_billing", "phone_service"]
NUMERIC_FEATURES     = [
    "tenure_months", "monthly_charges", "total_charges",
    "avg_monthly_charges_3m", "charge_change_pct_3m", "days_overdue_last",
    "addon_count", "tickets_30d", "tickets_90d", "tickets_total",
    "days_since_last_ticket", "pct_billing_tickets_90d", "pct_cancel_tickets_90d",
    "avg_csat_90d", "charge_x_tenure", "overdue_x_cancel_intent",
]

# Retention team capacity for threshold selection (customers/month)
RETENTION_CAPACITY = int(os.getenv("RETENTION_CAPACITY", "200"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


from sklearn.impute import SimpleImputer


def _build_preprocessor() -> ColumnTransformer:
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
    )


def _build_model(algorithm: str, params: dict, splits: DataSplits) -> Any:
    """Instantiate the raw estimator (no preprocessing pipeline yet)."""
    n_pos  = int(splits.y_train.sum())
    n_neg  = len(splits.y_train) - n_pos
    scale  = n_neg / max(n_pos, 1)

    if algorithm == "logistic_regression":
        p = {"class_weight": "balanced", "max_iter": 500, "C": 1.0, "solver": "lbfgs", "random_state": 42}
        p.update({k: v for k, v in params.items() if k != "algorithm"})
        return LogisticRegression(**p)

    elif algorithm == "xgboost":
        from xgboost import XGBClassifier
        p = {"scale_pos_weight": scale, "eval_metric": "aucpr", "random_state": 42}
        p.update({k: v for k, v in params.items() if k != "algorithm"})
        return XGBClassifier(**p)

    elif algorithm == "lightgbm":
        from lightgbm import LGBMClassifier
        p = {"is_unbalance": True, "random_state": 42, "verbose": -1}
        p.update({k: v for k, v in params.items() if k != "algorithm"})
        return LGBMClassifier(**p)

    elif algorithm == "catboost":
        from catboost import CatBoostClassifier
        p = {"auto_class_weights": "Balanced", "random_seed": 42, "verbose": 0}
        p.update({k: v for k, v in params.items() if k != "algorithm"})
        return CatBoostClassifier(**p)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def train(
    splits: DataSplits,
    algorithm: str,
    config: dict,
    feature_set_version: str,
    retention_capacity: int = RETENTION_CAPACITY,
) -> str:
    """
    Full training run. Returns the MLflow run_id.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    git_sha = _git_sha()
    fc = yaml.safe_load(FEATURE_CONFIG.read_text())

    with mlflow.start_run(run_name=f"{algorithm}_{feature_set_version}_{git_sha[:8]}") as run:

        # ── Log parameters ────────────────────────────────────
        mlflow.log_params({
            "algorithm":              algorithm,
            "feature_set_version":    feature_set_version,
            "train_cutoff":           splits.train_cutoff.isoformat(),
            "val_cutoff":             splits.val_cutoff.isoformat(),
            "test_cutoff":            splits.test_cutoff.isoformat(),
            "prediction_horizon_days": 30,
            "retention_capacity":     retention_capacity,
            "git_commit":             git_sha,
            "train_dataset_hash":     splits.train_dataset_hash,
            "val_dataset_hash":       splits.val_dataset_hash,
            "test_dataset_hash":      splits.test_dataset_hash,
            **config,
        })

        # ── Build pipeline ────────────────────────────────────
        preprocessor = _build_preprocessor()
        base_model   = _build_model(algorithm, config, splits)
        pipeline     = Pipeline([
            ("preprocessor", preprocessor),
            ("model",        base_model),
        ])

        # ── Train ─────────────────────────────────────────────
        pipeline.fit(splits.X_train, splits.y_train)

        # ── Val evaluation ────────────────────────────────────
        y_prob_val = pipeline.predict_proba(splits.X_val)[:, 1]
        val_res    = evaluate_model(splits.y_val, y_prob_val, threshold=0.5, prefix="val")
        mlflow.log_metrics(val_res.as_dict())

        # ── Threshold selection (val set, frozen) ─────────────
        threshold = select_threshold(splits.y_val, y_prob_val, retention_capacity)
        val_res_thresh = evaluate_model(splits.y_val, y_prob_val, threshold=threshold, prefix="val_thresh")
        mlflow.log_metrics(val_res_thresh.as_dict())
        mlflow.log_param("frozen_threshold", threshold)
        logger.info("Frozen threshold: %.4f | recall=%.4f | precision=%.4f",
                    threshold, val_res_thresh.recall, val_res_thresh.precision)

        # ── Calibration (fit on val_cal, evaluate on held-out val_check) ─
        # IMPORTANT: calibrator must be evaluated on data it was NOT fitted on.
        # Split val: 80% for calibration fitting, 20% as hold-out check.
        from sklearn.model_selection import train_test_split

        # Require min 200 val samples for a meaningful hold-out
        if len(splits.X_val) >= 200:
            X_val_cal, X_val_check, y_val_cal, y_val_check = train_test_split(
                splits.X_val, splits.y_val,
                test_size=0.20,
                random_state=42,
                stratify=splits.y_val,
            )
        else:
            logger.warning(
                "Val set too small (%d rows) for cal/check split — "
                "calibrating on full val set; ECE may be optimistic.",
                len(splits.X_val),
            )
            X_val_cal, X_val_check = splits.X_val, splits.X_val
            y_val_cal, y_val_check = splits.y_val, splits.y_val

        calibrated_pipeline = calibrate_model(pipeline, X_val_cal, y_val_cal)

        # Evaluate on HELD-OUT check set (never seen by calibrator)
        y_prob_val_check = calibrated_pipeline.predict_proba(X_val_check)[:, 1]
        cal_res = evaluate_model(
            y_val_check, y_prob_val_check,
            threshold=threshold, prefix="val_cal",
        )
        mlflow.log_metrics(cal_res.as_dict())
        mlflow.log_param("n_val_calibration", len(X_val_cal))
        mlflow.log_param("n_val_cal_check",   len(X_val_check))
        logger.info(
            "Calibration (hold-out check) | ECE=%.4f | n_check=%d",
            cal_res.ece, len(X_val_check),
        )

        # ── Train PR-AUC and overfitting gap ─────────────────
        y_prob_train   = pipeline.predict_proba(splits.X_train)[:, 1]
        train_res      = evaluate_model(splits.y_train, y_prob_train, threshold=threshold, prefix="train")
        train_val_gap  = round(train_res.pr_auc - val_res.pr_auc, 5)
        mlflow.log_metric("train_pr_auc",      train_res.pr_auc)
        mlflow.log_metric("train_val_pr_auc_gap", train_val_gap)
        if train_val_gap > 0.10:
            logger.warning(
                "OVERFITTING ALERT: train_val_pr_auc_gap=%.4f > 0.10 | "
                "train=%.4f val=%.4f",
                train_val_gap, train_res.pr_auc, val_res.pr_auc,
            )

        # ── Log artifacts ─────────────────────────────────────
        mlflow.log_artifact(str(FEATURE_CONFIG))

        # Log the calibrated pipeline as the canonical model artifact
        if algorithm == "xgboost":
            mlflow.sklearn.log_model(calibrated_pipeline, "model")
        else:
            mlflow.sklearn.log_model(calibrated_pipeline, "model")

        logger.info(
            "Run %s | algorithm=%s | val_pr_auc=%.4f | val_thresh_recall=%.4f",
            run.info.run_id, algorithm, val_res.pr_auc, val_res_thresh.recall,
        )
        return run.info.run_id


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--algorithm", required=True,
                        choices=["logistic_regression", "xgboost", "lightgbm", "catboost"])
    parser.add_argument("--config", required=True, help="Path to hyperparameter YAML")
    parser.add_argument("--data-path", required=True, help="Path to merged feature+label parquet")
    parser.add_argument("--feature-version", default="v1.0.0")
    parser.add_argument("--retention-capacity", type=int, default=RETENTION_CAPACITY)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from splitter import temporal_split
    df = pd.read_parquet(args.data_path)
    splits = temporal_split(df)
    config = load_config(args.config)
    config["algorithm"] = args.algorithm
    run_id = train(
        splits=splits,
        algorithm=args.algorithm,
        config=config,
        feature_set_version=args.feature_version,
        retention_capacity=args.retention_capacity,
    )
    logger.info("Training complete. MLflow run_id: %s", run_id)
