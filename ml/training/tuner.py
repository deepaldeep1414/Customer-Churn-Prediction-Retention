"""
ml/training/tuner.py
Optuna hyperparameter search with temporal cross-validation.
All tuning is performed on train+val data ONLY; test set is never touched.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from splitter import DataSplits, temporal_split

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS   = int(os.getenv("OPTUNA_TRIALS", "50"))
N_CV_FOLDS = int(os.getenv("OPTUNA_CV_FOLDS", "5"))


def _suggest_params(trial: optuna.Trial, algorithm: str) -> dict:
    if algorithm == "xgboost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0, log=True),
        }
    elif algorithm == "lightgbm":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 300),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq":     trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
    elif algorithm == "catboost":
        return {
            "iterations":       trial.suggest_int("iterations", 200, 1000),
            "depth":            trial.suggest_int("depth", 4, 10),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "border_count":     trial.suggest_int("border_count", 32, 255),
            "random_strength":  trial.suggest_float("random_strength", 0.0, 10.0),
        }
    elif algorithm == "logistic_regression":
        return {
            "C":       trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver":  trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
            "max_iter": 500,
        }
    raise ValueError(f"Unknown algorithm: {algorithm}")


def objective(
    trial: optuna.Trial,
    algorithm: str,
    X_trainval: pd.DataFrame,
    y_trainval: pd.Series,
    build_pipeline_fn,
) -> float:
    """Temporal CV objective maximising PR-AUC on out-of-fold val folds."""
    params   = _suggest_params(trial, algorithm)
    tscv     = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    pr_aucs  = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_trainval)):
        X_fold_train = X_trainval.iloc[train_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        X_fold_val   = X_trainval.iloc[val_idx]
        y_fold_val   = y_trainval.iloc[val_idx]

        pipeline = build_pipeline_fn(algorithm, params, y_fold_train)
        try:
            pipeline.fit(X_fold_train, y_fold_train)
            y_prob = pipeline.predict_proba(X_fold_val)[:, 1]
            pr_auc = average_precision_score(y_fold_val, y_prob)
            pr_aucs.append(pr_auc)
        except Exception as exc:
            logger.debug("Trial %d fold %d failed: %s", trial.number, fold_idx, exc)
            return 0.0

    return float(np.mean(pr_aucs))


def tune(
    algorithm: str,
    splits: DataSplits,
    build_pipeline_fn,
    n_trials: int = N_TRIALS,
) -> dict:
    """
    Run Optuna hyperparameter search.

    Args:
        algorithm:         Model algorithm name.
        splits:            DataSplits (only train+val used).
        build_pipeline_fn: Callable(algorithm, params, y_train) → fitted Pipeline.
        n_trials:          Number of Optuna trials.

    Returns:
        Best hyperparameters dict.
    """
    # Combine train + val for CV (never touch test)
    X_trainval = pd.concat([splits.X_train, splits.X_val], ignore_index=True)
    y_trainval = pd.concat([splits.y_train, splits.y_val], ignore_index=True)

    logger.info(
        "Starting Optuna tuning | algorithm=%s | trials=%d | cv_folds=%d | n=%d",
        algorithm, n_trials, N_CV_FOLDS, len(X_trainval),
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name=f"churn_{algorithm}",
    )
    study.optimize(
        lambda trial: objective(trial, algorithm, X_trainval, y_trainval, build_pipeline_fn),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info(
        "Best trial: PR-AUC=%.4f | params=%s",
        study.best_value, best,
    )
    return best
