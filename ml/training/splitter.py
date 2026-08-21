"""
ml/training/splitter.py
Temporal train/val/test split — NO random shuffling across time boundaries.
Splits are purely based on observation_date to prevent future leakage.

Strategy: sort by observation_date, then allocate first 70% as train,
next 15% as val, remaining 15% as test. No hardcoded dates — works for
any data time range automatically.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

# Default split proportions (override per-call if needed)
DEFAULT_TRAIN_PCT = 0.70
DEFAULT_VAL_PCT   = 0.15
# test_pct = 1 - 0.70 - 0.15 = 0.15


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val:   pd.DataFrame
    y_val:   pd.Series
    X_test:  pd.DataFrame
    y_test:  pd.Series
    train_cutoff: date
    val_cutoff:   date
    test_cutoff:  date

    @property
    def train_dataset_hash(self) -> str:
        return _df_hash(self.X_train, self.y_train)

    @property
    def val_dataset_hash(self) -> str:
        return _df_hash(self.X_val, self.y_val)

    @property
    def test_dataset_hash(self) -> str:
        return _df_hash(self.X_test, self.y_test)


def _df_hash(X: pd.DataFrame, y: pd.Series) -> str:
    """SHA-256 hash of the combined dataset (shape + content fingerprint)."""
    combined = pd.concat([X, y.rename("label")], axis=1)
    payload = json.dumps({
        "shape":    combined.shape,
        "columns":  list(combined.columns),
        "checksum": int(pd.util.hash_pandas_object(combined).sum()),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def temporal_split(
    df: pd.DataFrame,
    label_col: str = "churned",
    obs_date_col: str = "observation_date",
    feature_cols: list[str] | None = None,
    train_pct: float = DEFAULT_TRAIN_PCT,
    val_pct:   float = DEFAULT_VAL_PCT,
) -> DataSplits:
    """
    Split dataset by observation_date using dynamic percentage thresholds.

    Replaces hardcoded 2023 cutoff dates — works for any data time range.

    Rules:
    - train : first train_pct rows (sorted by observation_date)
    - val   : next  val_pct   rows
    - test  : remaining rows   (1 - train_pct - val_pct)

    WARNING: The test split must only be evaluated ONCE, after champion
             model selection on the val set.
    """
    if not (0 < train_pct < 1 and 0 < val_pct < 1):
        raise ValueError(
            f"train_pct={train_pct} and val_pct={val_pct} must both be in (0, 1)"
        )
    if train_pct + val_pct >= 1.0:
        raise ValueError(
            f"train_pct + val_pct = {train_pct + val_pct:.2f} >= 1.0; "
            "no samples remain for test set."
        )

    df = df.copy()
    df[obs_date_col] = pd.to_datetime(df[obs_date_col]).dt.date

    # Chronological sort — NEVER random shuffle
    df_sorted = df.sort_values(obs_date_col, kind="stable").reset_index(drop=True)
    n = len(df_sorted)

    if n < 30:
        raise ValueError(
            f"Dataset has only {n} rows — too small for a 3-way temporal split. "
            "Need at least 30 rows."
        )

    n_train = int(n * train_pct)
    n_val   = int(n * val_pct)
    # test = remainder (avoids floating-point rounding loss)
    n_test  = n - n_train - n_val

    for split_name, split_n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        if split_n < 10:
            logger.warning(
                "Small split '%s': n=%d rows. "
                "Consider collecting more data or adjusting pct params.",
                split_name, split_n,
            )

    df_train = df_sorted.iloc[:n_train]
    df_val   = df_sorted.iloc[n_train : n_train + n_val]
    df_test  = df_sorted.iloc[n_train + n_val:]

    train_cutoff = df_train[obs_date_col].max()
    val_cutoff   = df_val[obs_date_col].max()  if len(df_val)  > 0 else train_cutoff
    test_cutoff  = df_test[obs_date_col].max() if len(df_test) > 0 else val_cutoff

    if feature_cols is None:
        exclude = {label_col, obs_date_col, "customer_id", "snapshot_id",
                   "prediction_horizon_days", "feature_set_version", "created_at"}
        feature_cols = [c for c in df_sorted.columns if c not in exclude]

    for name, split, cutoff in [
        ("train", df_train, train_cutoff),
        ("val",   df_val,   val_cutoff),
        ("test",  df_test,  test_cutoff),
    ]:
        churn_rate = split[label_col].mean() if len(split) > 0 else 0.0
        logger.info(
            "Split '%s' | n=%d | cutoff=%s | churn_rate=%.3f",
            name, len(split), cutoff, churn_rate,
        )

    return DataSplits(
        X_train=df_train[feature_cols].reset_index(drop=True),
        y_train=df_train[label_col].reset_index(drop=True),
        X_val=df_val[feature_cols].reset_index(drop=True),
        y_val=df_val[label_col].reset_index(drop=True),
        X_test=df_test[feature_cols].reset_index(drop=True),
        y_test=df_test[label_col].reset_index(drop=True),
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
        test_cutoff=test_cutoff,
    )
