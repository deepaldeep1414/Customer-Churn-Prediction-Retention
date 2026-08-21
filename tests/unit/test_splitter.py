"""
tests/unit/test_splitter.py
Unit tests for the dynamic temporal split logic (F-14).

Covers:
  - No hardcoded dates — works for any time range
  - Chronological ordering preserved
  - Correct proportions (70/15/15)
  - Edge cases: tiny dataset, wrong pct args
  - No data leakage across splits (train cutoff < val start)
  - Dataset hashes are deterministic and non-equal across splits
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from datetime import date, timedelta


def _make_df(n: int = 200, start_date: date = date(2022, 1, 1)) -> pd.DataFrame:
    """Create a synthetic feature+label DataFrame with sequential dates."""
    rng   = np.random.default_rng(42)
    dates = [start_date + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "observation_date": dates,
        "churned":          rng.integers(0, 2, n),
        "tenure_months":    rng.integers(1, 60, n),
        "monthly_charges":  rng.uniform(20, 120, n),
    })


class TestDynamicTemporalSplit:

    def test_split_proportions_approximately_correct(self):
        from ml.training.splitter import temporal_split
        df     = _make_df(200)
        splits = temporal_split(df, train_pct=0.70, val_pct=0.15)

        n = len(df)
        assert abs(len(splits.X_train) - int(n * 0.70)) <= 1
        assert abs(len(splits.X_val)   - int(n * 0.15)) <= 1
        # Test gets the remainder
        assert len(splits.X_test) == n - len(splits.X_train) - len(splits.X_val)

    def test_chronological_order_preserved(self):
        """Train observations must all be before val, val before test."""
        from ml.training.splitter import temporal_split
        df     = _make_df(300)
        splits = temporal_split(df)

        assert splits.train_cutoff <= splits.val_cutoff
        assert splits.val_cutoff   <= splits.test_cutoff

    def test_no_date_overlap_between_splits(self):
        """No observation_date in val should be <= train_cutoff."""
        from ml.training.splitter import temporal_split
        df     = _make_df(200)
        splits = temporal_split(df)

        df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
        df_sorted = df.sort_values("observation_date").reset_index(drop=True)

        n_train = len(splits.X_train)
        n_val   = len(splits.X_val)

        train_max = df_sorted["observation_date"].iloc[n_train - 1]
        val_min   = df_sorted["observation_date"].iloc[n_train]
        val_max   = df_sorted["observation_date"].iloc[n_train + n_val - 1]
        test_min  = df_sorted["observation_date"].iloc[n_train + n_val]

        assert train_max <= val_min,  f"Train/val overlap: train_max={train_max} val_min={val_min}"
        assert val_max   <= test_min, f"Val/test overlap: val_max={val_max} test_min={test_min}"

    def test_works_for_various_date_ranges(self):
        """Must not assume any specific year — 2020, 2023, 2025 all work."""
        from ml.training.splitter import temporal_split

        for start_year in [2019, 2022, 2025]:
            df     = _make_df(150, start_date=date(start_year, 1, 1))
            splits = temporal_split(df)
            assert len(splits.X_train) > 0
            assert len(splits.X_val)   > 0
            assert len(splits.X_test)  > 0

    def test_feature_columns_exclude_metadata(self):
        """observation_date, customer_id etc. must not appear in X_train."""
        from ml.training.splitter import temporal_split
        df = _make_df(200)
        df["customer_id"]     = [f"C{i}" for i in range(len(df))]
        df["snapshot_id"]     = range(len(df))
        df["created_at"]      = pd.Timestamp("2024-01-01")
        splits = temporal_split(df)

        forbidden = {"observation_date", "customer_id", "snapshot_id", "created_at"}
        overlap   = forbidden & set(splits.X_train.columns)
        assert not overlap, f"Metadata columns leaked into X_train: {overlap}"

    def test_label_col_not_in_features(self):
        from ml.training.splitter import temporal_split
        df     = _make_df(200)
        splits = temporal_split(df)
        assert "churned" not in splits.X_train.columns

    def test_dataset_hashes_are_reproducible(self):
        """Same input DataFrame → same hash."""
        from ml.training.splitter import temporal_split
        df      = _make_df(200)
        splits1 = temporal_split(df)
        splits2 = temporal_split(df)
        assert splits1.train_dataset_hash == splits2.train_dataset_hash
        assert splits1.val_dataset_hash   == splits2.val_dataset_hash

    def test_train_val_test_hashes_are_distinct(self):
        """Different splits must have different hashes."""
        from ml.training.splitter import temporal_split
        df     = _make_df(300)
        splits = temporal_split(df)
        hashes = {splits.train_dataset_hash, splits.val_dataset_hash, splits.test_dataset_hash}
        assert len(hashes) == 3, "All 3 split hashes must be distinct"


class TestSplitterEdgeCases:

    def test_raises_on_tiny_dataset(self):
        from ml.training.splitter import temporal_split
        df = _make_df(n=10)   # < 30 rows
        with pytest.raises(ValueError, match="too small"):
            temporal_split(df)

    def test_raises_on_invalid_pct_sum(self):
        from ml.training.splitter import temporal_split
        df = _make_df(300)
        with pytest.raises(ValueError, match="no samples remain"):
            temporal_split(df, train_pct=0.70, val_pct=0.40)  # sum > 1

    def test_raises_on_zero_train_pct(self):
        from ml.training.splitter import temporal_split
        df = _make_df(300)
        with pytest.raises(ValueError):
            temporal_split(df, train_pct=0.0, val_pct=0.15)

    def test_custom_proportions_respected(self):
        from ml.training.splitter import temporal_split
        df     = _make_df(1000)
        splits = temporal_split(df, train_pct=0.80, val_pct=0.10)
        n      = len(df)
        assert abs(len(splits.X_train) - int(n * 0.80)) <= 1
        assert abs(len(splits.X_val)   - int(n * 0.10)) <= 1

    def test_class_balance_logged_per_split(self, caplog):
        """Churn rate should be logged for each split."""
        import logging
        from ml.training.splitter import temporal_split

        df = _make_df(300)
        with caplog.at_level(logging.INFO, logger="ml.training.splitter"):
            temporal_split(df)

        logged = caplog.text
        assert "train" in logged
        assert "val"   in logged
        assert "test"  in logged
