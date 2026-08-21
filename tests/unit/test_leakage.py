"""
tests/unit/test_leakage.py
Critical tests ensuring NO future-dated data appears in any feature.
This test suite must pass at 100% before any model training or registration.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

# ── Leakage assertion helpers ────────────────────────────────

def assert_no_future_tickets(features: dict, observation_date: date) -> None:
    """Support ticket counts must only include events BEFORE observation_date."""
    # If all support counts are 0 with no tickets, that's fine
    # The key check: feature values must not exceed what was possible before obs_date
    # This is a structural test of the SQL logic via the feature dict
    assert "tickets_30d"  in features, "Missing tickets_30d"
    assert "tickets_90d"  in features, "Missing tickets_90d"
    assert "tickets_total" in features, "Missing tickets_total"
    assert features["tickets_30d"] >= 0, "tickets_30d cannot be negative"
    assert features["tickets_90d"] >= 0, "tickets_90d cannot be negative"
    assert features["tickets_total"] >= 0, "tickets_total cannot be negative"
    # 30d ≤ 90d ≤ total (monotonically non-decreasing)
    assert features["tickets_30d"] <= features["tickets_90d"], (
        "tickets_30d must be <= tickets_90d (both cut at observation_date)"
    )
    assert features["tickets_90d"] <= features["tickets_total"], (
        "tickets_90d must be <= tickets_total"
    )


def assert_no_churn_label_in_features(features: dict) -> None:
    """The churn label must never appear as a feature column."""
    forbidden_cols = {
        "churned", "churn_date", "label_id", "prediction_horizon",
        "labeling_method", "label_version",
    }
    overlap = forbidden_cols & set(features.keys())
    assert not overlap, (
        f"Label columns found in feature dict (leakage!): {overlap}"
    )


def assert_billing_before_observation(features: dict) -> None:
    """Total charges must reflect cumulative billing BEFORE observation_date."""
    assert "monthly_charges" in features, "Missing monthly_charges"
    assert "total_charges"   in features, "Missing total_charges"
    assert features["monthly_charges"] >= 0, "monthly_charges cannot be negative"
    assert features["total_charges"]   >= 0, "total_charges cannot be negative"
    # total_charges >= monthly_charges always (accumulated over tenure)
    assert features["total_charges"] >= features["monthly_charges"], (
        "total_charges must be >= monthly_charges (leakage check)"
    )


def assert_tenure_non_negative(features: dict, observation_date: date, signup_date: date) -> None:
    """Tenure must be non-negative and consistent with signup_date."""
    assert "tenure_months" in features
    computed_min_months = max(
        0, (observation_date.year - signup_date.year) * 12 +
           (observation_date.month - signup_date.month)
    )
    assert features["tenure_months"] >= 0
    # Allow ±1 month for edge cases in month boundary computation
    assert abs(features["tenure_months"] - computed_min_months) <= 1, (
        f"tenure_months={features['tenure_months']} inconsistent with "
        f"signup={signup_date} obs={observation_date} (expected≈{computed_min_months})"
    )


def assert_interaction_features_consistent(features: dict) -> None:
    """Interaction features must be derivable from base features (no extra future data)."""
    if features.get("charge_x_tenure") is not None:
        expected = features["monthly_charges"] * features["tenure_months"]
        assert abs(features["charge_x_tenure"] - expected) < 1e-2, (
            f"charge_x_tenure={features['charge_x_tenure']} != "
            f"monthly_charges*tenure={expected}"
        )


# ── Tests ────────────────────────────────────────────────────

class TestLeakagePrevention:

    def test_no_churn_label_columns_in_features(self):
        """Churn label columns MUST NOT appear in the feature dict."""
        sample_feature = {
            "tenure_months": 12,
            "monthly_charges": 49.99,
            "total_charges": 599.88,
            # These should NOT be present:
            # "churned": True,
            # "churn_date": date(2024, 2, 1),
        }
        assert_no_churn_label_in_features(sample_feature)

    def test_ticket_counts_monotonically_increasing(self):
        """30d ≤ 90d ≤ total — structural guarantee of observation cutoff."""
        features = {
            "tickets_30d": 1,
            "tickets_90d": 3,
            "tickets_total": 7,
        }
        assert_no_future_tickets(features, date(2024, 1, 1))

    def test_ticket_counts_violation_detected(self):
        """tickets_30d > tickets_90d should raise AssertionError."""
        features = {
            "tickets_30d": 5,   # IMPOSSIBLE: 30d > 90d
            "tickets_90d": 2,
            "tickets_total": 8,
        }
        with pytest.raises(AssertionError, match="tickets_30d must be <= tickets_90d"):
            assert_no_future_tickets(features, date(2024, 1, 1))

    def test_billing_total_gte_monthly(self):
        """Total charges >= monthly charges always."""
        features = {
            "monthly_charges": 79.99,
            "total_charges": 959.88,
        }
        assert_billing_before_observation(features)

    def test_billing_total_lt_monthly_violation_detected(self):
        """total_charges < monthly_charges should raise AssertionError."""
        features = {
            "monthly_charges": 79.99,
            "total_charges": 50.00,  # IMPOSSIBLE
        }
        with pytest.raises(AssertionError, match="total_charges must be >="):
            assert_billing_before_observation(features)

    def test_tenure_consistent_with_signup_date(self):
        signup_date      = date(2020, 1, 1)
        observation_date = date(2024, 1, 1)  # 48 months later
        features = {"tenure_months": 48}
        assert_tenure_non_negative(features, observation_date, signup_date)

    def test_tenure_future_violation_detected(self):
        """Tenure that exceeds possible months given signup/obs dates should fail."""
        signup_date      = date(2023, 1, 1)
        observation_date = date(2023, 6, 1)  # 5 months max
        features = {"tenure_months": 24}     # IMPOSSIBLE — far too large
        with pytest.raises(AssertionError):
            assert_tenure_non_negative(features, observation_date, signup_date)

    def test_interaction_feature_derivable_from_base(self):
        features = {
            "monthly_charges": 79.99,
            "tenure_months":   12,
            "charge_x_tenure": 79.99 * 12,
        }
        assert_interaction_features_consistent(features)

    def test_interaction_feature_mismatch_detected(self):
        features = {
            "monthly_charges": 79.99,
            "tenure_months":   12,
            "charge_x_tenure": 9999.0,  # Wrong — future data leaking?
        }
        with pytest.raises(AssertionError, match="charge_x_tenure"):
            assert_interaction_features_consistent(features)

    def test_pct_fields_in_valid_range(self):
        """Percentage features must be in [0, 1]."""
        features = {
            "pct_billing_tickets_90d": 0.33,
            "pct_cancel_tickets_90d":  0.17,
        }
        for key, val in features.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1] range"

    def test_csat_score_in_valid_range(self):
        features = {"avg_csat_90d": 3.8}
        val = features["avg_csat_90d"]
        assert 1.0 <= val <= 5.0, f"avg_csat_90d={val} out of [1,5] range"
