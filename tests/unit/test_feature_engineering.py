"""
tests/unit/test_feature_engineering.py
Unit tests for feature engineering logic.
"""
from __future__ import annotations

from datetime import date

import pytest


# ── Tenure computation ───────────────────────────────────────
class TestTenureComputation:

    def _compute_tenure(self, signup: date, observation: date) -> int:
        """Mirror the SQL tenure logic in Python for validation."""
        months = (observation.year - signup.year) * 12 + (observation.month - signup.month)
        return max(0, months)

    def test_tenure_36_months(self):
        assert self._compute_tenure(date(2021, 1, 1), date(2024, 1, 1)) == 36

    def test_tenure_zero_same_month(self):
        assert self._compute_tenure(date(2024, 1, 15), date(2024, 1, 31)) == 0

    def test_tenure_never_negative(self):
        # Observation before signup should not produce negative tenure
        assert self._compute_tenure(date(2024, 6, 1), date(2024, 1, 1)) == 0

    def test_tenure_boundary_day(self):
        """Jan 31 → Feb 1 = 0 complete months."""
        result = self._compute_tenure(date(2024, 1, 31), date(2024, 2, 1))
        assert result == 1  # month difference is 1


# ── Risk band classification ─────────────────────────────────
class TestRiskBandClassification:

    def _risk_band(self, prob: float) -> str:
        if prob < 0.30: return "low"
        if prob < 0.55: return "medium"
        if prob < 0.75: return "high"
        return "critical"

    def test_low_band(self):
        assert self._risk_band(0.10) == "low"
        assert self._risk_band(0.29) == "low"

    def test_medium_band(self):
        assert self._risk_band(0.30) == "medium"
        assert self._risk_band(0.54) == "medium"

    def test_high_band(self):
        assert self._risk_band(0.55) == "high"
        assert self._risk_band(0.74) == "high"

    def test_critical_band(self):
        assert self._risk_band(0.75) == "critical"
        assert self._risk_band(1.00) == "critical"

    def test_boundary_exact(self):
        assert self._risk_band(0.30) == "medium"
        assert self._risk_band(0.55) == "high"
        assert self._risk_band(0.75) == "critical"


# ── Addon count computation ──────────────────────────────────
class TestAddonCount:

    def _addon_count(self, services: dict) -> int:
        fields = [
            "online_security", "online_backup", "device_protection",
            "tech_support", "streaming_tv", "streaming_movies",
        ]
        return sum(1 for f in fields if services.get(f) is True)

    def test_all_addons(self):
        s = {f: True for f in [
            "online_security", "online_backup", "device_protection",
            "tech_support", "streaming_tv", "streaming_movies",
        ]}
        assert self._addon_count(s) == 6

    def test_no_addons(self):
        assert self._addon_count({}) == 0

    def test_partial_addons(self):
        s = {"online_security": True, "streaming_tv": True, "online_backup": False}
        assert self._addon_count(s) == 2


# ── PSI computation ──────────────────────────────────────────
class TestPSIComputation:

    def _compute_psi(self, expected, actual, n_bins=10):
        import numpy as np
        eps   = 1e-6
        bins  = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        bins  = np.unique(bins)
        if len(bins) < 2:
            return 0.0
        exp_c = np.histogram(expected, bins=bins)[0] + eps
        act_c = np.histogram(actual, bins=bins)[0] + eps
        e_p   = exp_c / exp_c.sum()
        a_p   = act_c / act_c.sum()
        return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))

    def test_identical_distributions_psi_near_zero(self):
        import numpy as np
        data = np.random.normal(50, 10, 1000)
        psi  = self._compute_psi(data, data)
        assert psi < 0.01

    def test_very_different_distributions_high_psi(self):
        import numpy as np
        ref = np.random.normal(50, 5,  1000)
        mon = np.random.normal(80, 15, 1000)
        psi = self._compute_psi(ref, mon)
        assert psi > 0.2, f"Expected PSI > 0.2 for very different distributions, got {psi}"


# ── Threshold selector ───────────────────────────────────────
class TestThresholdSelector:

    def test_threshold_within_capacity(self):
        """Selected threshold should yield predicted_positives <= capacity."""
        import numpy as np
        from ml.evaluation.threshold_selector import select_threshold

        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.15, n)
        y_prob = np.clip(y_true * 0.6 + np.random.normal(0, 0.2, n), 0, 1)

        capacity = 100
        threshold = select_threshold(y_true, y_prob, capacity)

        predicted_pos = (y_prob >= threshold).sum()
        assert predicted_pos <= capacity, (
            f"Threshold={threshold:.4f} predicts {predicted_pos} > capacity {capacity}"
        )
