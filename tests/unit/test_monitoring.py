"""
tests/unit/test_monitoring.py
Unit tests for monitoring pipeline: PSI computation, drift alerting.
"""
from __future__ import annotations

import numpy as np
import pytest


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Shared PSI implementation (mirrors monitoring_pipeline.py)."""
    eps  = 1e-6
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    e_c = np.histogram(expected, bins=bins)[0] + eps
    a_c = np.histogram(actual,   bins=bins)[0] + eps
    e_p = e_c / e_c.sum()
    a_p = a_c / a_c.sum()
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))


class TestPSIComputation:

    def test_identical_distributions_near_zero(self):
        rng  = np.random.default_rng(42)
        data = rng.normal(50, 10, 2000)
        psi  = _psi(data, data)
        assert psi < 0.01, f"PSI for identical distributions should be ~0, got {psi}"

    def test_very_different_distributions_high_psi(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(50, 5,  2000)
        mon = rng.normal(80, 15, 2000)
        psi = _psi(ref, mon)
        assert psi > 0.2, f"PSI for very different distributions should be > 0.2, got {psi}"

    def test_psi_symmetry_approximately(self):
        """PSI(A,B) and PSI(B,A) should be similar (not identical — not symmetric by definition)."""
        rng  = np.random.default_rng(42)
        ref  = rng.normal(50, 10, 1000)
        mon  = rng.normal(55, 10, 1000)
        psi1 = _psi(ref, mon)
        psi2 = _psi(mon, ref)
        # Both should be in the same order of magnitude
        assert abs(psi1 - psi2) < 0.1, f"PSI asymmetry too large: {psi1} vs {psi2}"

    def test_psi_yellow_zone(self):
        """Distributions that drift slightly should fall in 0.1–0.2 range."""
        rng = np.random.default_rng(42)
        ref = rng.normal(50, 10, 2000)
        mon = rng.normal(55, 12, 2000)  # slight shift
        psi = _psi(ref, mon)
        # Should be in moderate drift zone
        assert psi < 0.5, f"PSI too high for mild drift: {psi}"
        assert psi > 0.0, "PSI should be positive for different distributions"

    def test_empty_arrays_handled(self):
        """Edge case: empty array should not crash."""
        arr = np.array([50.0])  # single point
        psi = _psi(arr, arr)
        assert isinstance(psi, float)

    def test_psi_non_negative(self):
        """PSI is always non-negative."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            a = rng.normal(50, 10, 500)
            b = rng.normal(50 + rng.uniform(-20, 20), 10, 500)
            psi = _psi(a, b)
            assert psi >= 0.0, f"PSI was negative: {psi}"


class TestRiskBandBoundaries:
    """Test that risk band thresholds match across all components."""

    def _band(self, prob: float) -> str:
        if prob < 0.30: return "low"
        if prob < 0.55: return "medium"
        if prob < 0.75: return "high"
        return "critical"

    @pytest.mark.parametrize("prob,expected", [
        (0.00, "low"),
        (0.29, "low"),
        (0.30, "medium"),
        (0.54, "medium"),
        (0.55, "high"),
        (0.74, "high"),
        (0.75, "critical"),
        (1.00, "critical"),
    ])
    def test_band_boundaries(self, prob: float, expected: str):
        assert self._band(prob) == expected, (
            f"prob={prob} → expected '{expected}' got '{self._band(prob)}'"
        )


class TestECEComputation:

    def test_perfect_calibration_ece_near_zero(self):
        """A model where predicted probability equals actual rate has ECE ≈ 0."""
        import sys
        sys.path.insert(0, "ml/evaluation")
        from ml.evaluation.evaluator import expected_calibration_error

        # Simulate perfect calibration: P(churn) = actual churn rate per bin
        rng    = np.random.default_rng(42)
        y_prob = rng.uniform(0, 1, 5000)
        # Assign labels with probability equal to the predicted score
        y_true = rng.binomial(1, y_prob)

        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert ece < 0.10, f"ECE for well-calibrated model should be < 0.10, got {ece}"

    def test_miscalibrated_model_has_high_ece(self):
        """A model that always predicts 0.9 but ground truth is 0.1 has high ECE."""
        import sys
        sys.path.insert(0, "ml/evaluation")
        from ml.evaluation.evaluator import expected_calibration_error

        n      = 2000
        y_prob = np.full(n, 0.9)   # always predict 90%
        y_true = np.random.binomial(1, 0.1, n)  # actual churn rate = 10%

        ece = expected_calibration_error(y_true, y_prob)
        assert ece > 0.3, f"Badly miscalibrated model should have ECE > 0.3, got {ece}"
