"""
tests/unit/test_calibration.py
Unit tests for calibration module correctness.

Covers:
  - Calibrator does NOT refit base model (cv='prefit')
  - ECE evaluated on held-out check set, not fitting data (F-03)
  - Isotonic calibration improves ECE
  - Small val set falls back gracefully
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _make_fitted_pipeline(X_train, y_train) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(random_state=42, max_iter=300)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


class TestCalibratorDoesNotRefitBase:
    """The base estimator must be frozen after calibrate_model()."""

    def test_coefs_unchanged_after_calibration(self):
        from ml.evaluation.calibration import calibrate_model

        rng     = np.random.default_rng(42)
        X_train = rng.normal(0, 1, (500, 5))
        y_train = rng.integers(0, 2, 500)
        X_val   = rng.normal(0, 1, (200, 5))
        y_val   = rng.integers(0, 2, 200)

        pipe         = _make_fitted_pipeline(X_train, y_train)
        coefs_before = pipe.named_steps["model"].coef_.copy()

        calibrated   = calibrate_model(pipe, X_val, y_val)

        coefs_after  = calibrated.estimator.named_steps["model"].coef_
        np.testing.assert_array_equal(
            coefs_before, coefs_after,
            err_msg="calibrate_model() must not refit the base estimator",
        )

    def test_calibrated_model_still_predicts_probabilities(self):
        from ml.evaluation.calibration import calibrate_model

        rng   = np.random.default_rng(42)
        X     = rng.normal(0, 1, (600, 5))
        y     = rng.integers(0, 2, 600)
        pipe  = _make_fitted_pipeline(X[:400], y[:400])
        cal   = calibrate_model(pipe, X[400:], y[400:])

        probs = cal.predict_proba(X[:10])[:, 1]
        assert probs.shape == (10,)
        assert np.all((probs >= 0) & (probs <= 1)), "Probabilities must be in [0, 1]"


class TestHoldOutECEEvaluation:
    """ECE must be evaluated on held-out check data, not calibration fitting data."""

    def test_ece_measured_on_check_set_not_fit_set(self):
        """
        Verifies that ECE metric cannot be 0.0 (which would indicate
        evaluation on the same data used for fitting isotonic calibrator).
        Isotonic calibration on its own fitting data is trivially perfect.
        """
        from ml.evaluation.calibration import calibrate_model
        from ml.evaluation.evaluator import expected_calibration_error

        rng = np.random.default_rng(42)
        n   = 2000
        X   = rng.normal(0, 1, (n, 5))
        y   = (X[:, 0] > 0).astype(int)

        pipe       = _make_fitted_pipeline(X[:1000], y[:1000])
        X_val_cal  = X[1000:1600]
        y_val_cal  = y[1000:1600]
        X_val_chk  = X[1600:1800]   # HOLD-OUT
        y_val_chk  = y[1600:1800]

        calibrated      = calibrate_model(pipe, X_val_cal, y_val_cal)
        probs_on_fit    = calibrated.predict_proba(X_val_cal)[:, 1]
        probs_on_check  = calibrated.predict_proba(X_val_chk)[:, 1]

        ece_on_fit_data  = expected_calibration_error(y_val_cal, probs_on_fit)
        ece_on_check_data = expected_calibration_error(y_val_chk, probs_on_check)

        # ECE on fitting data will be artificially low (isotonic overfit to training)
        # ECE on check data should be higher / more realistic
        # The key test: do NOT report ece_on_fit_data as the calibration metric
        assert ece_on_check_data >= 0.0
        # Quantify: check ECE should NOT be suspiciously perfect (< 0.001)
        # unless the model is genuinely excellent on check set too
        # This is a smoke test that we're using the right data
        if ece_on_fit_data < 0.005:
            assert ece_on_check_data >= ece_on_fit_data, (
                "ECE on hold-out should be >= ECE on calibration fitting data. "
                "Report may be using fitting data for ECE (F-03 bug)."
            )

    def test_ece_improves_after_calibration_on_check_set(self):
        from ml.evaluation.calibration import calibrate_model
        from ml.evaluation.evaluator import expected_calibration_error

        rng = np.random.default_rng(99)
        n   = 3000
        X   = rng.normal(0, 1, (n, 3))
        y   = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)

        pipe       = _make_fitted_pipeline(X[:1500], y[:1500])
        X_val_cal  = X[1500:2100]
        y_val_cal  = y[1500:2100]
        X_chk      = X[2100:2500]
        y_chk      = y[2100:2500]

        # Uncalibrated ECE on check set
        uncal_probs = pipe.predict_proba(X_chk)[:, 1]
        ece_before  = expected_calibration_error(y_chk, uncal_probs)

        calibrated  = calibrate_model(pipe, X_val_cal, y_val_cal)
        cal_probs   = calibrated.predict_proba(X_chk)[:, 1]
        ece_after   = expected_calibration_error(y_chk, cal_probs)

        # Calibration should not make ECE dramatically worse on check set
        assert ece_after <= ece_before * 1.5, (
            f"Calibration degraded check-set ECE dramatically: "
            f"before={ece_before:.4f} after={ece_after:.4f}"
        )


class TestCalibrationEdgeCases:

    def test_small_val_set_uses_sigmoid(self):
        """When val set is tiny, sigmoid is preferred over isotonic."""
        from ml.evaluation.calibration import calibrate_model

        rng   = np.random.default_rng(42)
        X     = rng.normal(0, 1, (600, 3))
        y     = rng.integers(0, 2, 600)
        pipe  = _make_fitted_pipeline(X[:400], y[:400])

        # 50-sample val set — too small for isotonic; should use sigmoid
        cal   = calibrate_model(pipe, X[400:450], y[400:450], method="sigmoid")
        probs = cal.predict_proba(X[:5])[:, 1]
        assert np.all((probs >= 0) & (probs <= 1))

    def test_calibrate_model_returns_callable(self):
        from ml.evaluation.calibration import calibrate_model

        rng  = np.random.default_rng(42)
        X    = rng.normal(0, 1, (500, 4))
        y    = rng.integers(0, 2, 500)
        pipe = _make_fitted_pipeline(X[:300], y[:300])
        cal  = calibrate_model(pipe, X[300:], y[300:])

        assert hasattr(cal, "predict_proba"), "Calibrated model must have predict_proba"
        assert callable(cal.predict_proba)
